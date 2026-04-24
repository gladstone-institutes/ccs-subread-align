"""Base composition calculation from subread-to-CCS alignments."""

import logging
from collections import defaultdict
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from ccs_subread_align._pool import get_pool
from ccs_subread_align.alignment import parse_cigar_to_reference_map

logger = logging.getLogger(__name__)

_BASE_TO_IDX = {"A": 0, "T": 1, "C": 2, "G": 3, "N": 4}
_BASE_CATEGORIES = ["A", "T", "C", "G", "N"]
_STRAND_CATEGORIES = ["fwd", "rev"]

_BASE_LUT = np.full(256, 4, dtype=np.int8)
for _b, _i in _BASE_TO_IDX.items():
    _BASE_LUT[ord(_b)] = _i

_FLUSH_ROWS = 1_000_000


def calculate_base_composition(
    ccs_read: Dict,
    assigned_subreads: List[Dict],
    ref_seq: str,
    chrM_length: int = 16569,
) -> pd.DataFrame:
    """
    Calculate per-position base composition from subreads aligned to a CCS read.

    For each CCS query position, counts how many subreads have each base (A/T/C/G/N)
    at the corresponding reference position, and computes the fraction agreeing with
    the CCS base call.

    Args:
        ccs_read: CCS read dictionary (from `io.stream_ccs_reads`) with keys
            including query_sequence, query_length, quality_array, zmw,
            strand, zmw_strand. If ``query_to_ref`` is absent, it is
            computed on the fly from ``cigartuples``, ``reference_start``
            and ``query_length`` so the parent process never holds the
            parsed arrays (~68 kB per CCS at full scale).
        assigned_subreads: List of assigned subread dicts (from process_subread_alignment)
            for this CCS read's (zmw, strand). Each must have aligned_sequence and
            position_map.
        ref_seq: Reference sequence for this chromosome.
        chrM_length: Genome length for coordinate normalization.

    Returns:
        DataFrame with one row per CCS position and columns: zmw, strand, zmw_strand,
        ccs_pos, ref_pos, ccs_base, reference_base, q_score, A_count, T_count,
        C_count, G_count, N_count, total_subreads, agreement_fraction.
    """
    ccs_len = ccs_read["query_length"]
    ccs_to_ref = ccs_read.get("query_to_ref")
    if ccs_to_ref is None:
        ccs_to_ref = parse_cigar_to_reference_map(
            ccs_read["cigartuples"],
            ccs_read["reference_start"],
            ccs_len,
            chrM_length,
        )
    ccs_seq = ccs_read["query_sequence"]

    base_counts = np.zeros((ccs_len, 5), dtype=np.int32)

    # Build ref_pos -> [ccs_pos, ...] once per CCS so the subread loop is O(1) per base.
    ref_to_ccs: Dict[int, List[int]] = defaultdict(list)
    for ccs_pos, ref_pos in enumerate(ccs_to_ref):
        if ref_pos >= 0:
            ref_to_ccs[int(ref_pos)].append(ccs_pos)

    for sr in assigned_subreads:
        sr_seq = sr["aligned_sequence"]
        position_map = sr["position_map"]
        sr_bytes = np.frombuffer(sr_seq.encode("ascii"), dtype=np.uint8)
        sr_base_idx = _BASE_LUT[sr_bytes]

        for sr_pos in range(len(sr_seq)):
            ref_pos = int(position_map[sr_pos])
            if ref_pos < 0:
                continue
            targets = ref_to_ccs.get(ref_pos)
            if not targets:
                continue
            bi = sr_base_idx[sr_pos]
            for ccs_pos in targets:
                base_counts[ccs_pos, bi] += 1

    total_subreads = base_counts.sum(axis=1)

    # Vectorized agreement counts via LUT + fancy indexing.
    ccs_bytes = np.frombuffer(ccs_seq.encode("ascii"), dtype=np.uint8)
    ccs_base_idx = _BASE_LUT[ccs_bytes]
    agreement_counts = base_counts[np.arange(ccs_len), ccs_base_idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        agreement_fraction = np.where(
            total_subreads > 0,
            agreement_counts / np.maximum(total_subreads, 1),
            0.0,
        ).astype(np.float32)

    # Vectorized reference_base lookup.
    ref_arr = np.frombuffer(ref_seq.encode("ascii"), dtype="S1")
    valid = (ccs_to_ref >= 0) & (ccs_to_ref < len(ref_seq))
    ref_base_bytes = np.full(ccs_len, b"N", dtype="S1")
    ref_base_bytes[valid] = ref_arr[ccs_to_ref[valid]]
    reference_base = pd.Categorical(
        np.char.decode(ref_base_bytes, "ascii"),
        categories=_BASE_CATEGORIES,
    )

    ccs_base = pd.Categorical(list(ccs_seq), categories=_BASE_CATEGORIES)
    strand = pd.Categorical([ccs_read["strand"]] * ccs_len, categories=_STRAND_CATEGORIES)
    zmw_strand = pd.Categorical(
        [ccs_read["zmw_strand"]] * ccs_len, categories=[ccs_read["zmw_strand"]]
    )

    df = pd.DataFrame(
        {
            "zmw": np.full(ccs_len, ccs_read["zmw"], dtype=np.int64),
            "strand": strand,
            "zmw_strand": zmw_strand,
            "ccs_pos": np.arange(ccs_len, dtype=np.int32),
            "ref_pos": np.asarray(ccs_to_ref, dtype=np.int32),
            "ccs_base": ccs_base,
            "reference_base": reference_base,
            "q_score": np.asarray(ccs_read["quality_array"], dtype=np.uint8),
            "A_count": base_counts[:, 0].astype(np.uint16),
            "T_count": base_counts[:, 1].astype(np.uint16),
            "C_count": base_counts[:, 2].astype(np.uint16),
            "G_count": base_counts[:, 3].astype(np.uint16),
            "N_count": base_counts[:, 4].astype(np.uint16),
            "total_subreads": total_subreads.astype(np.uint16),
            "agreement_fraction": agreement_fraction,
        }
    )

    return df


def _process_ccs_composition(args: Tuple) -> Optional[pd.DataFrame]:
    """Worker function for parallel base composition calculation."""
    ccs, zmw_strand_subreads, ref_seq, chrM_length = args
    if len(zmw_strand_subreads) > 0:
        return calculate_base_composition(ccs, zmw_strand_subreads, ref_seq, chrM_length)
    return None


def _iter_worker_dfs(
    work_items: Iterable[Tuple],
    n_cores: int,
) -> Iterable[Optional[pd.DataFrame]]:
    if n_cores == 1:
        for item in work_items:
            yield _process_ccs_composition(item)
    else:
        with get_pool(n_cores) as pool:
            yield from pool.imap(
                _process_ccs_composition, work_items, chunksize=10
            )


def _widen_field(f: pa.Field) -> pa.Field:
    """Promote at-risk columns to int64-offset variants before `sort_by`/`take`.

    Extend here (not inline) so any future wide column type can be added in
    one place.
    """
    t = f.type
    if pa.types.is_string(t):
        return f.with_type(pa.large_string())
    if pa.types.is_list(t):
        return f.with_type(pa.large_list(t.value_type))
    return f


def _group_subreads_from_parquet(
    path: Union[str, Path],
) -> Dict[Tuple[int, str], pa.Table]:
    """Load a streamed-alignment parquet and group rows by (zmw, strand).

    Rows are sorted once by (zmw, strand), then each group is a zero-copy
    ``table.slice(...)`` view into the sorted table. Peak memory stays at
    ~1x the parquet (transiently 2x during ``sort_by`` itself, which is
    reassigned immediately so the unsorted copy can be freed), versus 2x
    if we fanned out with per-group ``take()`` which allocates a fresh
    buffer for every group.
    """
    table = pq.read_table(str(path))
    # `sort_by` internally calls `take(sort_indices)`, which combines column
    # chunks into a single chunk. Two overflow classes surface here when
    # offsets are int32:
    #   - `string` columns: byte offsets overflow once aggregate bytes cross
    #     ~2 GB (`aligned_sequence` at scale).
    #   - `list<int32>` columns: element offsets overflow once aggregate
    #     element count crosses 2^31 (`position_map` at scale: ~8 B
    #     elements in production).
    # Promote both families to int64-offset variants. No-op on fresh files;
    # repairs legacy parquet written before the schema widening.
    new_schema = pa.schema([_widen_field(f) for f in table.schema])
    if new_schema != table.schema:
        table = table.cast(new_schema)
    # sort_by rejects dictionary-encoded columns, so decode `strand` to plain
    # string first. Strand values are 3-byte strings with cardinality 2, so the
    # decoded buffer is trivial relative to aligned_sequence.
    strand_idx = table.schema.get_field_index("strand")
    if pa.types.is_dictionary(table.schema.field(strand_idx).type):
        table = table.set_column(
            strand_idx, "strand", table.column("strand").cast(pa.string())
        )
    # Reassign so the unsorted copy is dropped as soon as sort_by returns.
    table = table.sort_by([("zmw", "ascending"), ("strand", "ascending")])
    zmws = table.column("zmw").to_numpy(zero_copy_only=False)
    strands = table.column("strand").to_pylist()
    result: Dict[Tuple[int, str], pa.Table] = {}
    n = len(zmws)
    i = 0
    while i < n:
        z, s = int(zmws[i]), strands[i]
        j = i + 1
        while j < n and int(zmws[j]) == z and strands[j] == s:
            j += 1
        result[(z, s)] = table.slice(i, j - i)
        i = j
    return result


def _pa_table_to_subread_dicts(table: pa.Table) -> List[Dict]:
    """Materialize a per-group pyarrow Table as the List[Dict] the worker expects.

    Short-lived: produced inside the work_items generator and consumed
    immediately by ``_process_ccs_composition`` in a worker process.
    """
    aligned = table.column("aligned_sequence").to_pylist()
    pos_maps = table.column("position_map").to_pylist()
    subread_names = table.column("subread_name").to_pylist()
    zmws = table.column("zmw").to_pylist()
    strands = table.column("strand").to_pylist()
    identities = table.column("identity").to_pylist()
    out: List[Dict] = []
    for i in range(table.num_rows):
        out.append(
            {
                "zmw": int(zmws[i]),
                "strand": strands[i],
                "zmw_strand": f"{int(zmws[i])}_{strands[i]}",
                "subread_name": subread_names[i],
                "aligned_sequence": aligned[i],
                "position_map": np.asarray(pos_maps[i], dtype=np.int32),
                "identity": identities[i],
            }
        )
    return out


def calculate_all_base_compositions(
    ccs_reads: Iterable[Dict],
    assigned_subreads: Union[List[Dict], str, Path],
    ref_seqs: Dict[str, str],
    zmw_to_chrom: Dict[int, str],
    chrM_length: int = 16569,
    n_cores: Optional[int] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Union[pd.DataFrame, Path]:
    """
    Calculate base composition for all CCS reads.

    Args:
        ccs_reads: Iterable of CCS read dicts. Typically the iterator
            returned by ``io.stream_ccs_reads``; a ``List[Dict]`` is also
            accepted. Iterated exactly once — the function does not
            pre-count, so tqdm shows rate progress instead of a percent.
        assigned_subreads: Either a ``List[Dict]`` of assigned subread records
            (legacy in-memory path) or a path to a parquet file written by
            ``process_subread_alignment(..., output_path=...)``. The parquet
            path is preferred at full scale: it loads as a pyarrow Table with
            tight columnar dtypes instead of a fat Python list of dicts.
        ref_seqs: Dictionary mapping chromosome names to reference sequences.
        zmw_to_chrom: Dictionary mapping ZMW to chromosome name.
        chrM_length: Genome length for coordinate normalization.
        n_cores: Number of cores for parallel processing (default: all).
        output_path: If provided, stream per-CCS results to this parquet file
            (zstd-compressed) and return the path. Avoids materializing the
            full DataFrame in memory, which OOMs on full-scale data. If None,
            return a single concatenated DataFrame (only safe at small scale).

    Returns:
        DataFrame with base composition at all positions across all CCS reads,
        or the output_path if streaming.
    """
    if n_cores is None:
        n_cores = cpu_count()

    subreads_by_zmw_strand: Dict[Tuple[int, str], Union[List[Dict], pa.Table]]
    from_parquet = isinstance(assigned_subreads, (str, Path))
    if from_parquet:
        subreads_by_zmw_strand = _group_subreads_from_parquet(assigned_subreads)
    else:
        grouped: Dict[Tuple[int, str], List[Dict]] = defaultdict(list)
        for sr in assigned_subreads:
            grouped[(sr["zmw"], sr["strand"])].append(sr)
        subreads_by_zmw_strand = grouped

    logger.info(f"{len(subreads_by_zmw_strand)} unique (zmw, strand) groups")

    def _iter_work_items() -> Iterator[Tuple]:
        for ccs in ccs_reads:
            chrom = zmw_to_chrom.get(ccs["zmw"])
            if chrom is None or chrom not in ref_seqs:
                continue
            key = (ccs["zmw"], ccs["strand"])
            group = subreads_by_zmw_strand.get(key)
            if group is None:
                matched: List[Dict] = []
            elif from_parquet:
                matched = _pa_table_to_subread_dicts(group)
            else:
                matched = group
            yield (ccs, matched, ref_seqs[chrom], chrM_length)

    desc = f"Processing CCS reads ({n_cores} cores)" if n_cores != 1 else "Processing CCS reads"
    # total=None: ccs_reads is an iterable we only walk once, so tqdm shows
    # rate + elapsed instead of a percent. Callers with an upfront count
    # can wrap their own tqdm.
    df_iter = tqdm(
        _iter_worker_dfs(_iter_work_items(), n_cores),
        total=None,
        desc=desc,
    )

    if output_path is not None:
        output_path = Path(output_path)
        writer: Optional[pq.ParquetWriter] = None
        buffer: List[pa.Table] = []
        buffered_rows = 0
        total_rows = 0
        try:
            for df in df_iter:
                if df is None:
                    continue
                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path, table.schema, compression="zstd"
                    )
                buffer.append(table)
                buffered_rows += table.num_rows
                if buffered_rows >= _FLUSH_ROWS:
                    writer.write_table(pa.concat_tables(buffer))
                    total_rows += buffered_rows
                    buffer.clear()
                    buffered_rows = 0
            if buffer:
                writer.write_table(pa.concat_tables(buffer))
                total_rows += buffered_rows
        finally:
            if writer is not None:
                writer.close()

        if writer is None:
            logger.info("No CCS reads produced composition rows; no parquet written")
        else:
            logger.info(f"Streamed composition for {total_rows:,} positions to: {output_path}")
        return output_path

    all_dfs = [df for df in df_iter if df is not None]
    if not all_dfs:
        return pd.DataFrame()

    df_all = pd.concat(all_dfs, ignore_index=True)
    # pd.concat falls back to object dtype when per-frame Categorical columns
    # have disjoint categories; restore the zmw_strand categorical so the
    # concatenated frame doesn't balloon back to an object column.
    if df_all["zmw_strand"].dtype == object:
        df_all["zmw_strand"] = df_all["zmw_strand"].astype("category")
    logger.info(f"Calculated composition for {len(df_all):,} positions")
    return df_all
