"""Base composition calculation from subread-to-CCS alignments."""

import logging
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

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
        ccs_read: CCS read dictionary (from load_ccs_reads) with keys including
            query_sequence, query_length, query_to_ref, quality_array, zmw,
            strand, zmw_strand.
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
    ccs_to_ref = ccs_read["query_to_ref"]
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
    work_items: List[Tuple],
    n_cores: int,
) -> Iterable[Optional[pd.DataFrame]]:
    if n_cores == 1:
        for item in work_items:
            yield _process_ccs_composition(item)
    else:
        with Pool(processes=n_cores) as pool:
            yield from pool.imap(
                _process_ccs_composition, work_items, chunksize=10
            )


def calculate_all_base_compositions(
    ccs_reads: List[Dict],
    assigned_subreads: List[Dict],
    ref_seqs: Dict[str, str],
    zmw_to_chrom: Dict[int, str],
    chrM_length: int = 16569,
    n_cores: Optional[int] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Union[pd.DataFrame, Path]:
    """
    Calculate base composition for all CCS reads.

    Args:
        ccs_reads: List of CCS read dictionaries (from load_ccs_reads).
        assigned_subreads: List of assigned subread dicts (from process_subread_alignment).
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

    subreads_by_zmw_strand: Dict[Tuple[int, str], List[Dict]] = defaultdict(list)
    for sr in assigned_subreads:
        subreads_by_zmw_strand[(sr["zmw"], sr["strand"])].append(sr)

    logger.info(f"{len(subreads_by_zmw_strand)} unique (zmw, strand) groups")

    work_items: List[Tuple] = []
    for ccs in ccs_reads:
        chrom = zmw_to_chrom.get(ccs["zmw"])
        if chrom is None or chrom not in ref_seqs:
            continue
        matched_subreads = subreads_by_zmw_strand.get((ccs["zmw"], ccs["strand"]), [])
        work_items.append((ccs, matched_subreads, ref_seqs[chrom], chrM_length))

    logger.info(
        f"Calculating base composition for {len(work_items)} CCS reads using {n_cores} cores"
    )

    desc = f"Processing CCS reads ({n_cores} cores)" if n_cores != 1 else "Processing CCS reads"
    df_iter = tqdm(
        _iter_worker_dfs(work_items, n_cores),
        total=len(work_items),
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
