"""Core alignment functions for assigning PacBio subreads to strands."""

import json
import logging
import re
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import edlib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from ccs_subread_align._pool import get_pool

logger = logging.getLogger(__name__)

# Populated in each worker via `_worker_init` when a pool is used. Keeping
# `ref_seqs` out of the per-task payload saves ~33 kB/chunk of pickle
# bandwidth and (more importantly) removes the pointer to a big dict from
# every work item, keeping the generator-fed imap lightweight.
_WORKER_REF_SEQS: Optional[Dict[str, str]] = None


def _worker_init(ref_seqs: Dict[str, str]) -> None:
    global _WORKER_REF_SEQS
    _WORKER_REF_SEQS = ref_seqs


# Row buffer flush threshold for streaming assigned-subread parquet output.
# Alignment rows carry variable-length aligned_sequence + position_map,
# so use a smaller buffer than the composition-side 1M-row value.
_ASSIGNED_SUBREAD_FLUSH_ROWS = 100_000

# Both variable-length columns use int64 offsets. `aligned_sequence` is
# `large_string` because a 100k-row flush batch can exceed 2 GB of sequence
# bytes. `position_map` is `large_list<int32>` because the aggregate element
# count across chunks can exceed int32's 2^31 cap once the reader combines
# chunks in `sort_by`/`take`; production hit this at ~476k subreads x ~17k
# positions ~= 8 B elements. Wider write schema also keeps pa.array
# construction safe at flush time if read lengths grow.
_ASSIGNED_SUBREAD_SCHEMA = pa.schema(
    [
        pa.field("zmw", pa.int64()),
        pa.field("strand", pa.dictionary(pa.int8(), pa.string())),
        pa.field("subread_name", pa.string()),
        pa.field("aligned_sequence", pa.large_string()),
        pa.field("position_map", pa.large_list(pa.int32())),
        pa.field("identity", pa.float32()),
    ]
)

_ASSIGNED_SUBREAD_SCHEMA_MARGIN = pa.schema(
    list(_ASSIGNED_SUBREAD_SCHEMA)
    + [pa.field("edit_distance_margin", pa.int32())]
)

# Edlib/SAM CIGAR string tokenizer: "(length)(op)" pairs, e.g. "100=5X10I20=".
_CIGAR_STR_RE = re.compile(r"(\d+)([MIDNSHP=X])")


def reverse_complement(seq: str) -> str:
    """Return reverse complement of DNA sequence."""
    complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(complement.get(base, "N") for base in reversed(seq))


def extract_zmw_from_name(read_name: str) -> Optional[int]:
    """Extract ZMW number from PacBio read name."""
    parts = read_name.split("/")
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def parse_cigar_to_reference_map(
    cigartuples,
    reference_start: int,
    query_length: int,
    chrM_length: int = 16569,
) -> np.ndarray:
    """
    Parse CIGAR to map query positions to normalized reference positions.

    Walks the pysam cigartuples once, writing matched reference positions
    directly into an int32 output array sized to ``query_length``.
    Insertions, soft clips, hard clips, and padding leave the corresponding
    query positions at -1. Reference positions are normalized modulo
    ``chrM_length`` to handle the circularized (doubled) reference.

    Args:
        cigartuples: List of (operation, length) tuples from pysam
        reference_start: Starting reference position
        query_length: Length of the query sequence (sets the array size)
        chrM_length: Actual mitochondrial genome length (default: 16569)

    Returns:
        np.ndarray[int32] of shape (query_length,) with normalized reference
        positions for matched bases and -1 for unmatched positions.
    """
    out = np.full(query_length, -1, dtype=np.int32)
    if not cigartuples:
        return out

    q = 0  # query cursor
    r = reference_start  # reference cursor
    for op, length in cigartuples:
        if op == 0 or op == 7 or op == 8:  # M / = / X: consume query + ref
            end = q + length
            if end > query_length:
                end = query_length
            seg = end - q
            if seg > 0:
                out[q:end] = (np.arange(r, r + seg) % chrM_length).astype(np.int32)
            q += length
            r += length
        elif op == 1 or op == 4:  # I / S: consume query only
            q += length
        elif op == 2 or op == 3:  # D / N: consume ref only
            r += length
        # 5 (H), 6 (P): consume neither

    return out


def parse_edlib_cigar_to_positions(
    cigar: str, query_seq: str, ref_start: int, chrM_length: int = 16569
) -> np.ndarray:
    """
    Parse edlib CIGAR string to map query positions to reference positions.

    Tokenizes the CIGAR string with a regex and walks it once, writing
    matched reference positions into an int32 output array. Insertions and
    soft clips stay at -1. Normalizes positions modulo ``chrM_length`` for
    the circularized reference.

    Args:
        cigar: Edlib CIGAR string (e.g., "100=5X10I20=")
        query_seq: Query sequence string
        ref_start: Starting reference position
        chrM_length: Actual mitochondrial genome length (default: 16569)

    Returns:
        np.array: Array mapping query positions to normalized reference positions
                  (-1 for gaps/insertions)
    """
    qlen = len(query_seq)
    out = np.full(qlen, -1, dtype=np.int32)
    if not cigar:
        return out

    q = 0
    r = ref_start
    for length_str, op in _CIGAR_STR_RE.findall(cigar):
        length = int(length_str)
        if op == "M" or op == "=" or op == "X":
            end = q + length
            if end > qlen:
                end = qlen
            seg = end - q
            if seg > 0:
                out[q:end] = (np.arange(r, r + seg) % chrM_length).astype(np.int32)
            q += length
            r += length
        elif op == "I" or op == "S":
            q += length
        elif op == "D" or op == "N":
            r += length
        # "H", "P": consume neither

    return out


def assign_subreads_to_strand(
    subread_seq: str,
    ref_seq: str,
    chrM_length: int,
    min_identity: float = 0.5,
    report_margin: bool = False,
) -> Optional[Dict]:
    """
    Align subread in native and RC orientation to reference.
    Assign to forward if native aligns better, reverse if RC aligns better.

    Args:
        subread_seq: Subread sequence string
        ref_seq: Full reference sequence (circularized)
        chrM_length: Actual mitochondrial genome length
        min_identity: Minimum alignment identity threshold

    Returns:
        dict or None: Assignment result with normalized positions, or None if failed
    """
    native_result = edlib.align(subread_seq, ref_seq, mode="HW", task="path")
    rc_seq = reverse_complement(subread_seq)
    rc_result = edlib.align(rc_seq, ref_seq, mode="HW", task="path")

    native_dist = native_result["editDistance"]
    rc_dist = rc_result["editDistance"]

    if native_dist < rc_dist:
        strand = "fwd"
        best_result = native_result
        best_seq = subread_seq
    elif rc_dist < native_dist:
        strand = "rev"
        best_result = rc_result
        best_seq = rc_seq
    else:
        return None  # Skip ties

    identity = 1.0 - (best_result["editDistance"] / len(subread_seq))
    if identity < min_identity:
        return None

    if best_result["locations"]:
        ref_start = best_result["locations"][0][0]
        position_map = parse_edlib_cigar_to_positions(
            best_result["cigar"], best_seq, ref_start, chrM_length
        )
    else:
        position_map = np.full(len(best_seq), -1, dtype=np.int32)

    result = {
        "strand": strand,
        "aligned_sequence": best_seq,
        "position_map": position_map,
        "edit_distance": best_result["editDistance"],
        "identity": identity,
    }
    if report_margin:
        result["edit_distance_margin"] = abs(native_dist - rc_dist)
    return result


def _assign_single_subread(
    subread_dict: Dict, chrM_length: int, min_identity: float, report_margin: bool = False
) -> Optional[Dict]:
    """
    Worker function for parallel subread assignment.

    Reads the reference sequence for this subread's chrom from the
    process-wide ``_WORKER_REF_SEQS`` dict populated by ``_worker_init``.
    For the single-core in-process fallback we set ``_WORKER_REF_SEQS``
    from the main process before dispatch (see ``process_subread_alignment``).
    """
    if len(subread_dict["query_sequence"]) < 25:
        return None

    ref_seq = _WORKER_REF_SEQS[subread_dict["chrom"]]

    assignment = assign_subreads_to_strand(
        subread_dict["query_sequence"],
        ref_seq,
        chrM_length,
        min_identity,
        report_margin=report_margin,
    )

    if assignment:
        result = {
            "zmw": subread_dict["zmw"],
            "strand": assignment["strand"],
            "zmw_strand": f"{subread_dict['zmw']}_{assignment['strand']}",
            "subread_name": subread_dict["read_name"],
            "aligned_sequence": assignment["aligned_sequence"],
            "position_map": assignment["position_map"],
            "identity": assignment["identity"],
        }
        if report_margin:
            result["edit_distance_margin"] = assignment["edit_distance_margin"]
        return result
    return None


def _iter_worker_results(
    subreads_iter: Iterable[Dict],
    worker,
    n_cores: int,
    ref_seqs: Dict[str, str],
    total: Optional[int],
) -> Iterable[Optional[Dict]]:
    """Yield worker outputs one-at-a-time from a generator of subread dicts.

    The pool is created with an initializer that seeds each worker's
    module-global ``_WORKER_REF_SEQS`` once, so per-task payloads carry
    only the zmw's ``chrom`` string instead of a full ref_seq reference.
    """
    if n_cores == 1:
        # Single-core fallback runs inline; seed the module-global so
        # _assign_single_subread's ref_seq lookup works without a pool.
        _worker_init(ref_seqs)
        for sr in tqdm(subreads_iter, total=total, desc="Assigning subreads"):
            yield worker(sr)
    else:
        with get_pool(n_cores, initializer=_worker_init, initargs=(ref_seqs,)) as pool:
            yield from tqdm(
                pool.imap(worker, subreads_iter, chunksize=50),
                total=total,
                desc=f"Assigning subreads ({n_cores} cores)",
            )


def _assigned_batch_to_table(batch: List[Dict], report_margin: bool) -> pa.Table:
    """Convert a buffer of assignment dicts into a pyarrow Table matching the schema."""
    zmws = [int(r["zmw"]) for r in batch]
    strands = [r["strand"] for r in batch]
    subread_names = [r["subread_name"] for r in batch]
    aligned = [r["aligned_sequence"] for r in batch]
    pos_maps = [np.asarray(r["position_map"], dtype=np.int32).tolist() for r in batch]
    identities = [float(r["identity"]) for r in batch]

    cols = {
        "zmw": pa.array(zmws, type=pa.int64()),
        "strand": pa.array(strands, type=pa.string()).dictionary_encode(),
        "subread_name": pa.array(subread_names, type=pa.string()),
        "aligned_sequence": pa.array(aligned, type=pa.large_string()),
        "position_map": pa.array(pos_maps, type=pa.large_list(pa.int32())),
        "identity": pa.array(identities, type=pa.float32()),
    }
    schema = _ASSIGNED_SUBREAD_SCHEMA
    if report_margin:
        cols["edit_distance_margin"] = pa.array(
            [int(r["edit_distance_margin"]) for r in batch], type=pa.int32()
        )
        schema = _ASSIGNED_SUBREAD_SCHEMA_MARGIN
    return pa.Table.from_pydict(cols, schema=schema)


_BUCKET_MANIFEST_NAME = "manifest.json"
_BUCKET_SCHEMA_VERSION = "0.7"


def _bucket_path(output_dir: Path, bucket_idx: int) -> Path:
    """Canonical bucket file path inside a bucketed output directory."""
    return output_dir / f"bucket_{bucket_idx:02d}.parquet"


def process_subread_alignment(
    zmw_list: List[int],
    subreads_by_zmw: Dict[int, List[Dict]],
    ref_seqs: Dict[str, str],
    zmw_to_chrom: Dict[int, str],
    chrM_length: int,
    min_identity: float,
    n_cores: Optional[int] = None,
    report_margin: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    n_buckets: int = 1,
) -> Union[List[Dict], Path]:
    """
    Align subreads to reference and assign to strands.

    Args:
        zmw_list: List of ZMWs to process
        subreads_by_zmw: Dictionary mapping ZMW to subreads
        ref_seqs: Dictionary mapping chromosome names to reference sequences
        zmw_to_chrom: Dictionary mapping ZMW to chromosome name
        chrM_length: Mitochondrial genome length
        min_identity: Minimum alignment identity
        n_cores: Number of cores for parallel processing
        report_margin: If True, include ``edit_distance_margin`` in each result.
        output_path: If provided, stream assigned-subread records straight to
            this zstd-compressed parquet file and return the path. Avoids
            materializing the full ``List[Dict]`` of assignments, which can
            reach tens of GB at full scale. If None, return the legacy
            ``List[Dict]`` (safe only at small scale).
        n_buckets: When > 1, ``output_path`` is treated as a directory and
            rows are sharded across ``bucket_{i:02d}.parquet`` files keyed by
            ``zmw % n_buckets``. A ``manifest.json`` is written last as the
            atomic completion marker. Downstream composition reads buckets
            one at a time so the read-side peak memory stays bounded at
            ``total / n_buckets``. Requires ``output_path`` to be set.
            Default ``1`` preserves the single-file write path.

    Returns:
        List of assigned subread dictionaries, or ``output_path`` if streaming.
    """
    if n_cores is None:
        n_cores = cpu_count()

    if n_buckets < 1:
        raise ValueError(f"n_buckets must be >= 1, got {n_buckets}")
    if n_buckets > 1 and output_path is None:
        raise ValueError("n_buckets > 1 requires output_path to be set")

    skipped_zmws = {
        zmw: zmw_to_chrom.get(zmw)
        for zmw in zmw_list
        if zmw_to_chrom.get(zmw) is None or zmw_to_chrom.get(zmw) not in ref_seqs
    }
    if skipped_zmws:
        logger.warning(
            f"Skipping {len(skipped_zmws)} ZMWs mapped to chromosomes not in reference: "
            f"{skipped_zmws}"
        )

    # O(len(zmw_list)) pre-count; cheap compared to building the full list
    # and keeps tqdm's total accurate.
    total = sum(
        len(subreads_by_zmw.get(zmw, []))
        for zmw in zmw_list
        if zmw not in skipped_zmws
    )
    logger.info(f"Assigning {total} subreads using {n_cores} cores")

    def _iter_subreads_for_pool() -> Iterable[Dict]:
        # Attach zmw + chrom to each subread just-in-time. Keeping this a
        # generator means the parent never holds all ~1.68M-at-full-scale
        # subread dicts simultaneously; imap pulls one at a time.
        for zmw in zmw_list:
            if zmw in skipped_zmws:
                continue
            chrom = zmw_to_chrom[zmw]
            for sr in subreads_by_zmw.get(zmw, []):
                sr_copy = sr.copy()
                sr_copy["zmw"] = zmw
                sr_copy["chrom"] = chrom
                yield sr_copy

    worker = partial(
        _assign_single_subread,
        chrM_length=chrM_length,
        min_identity=min_identity,
        report_margin=report_margin,
    )

    result_iter = _iter_worker_results(
        _iter_subreads_for_pool(), worker, n_cores, ref_seqs, total
    )

    if output_path is not None and n_buckets == 1:
        output_path = Path(output_path)
        schema = (
            _ASSIGNED_SUBREAD_SCHEMA_MARGIN if report_margin else _ASSIGNED_SUBREAD_SCHEMA
        )
        writer: Optional[pq.ParquetWriter] = None
        buffer: List[Dict] = []
        n_fwd = n_rev = 0
        total_written = 0
        try:
            for r in result_iter:
                if r is None:
                    continue
                if r["strand"] == "fwd":
                    n_fwd += 1
                else:
                    n_rev += 1
                buffer.append(r)
                if len(buffer) >= _ASSIGNED_SUBREAD_FLUSH_ROWS:
                    table = _assigned_batch_to_table(buffer, report_margin)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            output_path, schema, compression="zstd"
                        )
                    writer.write_table(table)
                    total_written += table.num_rows
                    buffer.clear()
            if buffer:
                table = _assigned_batch_to_table(buffer, report_margin)
                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path, schema, compression="zstd"
                    )
                writer.write_table(table)
                total_written += table.num_rows
                buffer.clear()
        finally:
            if writer is not None:
                writer.close()

        if writer is None:
            logger.info(
                f"No subreads assigned; no parquet written to {output_path}"
            )
        else:
            logger.info(
                f"Streamed {total_written} assigned subreads "
                f"(fwd={n_fwd}, rev={n_rev}) to: {output_path}"
            )
        return output_path

    if output_path is not None and n_buckets > 1:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        schema = (
            _ASSIGNED_SUBREAD_SCHEMA_MARGIN if report_margin else _ASSIGNED_SUBREAD_SCHEMA
        )
        # Divide the global flush budget across buckets so total buffered
        # rows stays bounded at ~_ASSIGNED_SUBREAD_FLUSH_ROWS regardless of
        # n_buckets. Otherwise N independent 100k-row buffers at full scale
        # would hold ~48 GB of aligned_sequence across all buckets.
        per_bucket_flush = max(1, _ASSIGNED_SUBREAD_FLUSH_ROWS // n_buckets)
        writers: List[Optional[pq.ParquetWriter]] = [None] * n_buckets
        buffers: List[List[Dict]] = [[] for _ in range(n_buckets)]
        n_fwd = n_rev = 0
        total_written = 0

        def _flush(idx: int) -> None:
            nonlocal total_written
            buf = buffers[idx]
            if not buf:
                return
            table = _assigned_batch_to_table(buf, report_margin)
            if writers[idx] is None:
                writers[idx] = pq.ParquetWriter(
                    _bucket_path(output_path, idx), schema, compression="zstd"
                )
            writers[idx].write_table(table)
            total_written += table.num_rows
            buf.clear()

        try:
            for r in result_iter:
                if r is None:
                    continue
                if r["strand"] == "fwd":
                    n_fwd += 1
                else:
                    n_rev += 1
                idx = int(r["zmw"]) % n_buckets
                buffers[idx].append(r)
                if len(buffers[idx]) >= per_bucket_flush:
                    _flush(idx)
            for idx in range(n_buckets):
                _flush(idx)
        finally:
            for w in writers:
                if w is not None:
                    w.close()

        # Manifest is the atomic completion marker: readers must require it,
        # so write it only after every bucket writer has closed cleanly.
        manifest = {
            "n_buckets": n_buckets,
            "schema_version": _BUCKET_SCHEMA_VERSION,
            "has_margin": report_margin,
        }
        (output_path / _BUCKET_MANIFEST_NAME).write_text(json.dumps(manifest))

        populated = sum(1 for w in writers if w is not None)
        logger.info(
            f"Streamed {total_written} assigned subreads "
            f"(fwd={n_fwd}, rev={n_rev}) across {populated}/{n_buckets} buckets to: {output_path}"
        )
        return output_path

    assigned = [r for r in result_iter if r is not None]
    logger.info(
        f"Assigned {len(assigned)} subreads "
        f"(fwd={sum(1 for s in assigned if s['strand'] == 'fwd')}, "
        f"rev={sum(1 for s in assigned if s['strand'] == 'rev')})"
    )
    return assigned
