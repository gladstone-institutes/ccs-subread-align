"""Core alignment functions for assigning PacBio subreads to strands."""

import logging
import re
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import edlib
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

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

    Args:
        subread_dict: Dictionary with 'zmw', 'read_name', 'query_sequence', '_ref_seq'
        chrM_length: Mitochondrial genome length
        min_identity: Minimum alignment identity

    Returns:
        dict or None: Assignment result with zmw info, or None if failed
    """
    if len(subread_dict["query_sequence"]) < 25:
        return None

    assignment = assign_subreads_to_strand(
        subread_dict["query_sequence"],
        subread_dict["_ref_seq"],
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


def process_subread_alignment(
    zmw_list: List[int],
    subreads_by_zmw: Dict[int, List[Dict]],
    ref_seqs: Dict[str, str],
    zmw_to_chrom: Dict[int, str],
    chrM_length: int,
    min_identity: float,
    n_cores: Optional[int] = None,
    report_margin: bool = False,
) -> List[Dict]:
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

    Returns:
        List of assigned subread dictionaries
    """
    if n_cores is None:
        n_cores = cpu_count()

    all_subreads = []
    skipped_zmws = {}
    for zmw in zmw_list:
        chrom = zmw_to_chrom.get(zmw)
        if chrom is None or chrom not in ref_seqs:
            skipped_zmws[zmw] = chrom
            continue
        ref_seq = ref_seqs[chrom]
        for sr in subreads_by_zmw.get(zmw, []):
            sr_copy = sr.copy()
            sr_copy["zmw"] = zmw
            sr_copy["_ref_seq"] = ref_seq
            all_subreads.append(sr_copy)

    if skipped_zmws:
        logger.warning(
            f"Skipping {len(skipped_zmws)} ZMWs mapped to chromosomes not in reference: "
            f"{skipped_zmws}"
        )

    logger.info(f"Assigning {len(all_subreads)} subreads using {n_cores} cores")

    worker = partial(
        _assign_single_subread,
        chrM_length=chrM_length,
        min_identity=min_identity,
        report_margin=report_margin,
    )

    if n_cores == 1:
        results = [worker(sr) for sr in tqdm(all_subreads, desc="Assigning subreads")]
    else:
        with Pool(processes=n_cores) as pool:
            results = list(
                tqdm(
                    pool.imap(worker, all_subreads, chunksize=50),
                    total=len(all_subreads),
                    desc=f"Assigning subreads ({n_cores} cores)",
                )
            )

    assigned = [r for r in results if r is not None]
    logger.info(
        f"Assigned {len(assigned)} subreads "
        f"(fwd={sum(1 for s in assigned if s['strand'] == 'fwd')}, "
        f"rev={sum(1 for s in assigned if s['strand'] == 'rev')})"
    )
    return assigned
