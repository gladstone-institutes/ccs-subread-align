"""Core alignment functions for assigning PacBio subreads to strands."""

import logging
import re
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import edlib
import numpy as np
from aligntools import Cigar
from tqdm import tqdm

logger = logging.getLogger(__name__)

# pysam CIGAR operation codes to character mapping
_PYSAM_OP_TO_CHAR = {
    0: "M",
    1: "I",
    2: "D",
    3: "N",
    4: "S",
    5: "H",
    6: "P",
    7: "=",
    8: "X",
}


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


def _cigartuples_to_string(cigartuples) -> str:
    """Convert pysam cigartuples to a CIGAR string."""
    return "".join(
        f"{length}{_PYSAM_OP_TO_CHAR[op]}" for op, length in cigartuples
    )


def parse_cigar_to_reference_map(
    cigartuples, reference_start: int, chrM_length: int = 16569
) -> Dict[int, Optional[int]]:
    """
    Parse CIGAR to create mapping from query positions to reference positions.

    Uses aligntools for CIGAR parsing with chrM_length normalization.

    Args:
        cigartuples: List of (operation, length) tuples from pysam
        reference_start: Starting reference position
        chrM_length: Actual mitochondrial genome length (default: 16569)

    Returns:
        dict: Mapping from query positions to normalized reference positions
    """
    cigar_str = _cigartuples_to_string(cigartuples)
    cigar = Cigar.coerce(cigar_str)
    mapping = cigar.coordinate_mapping

    query_to_ref = {}
    for query_pos, ref_pos in mapping.query_to_ref.items():
        normalized = (ref_pos + reference_start) % chrM_length
        query_to_ref[query_pos] = normalized

    # Mark insertion positions (query positions not in the mapping) as None
    query_length = cigar.query_length
    for qpos in range(query_length):
        if qpos not in query_to_ref:
            query_to_ref[qpos] = None

    return query_to_ref


def parse_edlib_cigar_to_positions(
    cigar: str, query_seq: str, ref_start: int, chrM_length: int = 16569
) -> np.ndarray:
    """
    Parse edlib CIGAR string to map query positions to reference positions.

    Uses aligntools for CIGAR parsing. Normalizes positions to actual mtDNA
    coordinates (0 to chrM_length-1) using modulo.

    Args:
        cigar: Edlib CIGAR string (e.g., "100=5X10I20=")
        query_seq: Query sequence string
        ref_start: Starting reference position
        chrM_length: Actual mitochondrial genome length (default: 16569)

    Returns:
        np.array: Array mapping query positions to normalized reference positions
                  (-1 for gaps/insertions)
    """
    if not cigar:
        return np.full(len(query_seq), -1, dtype=np.int32)

    position_map = np.full(len(query_seq), -1, dtype=np.int32)

    parsed = Cigar.coerce(cigar)
    mapping = parsed.coordinate_mapping

    for query_pos, ref_pos in mapping.query_to_ref.items():
        if query_pos < len(query_seq):
            normalized = (ref_pos + ref_start) % chrM_length
            position_map[query_pos] = normalized

    return position_map


def assign_subreads_to_strand(
    subread_seq: str, ref_seq: str, chrM_length: int, min_identity: float = 0.5
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

    return {
        "strand": strand,
        "aligned_sequence": best_seq,
        "position_map": position_map,
        "edit_distance": best_result["editDistance"],
        "identity": identity,
    }


def _assign_single_subread(
    subread_dict: Dict, chrM_length: int, min_identity: float
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
        subread_dict["query_sequence"], subread_dict["_ref_seq"], chrM_length, min_identity
    )

    if assignment:
        return {
            "zmw": subread_dict["zmw"],
            "strand": assignment["strand"],
            "zmw_strand": f"{subread_dict['zmw']}_{assignment['strand']}",
            "subread_name": subread_dict["read_name"],
            "aligned_sequence": assignment["aligned_sequence"],
            "position_map": assignment["position_map"],
            "identity": assignment["identity"],
        }
    return None


def process_subread_alignment(
    zmw_list: List[int],
    subreads_by_zmw: Dict[int, List[Dict]],
    ref_seqs: Dict[str, str],
    zmw_to_chrom: Dict[int, str],
    chrM_length: int,
    min_identity: float,
    n_cores: Optional[int] = None,
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
