"""Base composition calculation from subread-to-CCS alignments."""

import logging
from collections import defaultdict
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

_BASE_TO_IDX = {"A": 0, "T": 1, "C": 2, "G": 3, "N": 4}


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
            query_sequence, query_length, query_to_ref_map, quality_array, zmw,
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

    base_counts = np.zeros((ccs_len, 5), dtype=np.int32)

    # Build CCS position -> reference position array
    ccs_to_ref = np.full(ccs_len, -1, dtype=np.int32)
    for ccs_pos, ref_pos in ccs_read["query_to_ref_map"].items():
        if ref_pos is not None and 0 <= ccs_pos < ccs_len:
            ccs_to_ref[ccs_pos] = ref_pos

    # Count bases from subreads at each CCS position
    for sr in assigned_subreads:
        sr_seq = sr["aligned_sequence"]
        position_map = sr["position_map"]

        for sr_pos in range(len(sr_seq)):
            ref_pos = position_map[sr_pos]
            if ref_pos >= 0:
                ccs_positions = np.where(ccs_to_ref == ref_pos)[0]
                for ccs_pos in ccs_positions:
                    base = sr_seq[sr_pos]
                    base_idx = _BASE_TO_IDX.get(base, 4)
                    base_counts[ccs_pos, base_idx] += 1

    ccs_seq = ccs_read["query_sequence"]

    df = pd.DataFrame(
        {
            "zmw": ccs_read["zmw"],
            "strand": ccs_read["strand"],
            "zmw_strand": ccs_read["zmw_strand"],
            "ccs_pos": np.arange(ccs_len),
            "ref_pos": ccs_to_ref,
            "ccs_base": list(ccs_seq),
            "q_score": ccs_read["quality_array"],
            "A_count": base_counts[:, 0],
            "T_count": base_counts[:, 1],
            "C_count": base_counts[:, 2],
            "G_count": base_counts[:, 3],
            "N_count": base_counts[:, 4],
        }
    )

    df["total_subreads"] = base_counts.sum(axis=1)

    # Agreement fraction: proportion of subreads matching CCS base
    agreement_counts = np.array(
        [base_counts[i, _BASE_TO_IDX.get(ccs_seq[i], 4)] for i in range(ccs_len)]
    )
    df["agreement_fraction"] = np.where(
        df["total_subreads"] > 0,
        agreement_counts / df["total_subreads"],
        0.0,
    )

    # Reference base lookup
    df["reference_base"] = df["ref_pos"].apply(
        lambda x: ref_seq[x] if 0 <= x < len(ref_seq) else "N"
    )

    return df


def _process_ccs_composition(args: Tuple) -> Optional[pd.DataFrame]:
    """Worker function for parallel base composition calculation."""
    ccs, zmw_strand_subreads, ref_seq, chrM_length = args
    if len(zmw_strand_subreads) > 0:
        return calculate_base_composition(ccs, zmw_strand_subreads, ref_seq, chrM_length)
    return None


def calculate_all_base_compositions(
    ccs_reads: List[Dict],
    assigned_subreads: List[Dict],
    ref_seqs: Dict[str, str],
    zmw_to_chrom: Dict[int, str],
    chrM_length: int = 16569,
    n_cores: Optional[int] = None,
) -> pd.DataFrame:
    """
    Calculate base composition for all CCS reads.

    Args:
        ccs_reads: List of CCS read dictionaries (from load_ccs_reads).
        assigned_subreads: List of assigned subread dicts (from process_subread_alignment).
        ref_seqs: Dictionary mapping chromosome names to reference sequences.
        zmw_to_chrom: Dictionary mapping ZMW to chromosome name.
        chrM_length: Genome length for coordinate normalization.
        n_cores: Number of cores for parallel processing (default: all).

    Returns:
        DataFrame with base composition at all positions across all CCS reads.
    """
    if n_cores is None:
        n_cores = cpu_count()

    # Group subreads by (zmw, strand)
    subreads_by_zmw_strand = defaultdict(list)
    for sr in assigned_subreads:
        subreads_by_zmw_strand[(sr["zmw"], sr["strand"])].append(sr)

    logger.info(f"{len(subreads_by_zmw_strand)} unique (zmw, strand) groups")

    # Build work items
    work_items = []
    for ccs in ccs_reads:
        chrom = zmw_to_chrom.get(ccs["zmw"])
        if chrom is None or chrom not in ref_seqs:
            continue
        matched_subreads = subreads_by_zmw_strand.get((ccs["zmw"], ccs["strand"]), [])
        work_items.append((ccs, matched_subreads, ref_seqs[chrom], chrM_length))

    logger.info(f"Calculating base composition for {len(work_items)} CCS reads using {n_cores} cores")

    if n_cores == 1:
        all_dfs = [
            _process_ccs_composition(item)
            for item in tqdm(work_items, desc="Processing CCS reads")
        ]
    else:
        with Pool(processes=n_cores) as pool:
            all_dfs = list(
                tqdm(
                    pool.imap(_process_ccs_composition, work_items, chunksize=10),
                    total=len(work_items),
                    desc=f"Processing CCS reads ({n_cores} cores)",
                )
            )

    all_dfs = [df for df in all_dfs if df is not None]

    if not all_dfs:
        return pd.DataFrame()

    df_all = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Calculated composition for {len(df_all):,} positions")
    return df_all
