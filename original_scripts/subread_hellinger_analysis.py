#!/usr/bin/env python3
"""
Generalized Subread Alignment and Hellinger Distance Calculation

This script processes PacBio CCS sequences from mt-mismatch kinetics output,
aligns subreads to CCS reads, calculates base composition at each position,
and computes Hellinger distance for ZMWs with mismatches (both strands).

NOTE: This script has been developed and tested specifically for human
mitochondrial DNA (hg38 chrM, 16,569 bp). While it may work with other
circular genomes, parameters and assumptions are optimized for human mtDNA
analysis. Use with other organisms or nuclear DNA has not been validated.

The Hellinger distance measures the divergence between the observed subread
base distribution and the expected distribution at each position, providing
a metric for identifying positions with unusual base composition patterns.

Algorithm Overview:
    1. Load kinetics data (from mt-mismatch) and identify ZMWs with mismatches (bulge/bubble)
    2. Load CCS reads and subreads from BAM files
    3. Align subreads to CCS using competitive alignment (native vs RC)
    4. Calculate base composition (A/T/C/G/N counts) at each CCS position
    5. For mismatch ZMWs, calculate Hellinger distance using genome-wide
       expected distribution (excluding mismatch positions)

Key Features:
    - Processes ALL CCS positions for base composition
    - Calculates Hellinger distance for BOTH strands of mismatch ZMWs
    - Handles circular mitochondrial genome (position normalization)
    - Excludes mismatch positions from expected distribution calculation

"""

import argparse
import logging
import os
import re
import sys
from collections import defaultdict
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pysam
import edlib
from tqdm import tqdm

# Parallel processing configuration
# Default: use NSLOTS env var (SGE/cluster) or all available cores
N_CORES = int(os.environ.get('NSLOTS', cpu_count()))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions (from notebook cell 6)
# =============================================================================

def reverse_complement(seq: str) -> str:
    """Return reverse complement of DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(seq))


def extract_zmw_from_name(read_name: str) -> Optional[int]:
    """Extract ZMW number from PacBio read name."""
    parts = read_name.split('/')
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def parse_cigar_to_reference_map(cigartuples, reference_start: int, chrM_length: int = 16569) -> Dict[int, Optional[int]]:
    """
    Parse CIGAR to create mapping from query positions to reference positions.

    Args:
        cigartuples: List of (operation, length) tuples from pysam
        reference_start: Starting reference position
        chrM_length: Actual mitochondrial genome length (default: 16569)

    Returns:
        dict: Mapping from query positions to normalized reference positions
    """
    query_to_ref = {}
    query_pos = 0
    ref_pos = reference_start

    for op, length in cigartuples:
        if op in [0, 7, 8]:  # M, =, X - consume both
            for i in range(length):
                # CRITICAL: Normalize to actual mtDNA coordinates (0-16,568)
                normalized_pos = ref_pos % chrM_length
                query_to_ref[query_pos] = normalized_pos
                query_pos += 1
                ref_pos += 1
        elif op == 1:  # I - insertion
            for i in range(length):
                query_to_ref[query_pos] = None
                query_pos += 1
        elif op == 2:  # D - deletion
            ref_pos += length
        elif op == 4:  # S - soft clip
            query_pos += length

    return query_to_ref


def parse_edlib_cigar_to_positions(cigar: str, query_seq: str, ref_start: int, chrM_length: int = 16569) -> np.ndarray:
    """
    Parse edlib CIGAR string to map query positions to reference positions.

    CRITICAL: Normalizes positions to actual mtDNA coordinates (0-16,568) using modulo.

    Args:
        cigar: Edlib CIGAR string (e.g., "100=5X10I20=")
        query_seq: Query sequence string
        ref_start: Starting reference position
        chrM_length: Actual mitochondrial genome length (default: 16569)

    Returns:
        np.array: Array mapping query positions to normalized reference positions (-1 for gaps)
    """
    if not cigar:
        return np.full(len(query_seq), -1, dtype=np.int32)

    position_map = np.full(len(query_seq), -1, dtype=np.int32)
    pattern = re.compile(r'(\d+)([=XID])')

    query_pos = 0
    ref_pos = ref_start

    for match in pattern.finditer(cigar):
        length = int(match.group(1))
        op = match.group(2)

        if op in ['=', 'X']:  # Match or mismatch
            for i in range(length):
                if query_pos < len(query_seq):
                    # CRITICAL: Normalize to actual mtDNA coordinates (0-16,568)
                    normalized_pos = ref_pos % chrM_length
                    position_map[query_pos] = normalized_pos
                query_pos += 1
                ref_pos += 1
        elif op == 'I':  # Insertion
            query_pos += length
        elif op == 'D':  # Deletion
            ref_pos += length

    return position_map


# =============================================================================
# Subread Assignment (from notebook cell 11)
# =============================================================================

def assign_subreads_to_strand(subread_seq: str, ref_seq: str, chrM_length: int, min_identity: float = 0.5) -> Optional[Dict]:
    """
    Align subread in native and RC orientation to reference.
    Assign to forward if native aligns better, reverse if RC aligns better.

    CRITICAL: Normalizes all positions to actual mtDNA coordinates (0-16,568).

    Args:
        subread_seq: Subread sequence string
        ref_seq: Full reference sequence (circularized, 33,138 bp)
        chrM_length: Actual mitochondrial genome length (16,569 bp)
        min_identity: Minimum alignment identity threshold

    Returns:
        dict or None: Assignment result with normalized positions, or None if failed
    """
    # Align native
    native_result = edlib.align(subread_seq, ref_seq, mode='HW', task='path')

    # Align reverse complement
    rc_seq = reverse_complement(subread_seq)
    rc_result = edlib.align(rc_seq, ref_seq, mode='HW', task='path')

    native_dist = native_result['editDistance']
    rc_dist = rc_result['editDistance']

    if native_dist < rc_dist:
        strand = 'fwd'
        best_result = native_result
        best_seq = subread_seq
    elif rc_dist < native_dist:
        strand = 'rev'
        best_result = rc_result
        best_seq = rc_seq
    else:
        return None  # Skip ties

    # Check identity
    identity = 1.0 - (best_result['editDistance'] / len(subread_seq))
    if identity < min_identity:
        return None

    # Parse alignment WITH POSITION NORMALIZATION
    if best_result['locations']:
        ref_start = best_result['locations'][0][0]
        # CRITICAL: Pass chrM_length for position normalization
        position_map = parse_edlib_cigar_to_positions(
            best_result['cigar'], best_seq, ref_start, chrM_length
        )
    else:
        position_map = np.full(len(best_seq), -1, dtype=np.int32)

    return {
        'strand': strand,
        'aligned_sequence': best_seq,
        'position_map': position_map,
        'edit_distance': best_result['editDistance'],
        'identity': identity
    }


def _assign_single_subread(subread_dict: Dict, ref_seq: str, chrM_length: int, min_identity: float) -> Optional[Dict]:
    """
    Worker function for parallel subread assignment.

    Args:
        subread_dict: Dictionary with 'zmw', 'read_name', 'query_sequence'
        ref_seq: Reference sequence string
        chrM_length: Mitochondrial genome length
        min_identity: Minimum alignment identity

    Returns:
        dict or None: Assignment result with zmw info, or None if failed
    """
    if len(subread_dict['query_sequence']) < 25:
        return None

    assignment = assign_subreads_to_strand(
        subread_dict['query_sequence'], ref_seq, chrM_length, min_identity
    )

    if assignment:
        return {
            'zmw': subread_dict['zmw'],
            'strand': assignment['strand'],
            'zmw_strand': f"{subread_dict['zmw']}_{assignment['strand']}",
            'subread_name': subread_dict['read_name'],
            'aligned_sequence': assignment['aligned_sequence'],
            'position_map': assignment['position_map'],
            'identity': assignment['identity']
        }
    return None


# =============================================================================
# Base Composition Calculation (from notebook cell 13)
# =============================================================================

def calculate_base_composition(ccs_read: Dict, assigned_subreads: List[Dict], ref_seq: str) -> pd.DataFrame:
    """
    Calculate base composition from subreads at each CCS position.

    Args:
        ccs_read: Dictionary with CCS read information
        assigned_subreads: List of assigned subread dictionaries
        ref_seq: Reference sequence

    Returns:
        DataFrame with base composition at each position
    """
    ccs_len = ccs_read['query_length']
    base_to_idx = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}

    # Initialize count arrays
    base_counts = np.zeros((ccs_len, 5), dtype=np.int32)

    # Map CCS positions to reference
    ccs_to_ref = np.full(ccs_len, -1, dtype=np.int32)
    for ccs_pos, ref_pos in ccs_read['query_to_ref_map'].items():
        if ref_pos is not None and 0 <= ccs_pos < ccs_len:
            ccs_to_ref[ccs_pos] = ref_pos

    # Count bases from subreads
    for sr in assigned_subreads:
        sr_seq = sr['aligned_sequence']
        position_map = sr['position_map']

        for sr_pos in range(len(sr_seq)):
            ref_pos = position_map[sr_pos]
            if ref_pos >= 0:
                ccs_positions = np.where(ccs_to_ref == ref_pos)[0]
                for ccs_pos in ccs_positions:
                    base = sr_seq[sr_pos]
                    base_idx = base_to_idx.get(base, 4)
                    base_counts[ccs_pos, base_idx] += 1

    # Create DataFrame
    ccs_seq = ccs_read['query_sequence']

    df = pd.DataFrame({
        'zmw': ccs_read['zmw'],
        'strand': ccs_read['strand'],
        'zmw_strand': ccs_read['zmw_strand'],
        'ccs_pos': np.arange(ccs_len),
        'ref_pos': ccs_to_ref,
        'ccs_base': list(ccs_seq),
        'q_score': ccs_read['quality_array'],
        'A_count': base_counts[:, 0],
        'T_count': base_counts[:, 1],
        'C_count': base_counts[:, 2],
        'G_count': base_counts[:, 3],
        'N_count': base_counts[:, 4],
    })

    df['total_subreads'] = base_counts.sum(axis=1)

    # Agreement fraction
    agreement_counts = np.array([
        base_counts[i, base_to_idx.get(ccs_seq[i], 4)]
        for i in range(ccs_len)
    ])
    df['agreement_fraction'] = np.where(
        df['total_subreads'] > 0,
        agreement_counts / df['total_subreads'],
        0.0
    )

    # Add reference base
    df['reference_base'] = df['ref_pos'].apply(
        lambda x: ref_seq[x] if 0 <= x < len(ref_seq) else 'N'
    )

    return df


def _process_ccs_composition(args: Tuple) -> Optional[pd.DataFrame]:
    """
    Worker function for parallel base composition calculation.

    Args:
        args: Tuple of (ccs_read, zmw_strand_subreads, ref_seq)

    Returns:
        DataFrame or None: Base composition DataFrame
    """
    ccs, zmw_strand_subreads, ref_seq = args
    if len(zmw_strand_subreads) > 0:
        return calculate_base_composition(ccs, zmw_strand_subreads, ref_seq)
    return None


# =============================================================================
# Hellinger Distance Calculation (from notebook cell 17)
# =============================================================================

def calculate_per_zmw_strand_expected_distribution(
    df: pd.DataFrame,
    zmw: int,
    strand: str,
    exclude_pos: Optional[int] = None,
    mismatch_positions: Optional[Set] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate expected base distribution for a specific (ZMW, strand) combination (genome-wide).

    Args:
        df: DataFrame with all position data
        zmw: ZMW identifier
        strand: Strand ('fwd' or 'rev')
        exclude_pos: Single position to exclude (e.g., target position)
        mismatch_positions: Set of (zmw, strand, tpos) tuples to exclude

    Returns:
        Dictionary mapping CCS base to expected distribution
    """
    data = df[(df['zmw'] == zmw) & (df['strand'] == strand)].copy()

    # CRITICAL: Exclude positions that don't align to reference (insertions, soft-clips)
    data = data[data['ref_pos'] >= 0]

    # Exclude single position if specified
    if exclude_pos is not None:
        data = data[data['ref_pos'] != exclude_pos]

    # Exclude mismatch positions for this specific (ZMW, strand)
    if mismatch_positions is not None and len(mismatch_positions) > 0:
        def is_not_mismatch(row):
            return (row['zmw'], row['strand'], row['ref_pos']) not in mismatch_positions
        data = data[data.apply(is_not_mismatch, axis=1)]

    data = data[data['total_subreads'] >= 5]

    if len(data) == 0:
        return {}

    error_dist = {}
    for ccs_base in ['A', 'T', 'C', 'G']:
        base_positions = data[data['ccs_base'] == ccs_base]
        if len(base_positions) == 0:
            continue

        total_A = base_positions['A_count'].sum()
        total_T = base_positions['T_count'].sum()
        total_C = base_positions['C_count'].sum()
        total_G = base_positions['G_count'].sum()
        total_N = base_positions['N_count'].sum()
        total_all = total_A + total_T + total_C + total_G + total_N

        if total_all > 0:
            error_dist[ccs_base] = {
                'A': total_A / total_all,
                'T': total_T / total_all,
                'C': total_C / total_all,
                'G': total_G / total_all,
                'N': total_N / total_all
            }

    return error_dist


def calculate_hellinger_distance(observed_counts: Dict[str, int], ccs_base: str, expected_dist: Dict[str, Dict[str, float]]) -> Optional[float]:
    """
    Calculate Hellinger distance: H(P, Q) = (1/sqrt(2)) * sqrt(sum((sqrt(P(x)) - sqrt(Q(x)))^2))

    Args:
        observed_counts: Dictionary of observed base counts
        ccs_base: The CCS base call
        expected_dist: Dictionary of expected base distributions per CCS base

    Returns:
        float or None: Hellinger distance value, or None if calculation fails
    """
    bases = ['A', 'T', 'C', 'G', 'N']

    # Calculate observed probability distribution P
    total = sum(observed_counts.get(b, 0) for b in bases)
    if total == 0:
        return None

    P = {base: observed_counts.get(base, 0) / total for base in bases}

    # Get expected probability distribution Q
    if ccs_base not in expected_dist:
        return None

    Q = expected_dist[ccs_base]

    # Calculate Hellinger distance: (1/sqrt(2)) * sqrt(sum((sqrt(P(x)) - sqrt(Q(x)))^2))
    hellinger_dist = 0.0
    for base in bases:
        p = P[base]
        q = Q.get(base, 1e-10)  # Small epsilon to avoid issues
        hellinger_dist += (np.sqrt(p) - np.sqrt(q)) ** 2

    hellinger_dist = np.sqrt(hellinger_dist) / np.sqrt(2)

    return hellinger_dist


def _process_zmw_strand_hellinger(args: Tuple) -> List[Dict]:
    """
    Worker function for parallel Hellinger distance calculation.

    Args:
        args: Tuple of (zmw, strand, df_group_records, expected_dist)

    Returns:
        list: List of result dictionaries for each position
    """
    zmw, strand, df_group_records, expected_dist = args

    results = []

    # Convert records back to DataFrame for processing
    df_group = pd.DataFrame.from_records(df_group_records)

    for _, pos_row in df_group.iterrows():
        observed = {
            'A': pos_row['A_count'],
            'T': pos_row['T_count'],
            'C': pos_row['C_count'],
            'G': pos_row['G_count'],
            'N': pos_row['N_count']
        }

        hellinger = calculate_hellinger_distance(observed, pos_row['ccs_base'], expected_dist)

        if hellinger is not None:
            results.append({
                'zmw': zmw,
                'strand': strand,
                'zmw_strand': pos_row['zmw_strand'],
                'ref_pos': pos_row['ref_pos'],
                'ccs_base': pos_row['ccs_base'],
                'reference_base': pos_row['reference_base'],
                'q_score': pos_row['q_score'],
                'total_subreads': pos_row['total_subreads'],
                'agreement_fraction': pos_row['agreement_fraction'],
                'hellinger_distance': hellinger
            })

    return results


# =============================================================================
# Main Processing Functions
# =============================================================================

def load_kinetics_data(kinetics_path: str, chrM_length: int) -> Tuple[pd.DataFrame, Set[int], Set[Tuple[int, str, int]]]:
    """
    Load kinetics data and extract mismatch information.

    Args:
        kinetics_path: Path to kinetics CSV file
        chrM_length: Mitochondrial genome length

    Returns:
        Tuple of (kinetics_df, mismatch_zmws, mismatch_positions)
    """
    logger.info(f"Loading kinetics data from: {kinetics_path}")
    kinetics_df = pd.read_csv(kinetics_path)
    logger.info(f"  Total kinetics rows: {len(kinetics_df):,}")

    # Create notebook strand label from ref_strand
    # ref_strand: '+' = forward alignment (is_reverse=False) -> 'fwd'
    #            '-' = reverse alignment (is_reverse=True) -> 'rev'
    kinetics_df['notebook_strand'] = kinetics_df['ref_strand'].apply(
        lambda x: 'rev' if x == '-' else 'fwd'
    )

    # Extract mismatch positions directly from is_mismatch column
    mismatch_rows = kinetics_df[kinetics_df['is_mismatch'] == True]
    logger.info(f"  Rows with is_mismatch=True: {len(mismatch_rows):,}")

    mismatch_positions = set(
        mismatch_rows[['zmw', 'notebook_strand', 'tpos']]
        .apply(lambda row: (int(row['zmw']), row['notebook_strand'], int(row['tpos']) % chrM_length), axis=1)
    )
    logger.info(f"  Created {len(mismatch_positions):,} mismatch position tuples")

    # Get ZMWs with mismatches
    mismatch_zmws = set(kinetics_df[kinetics_df['is_mismatch'] == True]['zmw'].unique())
    logger.info(f"  ZMWs with mismatches: {len(mismatch_zmws)}")

    return kinetics_df, mismatch_zmws, mismatch_positions


def load_ccs_reads(ccs_bam_path: str, zmw_list: List[int], chrM_length: int) -> List[Dict]:
    """
    Load CCS reads from BAM file.

    Args:
        ccs_bam_path: Path to CCS BAM file
        zmw_list: List of ZMWs to load
        chrM_length: Mitochondrial genome length for position normalization

    Returns:
        List of CCS read dictionaries
    """
    logger.info(f"Loading CCS reads from: {ccs_bam_path}")
    zmw_set = set(zmw_list)
    ccs_reads = []

    with pysam.AlignmentFile(ccs_bam_path, 'rb') as bam:
        for read in bam.fetch():
            zmw = extract_zmw_from_name(read.query_name)
            if zmw in zmw_set:
                # Use SAM flag to determine strand
                strand = 'rev' if read.is_reverse else 'fwd'

                ccs_reads.append({
                    'zmw': zmw,
                    'strand': strand,
                    'zmw_strand': f"{zmw}_{strand}",
                    'read_name': read.query_name,
                    'sam_flag': read.flag,
                    'reference_start': read.reference_start,
                    'query_sequence': read.query_sequence,
                    'query_length': read.query_length,
                    'cigartuples': read.cigartuples,
                    'quality_array': np.array(read.query_qualities) if read.query_qualities else np.zeros(read.query_length),
                    'mapping_quality': read.mapping_quality
                })

    # Add query-to-ref mapping WITH POSITION NORMALIZATION
    for ccs in ccs_reads:
        if ccs['cigartuples']:
            ccs['query_to_ref_map'] = parse_cigar_to_reference_map(
                ccs['cigartuples'], ccs['reference_start'], chrM_length
            )
        else:
            ccs['query_to_ref_map'] = {}

    logger.info(f"  Loaded {len(ccs_reads)} CCS reads")
    logger.info(f"  Unique ZMWs: {len(set(c['zmw'] for c in ccs_reads))}")
    logger.info(f"  Forward strand: {sum(1 for c in ccs_reads if c['strand'] == 'fwd')}")
    logger.info(f"  Reverse strand: {sum(1 for c in ccs_reads if c['strand'] == 'rev')}")

    return ccs_reads


def load_subreads(subreads_bam_path: str, zmw_list: List[int]) -> Dict[int, List[Dict]]:
    """
    Load subreads from BAM file.

    Args:
        subreads_bam_path: Path to subreads BAM file
        zmw_list: List of ZMWs to load

    Returns:
        Dictionary mapping ZMW to list of subread dictionaries
    """
    logger.info(f"Loading subreads from: {subreads_bam_path}")
    zmw_set = set(zmw_list)
    subreads_by_zmw = defaultdict(list)

    with pysam.AlignmentFile(subreads_bam_path, 'rb', check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            zmw = extract_zmw_from_name(read.query_name)
            if zmw in zmw_set:
                subreads_by_zmw[zmw].append({
                    'read_name': read.query_name,
                    'zmw': zmw,
                    'query_sequence': read.query_sequence,
                    'query_length': read.query_length
                })

    total_subreads = sum(len(v) for v in subreads_by_zmw.values())
    logger.info(f"  Loaded {total_subreads} subreads across {len(subreads_by_zmw)} ZMWs")
    logger.info(f"  Mean subreads per ZMW: {total_subreads / len(subreads_by_zmw):.1f}" if subreads_by_zmw else "  No subreads found")

    return dict(subreads_by_zmw)


def process_subread_alignment(
    zmw_list: List[int],
    subreads_by_zmw: Dict[int, List[Dict]],
    ref_seq: str,
    chrM_length: int,
    min_identity: float,
    n_cores: int = None
) -> List[Dict]:
    """
    Align subreads to reference and assign to strands.

    Args:
        zmw_list: List of ZMWs to process
        subreads_by_zmw: Dictionary mapping ZMW to subreads
        ref_seq: Reference sequence
        chrM_length: Mitochondrial genome length
        min_identity: Minimum alignment identity
        n_cores: Number of cores for parallel processing (default: N_CORES global)

    Returns:
        List of assigned subread dictionaries
    """
    if n_cores is None:
        n_cores = N_CORES

    logger.info(f"Assigning subreads to strands using {n_cores} cores...")

    # Flatten all subreads into single list with zmw info
    all_subreads = []
    for zmw in zmw_list:
        for sr in subreads_by_zmw.get(zmw, []):
            sr_copy = sr.copy()
            sr_copy['zmw'] = zmw
            all_subreads.append(sr_copy)

    logger.info(f"  Total subreads to process: {len(all_subreads)}")

    # Create worker function with fixed parameters
    worker = partial(
        _assign_single_subread,
        ref_seq=ref_seq,
        chrM_length=chrM_length,
        min_identity=min_identity
    )

    # Parallel or serial processing
    if n_cores == 1:
        # Serial mode (for debugging)
        results = [worker(sr) for sr in tqdm(all_subreads, desc="Assigning subreads (serial)")]
    else:
        # Parallel mode
        with Pool(processes=n_cores) as pool:
            results = list(tqdm(
                pool.imap(worker, all_subreads, chunksize=50),
                total=len(all_subreads),
                desc=f"Assigning subreads ({n_cores} cores)"
            ))

    # Filter out None results
    assigned_subreads = [r for r in results if r is not None]

    logger.info(f"  Assigned {len(assigned_subreads)} subreads")
    logger.info(f"  Forward: {sum(1 for s in assigned_subreads if s['strand'] == 'fwd')}")
    logger.info(f"  Reverse: {sum(1 for s in assigned_subreads if s['strand'] == 'rev')}")

    return assigned_subreads


def calculate_all_base_compositions(
    ccs_reads: List[Dict],
    assigned_subreads: List[Dict],
    ref_seq: str,
    n_cores: int = None
) -> pd.DataFrame:
    """
    Calculate base composition for all CCS reads.

    Args:
        ccs_reads: List of CCS read dictionaries
        assigned_subreads: List of assigned subread dictionaries
        ref_seq: Reference sequence
        n_cores: Number of cores for parallel processing (default: N_CORES global)

    Returns:
        DataFrame with base composition at all positions
    """
    if n_cores is None:
        n_cores = N_CORES

    logger.info(f"Calculating base composition for all CCS reads using {n_cores} cores...")

    # Pre-group subreads by (zmw, strand) for efficient lookup
    logger.info("  Pre-grouping subreads by (zmw, strand)...")
    subreads_by_zmw_strand = defaultdict(list)
    for sr in assigned_subreads:
        key = (sr['zmw'], sr['strand'])
        subreads_by_zmw_strand[key].append(sr)

    logger.info(f"  {len(subreads_by_zmw_strand)} unique (zmw, strand) groups")

    # Prepare work items - tuple of (ccs, subreads, ref_seq)
    work_items = [
        (ccs, subreads_by_zmw_strand.get((ccs['zmw'], ccs['strand']), []), ref_seq)
        for ccs in ccs_reads
    ]

    # Parallel or serial processing
    if n_cores == 1:
        # Serial mode (for debugging)
        all_positions = [
            _process_ccs_composition(item)
            for item in tqdm(work_items, desc="Processing CCS reads (serial)")
        ]
    else:
        # Parallel mode
        with Pool(processes=n_cores) as pool:
            all_positions = list(tqdm(
                pool.imap(_process_ccs_composition, work_items, chunksize=10),
                total=len(work_items),
                desc=f"Processing CCS reads ({n_cores} cores)"
            ))

    # Filter out None results and concatenate
    all_positions = [df for df in all_positions if df is not None]

    if not all_positions:
        return pd.DataFrame()

    df_all = pd.concat(all_positions, ignore_index=True)
    logger.info(f"  Calculated composition for {len(df_all):,} positions")
    logger.info(f"  Unique zmw_strand combinations: {df_all['zmw_strand'].nunique()}")

    return df_all


def calculate_hellinger_for_mismatch_zmws(
    df_all: pd.DataFrame,
    mismatch_zmws: Set[int],
    mismatch_positions: Set[Tuple[int, str, int]],
    n_cores: int = None
) -> pd.DataFrame:
    """
    Calculate Hellinger distance for ZMWs with mismatches (both strands).

    Args:
        df_all: DataFrame with all base composition data
        mismatch_zmws: Set of ZMWs with mismatches
        mismatch_positions: Set of (zmw, strand, tpos) tuples to exclude
        n_cores: Number of cores for parallel processing (default: N_CORES global)

    Returns:
        DataFrame with Hellinger distances
    """
    if n_cores is None:
        n_cores = N_CORES

    logger.info(f"Calculating Hellinger distance for mismatch ZMWs using {n_cores} cores...")

    # Filter to aligned positions only
    df_aligned = df_all[df_all['ref_pos'] >= 0].copy()

    # Get zmw_strand pairs for mismatch ZMWs (BOTH strands)
    zmw_strand_pairs = df_aligned[df_aligned['zmw'].isin(mismatch_zmws)][['zmw', 'strand']].drop_duplicates()
    logger.info(f"  Processing {len(zmw_strand_pairs)} (zmw, strand) pairs from {len(mismatch_zmws)} mismatch ZMWs")

    # Pre-calculate genome-wide expected distributions
    logger.info("  Pre-calculating genome-wide expected distributions...")
    genome_wide_dists = {}
    for _, row in tqdm(zmw_strand_pairs.iterrows(), total=len(zmw_strand_pairs), desc="Expected distributions"):
        zmw = row['zmw']
        strand = row['strand']
        expected_dist = calculate_per_zmw_strand_expected_distribution(
            df_all, zmw, strand,
            exclude_pos=None,
            mismatch_positions=mismatch_positions
        )
        genome_wide_dists[(zmw, strand)] = expected_dist

    # Group data for efficient lookup
    df_grouped = df_aligned.groupby(['zmw', 'strand'], sort=False)

    # Prepare work items for parallel processing
    # Convert DataFrames to records (lists of dicts) for pickling
    logger.info("  Preparing work items for parallel processing...")
    work_items = []
    for _, row in zmw_strand_pairs.iterrows():
        zmw = row['zmw']
        strand = row['strand']
        try:
            df_group = df_grouped.get_group((zmw, strand))
            # Convert to records for serialization
            records = df_group.to_dict('records')
            expected_dist = genome_wide_dists.get((zmw, strand), {})
            if expected_dist:
                work_items.append((zmw, strand, records, expected_dist))
        except KeyError:
            continue

    logger.info(f"  Prepared {len(work_items)} work items")

    # Parallel or serial processing
    if n_cores == 1:
        # Serial mode (for debugging)
        nested_results = [
            _process_zmw_strand_hellinger(item)
            for item in tqdm(work_items, desc="Hellinger distance (serial)")
        ]
    else:
        # Parallel mode
        with Pool(processes=n_cores) as pool:
            nested_results = list(tqdm(
                pool.imap(_process_zmw_strand_hellinger, work_items, chunksize=5),
                total=len(work_items),
                desc=f"Hellinger distance ({n_cores} cores)"
            ))

    # Flatten results
    hellinger_results = [r for batch in nested_results for r in batch]

    df_hellinger = pd.DataFrame(hellinger_results)
    logger.info(f"  Calculated Hellinger distance for {len(df_hellinger):,} positions")

    return df_hellinger


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="""
Generalized Subread Alignment and Hellinger Distance Calculation

This script processes PacBio CCS sequences from mt-mismatch kinetics output,
aligns subreads to CCS reads, calculates base composition at each position,
and computes Hellinger distance for ZMWs with mismatches.

NOTE: This script has been developed and tested specifically for human
mitochondrial DNA (hg38 chrM, 16,569 bp). While it may work with other
circular genomes, parameters and assumptions are optimized for human mtDNA
analysis. Use with other organisms or nuclear DNA has not been validated.

ALGORITHM OVERVIEW:
  1. Load kinetics CSV and identify ZMWs with mismatches (is_mismatch=True)
  2. Load CCS reads and subreads from indexed BAM files
  3. Align each subread to CCS using competitive alignment:
     - Test native orientation vs reverse complement
     - Assign to strand (fwd/rev) based on lower edit distance
  4. Calculate base composition (A/T/C/G/N counts) at each CCS position
  5. For ZMWs with mismatches, calculate Hellinger distance:
     - Expected distribution: genome-wide per-base error rates
     - Excludes mismatch positions from expected calculation
     - Processes BOTH strands (fwd and rev) for each mismatch ZMW

HELLINGER DISTANCE:
  H(P,Q) = (1/sqrt(2)) * sqrt(sum((sqrt(P(x)) - sqrt(Q(x)))^2))

  Where P is the observed base distribution from subreads and Q is the
  expected distribution calculated from all non-mismatch positions.
  Values range from 0 (identical) to 1 (completely different).

CIRCULAR GENOME HANDLING:
  For mitochondrial DNA, the reference is typically circularized by doubling
  (e.g., 33,138 bp = 16,569 x 2). All positions are normalized using modulo
  to map to the actual genome coordinates (0 to chrM_length-1).
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
OUTPUT FILES:
  {output}_subread_composition.csv
      Base composition for ALL CCS positions. Columns include:
      - zmw, strand, zmw_strand: Read identifiers
      - ccs_pos, ref_pos: Position in CCS and reference (normalized)
      - ccs_base, reference_base: Base calls
      - A_count, T_count, C_count, G_count, N_count: Subread base counts
      - total_subreads: Coverage at position
      - agreement_fraction: Fraction of subreads matching CCS base

  {output}_hellinger.csv
      Hellinger distance for mismatch ZMWs (both strands). Columns include:
      - zmw, strand, zmw_strand, ref_pos: Position identifiers
      - ccs_base, reference_base, q_score: Base and quality info
      - total_subreads, agreement_fraction: Coverage statistics
      - hellinger_distance: Calculated Hellinger distance (0-1)

  {output}_alignment_summary.tsv
      Per-subread alignment statistics including identity scores.

EXAMPLES:
  Basic usage (human mtDNA):
    python subread_hellinger_analysis.py \\
        sample.kinetics_with_anomalies_all_molecules.csv \\
        sample.subreads.bam \\
        sample.ccs.bam \\
        hg38_chrM_circularized.fa \\
        -o sample_hellinger

  With custom parameters:
    python subread_hellinger_analysis.py \\
        kinetics.csv subreads.bam ccs.bam ref.fa \\
        -o output_prefix \\
        -d /path/to/output/ \\
        --min-identity 0.7 \\
        --min-subreads 10 \\
        --chrm-length 16569

  Snakemake pipeline integration:
    python /opt/subread_hellinger_analysis.py \\
        {input.kinetics} \\
        {input.subread_bam} \\
        {input.ccs_bam} \\
        {input.reference} \\
        -o {params.out_prefix}

REQUIREMENTS:
  - Python 3.8+
  - pandas, numpy, pysam, edlib, tqdm

INPUT FILE REQUIREMENTS:
  - kinetics_csv: Must contain columns: zmw, ref_strand, tpos, is_mismatch
  - subread_bam: PacBio subreads BAM (indexed with .bai)
  - ccs_bam: PacBio CCS BAM with by-strand reads (indexed with .bai)
  - reference_fasta: FASTA file (optionally circularized for mtDNA)
        """
    )

    # Required arguments
    parser.add_argument("kinetics_csv",
        metavar="KINETICS_CSV",
        help="""Path to kinetics CSV from mt-mismatch pipeline. Expected file:
                kinetics_with_anomalies_all_molecules.csv. Must contain columns:
                zmw, ref_strand, tpos, is_mismatch.""")
    parser.add_argument("subread_bam",
        metavar="SUBREADS_BAM",
        help="""Path to indexed BAM file containing PacBio subreads.
                Must have corresponding .bai index file.""")
    parser.add_argument("ccs_bam",
        metavar="CCS_BAM",
        help="""Path to indexed BAM file containing CCS reads.
                Should be from PacBio --by-strand mode for proper strand
                identification. Must have corresponding .bai index file.""")
    parser.add_argument("reference_fasta",
        metavar="REFERENCE_FASTA",
        help="""Path to reference FASTA file. For mitochondrial DNA, this
                should be the circularized reference (doubled sequence).""")

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument("-o", "--output",
        metavar="PREFIX",
        default="step11_hellinger_analysis",
        help="""Output file prefix. Three files will be created:
                {PREFIX}_subread_composition.csv, {PREFIX}_hellinger.csv,
                {PREFIX}_alignment_summary.tsv (default: step11_hellinger_analysis)""")
    output_group.add_argument("-d", "--output-dir",
        metavar="DIR",
        default=".",
        help="Output directory. Created if it doesn't exist (default: current directory)")

    # Processing options
    proc_group = parser.add_argument_group('Processing Options')
    proc_group.add_argument("--min-subreads",
        type=int,
        metavar="N",
        default=5,
        help="""Minimum number of subreads at a position for inclusion in
                expected distribution calculation (default: 5)""")
    proc_group.add_argument("--min-identity",
        type=float,
        metavar="FRAC",
        default=0.5,
        help="""Minimum alignment identity (0.0-1.0) for a subread to be
                assigned to a strand. Subreads below this threshold are
                discarded (default: 0.5)""")
    proc_group.add_argument("--chrm-length",
        type=int,
        metavar="BP",
        default=16569,
        help="""Actual mitochondrial genome length in base pairs. Used for
                position normalization when reference is circularized by
                doubling. Human mtDNA = 16569 bp (default: 16569)""")
    proc_group.add_argument("-c", "--cores",
        type=int,
        metavar="N",
        default=N_CORES,
        help=f"""Number of CPU cores for parallel processing. Use 1 for
                serial mode (useful for debugging). Default: NSLOTS env var
                or all available cores (current: {N_CORES})""")

    args = parser.parse_args()

    # Validate input files
    for filepath in [args.kinetics_csv, args.subread_bam, args.ccs_bam, args.reference_fasta]:
        if not os.path.exists(filepath):
            logger.error(f"Input file not found: {filepath}")
            sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log parallel processing configuration
    n_cores = args.cores
    logger.info(f"Parallel processing: {n_cores} cores")

    try:
        # Step 1: Load kinetics data
        logger.info("=" * 60)
        logger.info("Step 1: Loading kinetics data")
        kinetics_df, mismatch_zmws, mismatch_positions = load_kinetics_data(
            args.kinetics_csv, args.chrm_length
        )

        # Get all ZMWs from kinetics
        all_zmws = kinetics_df['zmw'].unique().tolist()
        logger.info(f"  Total ZMWs in kinetics: {len(all_zmws)}")

        # Step 2: Load reference
        logger.info("=" * 60)
        logger.info("Step 2: Loading reference sequence")
        with pysam.FastaFile(args.reference_fasta) as fasta:
            ref_seq = fasta.fetch(fasta.references[0])
        logger.info(f"  Reference length: {len(ref_seq):,} bp")

        # Step 3: Load CCS reads
        logger.info("=" * 60)
        logger.info("Step 3: Loading CCS reads")
        ccs_reads = load_ccs_reads(args.ccs_bam, all_zmws, args.chrm_length)

        # Step 4: Load subreads
        logger.info("=" * 60)
        logger.info("Step 4: Loading subreads")
        subreads_by_zmw = load_subreads(args.subread_bam, all_zmws)

        # Step 5: Align subreads
        logger.info("=" * 60)
        logger.info("Step 5: Aligning subreads to strands")
        assigned_subreads = process_subread_alignment(
            all_zmws, subreads_by_zmw, ref_seq, args.chrm_length, args.min_identity,
            n_cores=n_cores
        )

        # Step 6: Calculate base composition for ALL positions
        logger.info("=" * 60)
        logger.info("Step 6: Calculating base composition")
        df_all = calculate_all_base_compositions(
            ccs_reads, assigned_subreads, ref_seq, n_cores=n_cores
        )

        if len(df_all) == 0:
            logger.error("No base composition data generated")
            sys.exit(1)

        # Step 7: Calculate Hellinger distance for mismatch ZMWs
        logger.info("=" * 60)
        logger.info("Step 7: Calculating Hellinger distance for mismatch ZMWs")
        df_hellinger = calculate_hellinger_for_mismatch_zmws(
            df_all, mismatch_zmws, mismatch_positions, n_cores=n_cores
        )

        # Step 8: Save outputs
        logger.info("=" * 60)
        logger.info("Step 8: Saving results")

        # Save subread composition CSV with base counts
        composition_path = output_dir / f"{args.output}_subread_composition.csv"
        df_all.to_csv(composition_path, index=False)
        logger.info(f"  Saved: {composition_path}")

        # Save Hellinger distance CSV
        hellinger_path = output_dir / f"{args.output}_hellinger.csv"
        df_hellinger.to_csv(hellinger_path, index=False)
        logger.info(f"  Saved: {hellinger_path}")

        # Save alignment summary
        summary_path = output_dir / f"{args.output}_alignment_summary.tsv"
        summary_data = []
        for sr in assigned_subreads:
            summary_data.append({
                'zmw': sr['zmw'],
                'strand': sr['strand'],
                'subread_name': sr['subread_name'],
                'identity': sr['identity']
            })
        pd.DataFrame(summary_data).to_csv(summary_path, sep='\t', index=False)
        logger.info(f"  Saved: {summary_path}")

        # Print summary
        print(f"\n{'=' * 60}")
        print("Generalized Subread Analysis Complete!")
        print(f"{'=' * 60}")
        print(f"Total ZMWs processed: {len(all_zmws)}")
        print(f"ZMWs with mismatches: {len(mismatch_zmws)}")
        print(f"Total positions with base composition: {len(df_all):,}")
        print(f"Positions with Hellinger distance: {len(df_hellinger):,}")
        print(f"\nOutput files:")
        print(f"  - {composition_path}")
        print(f"  - {hellinger_path}")
        print(f"  - {summary_path}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
