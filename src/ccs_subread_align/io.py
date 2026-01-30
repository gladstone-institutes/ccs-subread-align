"""BAM/FASTA I/O functions for loading PacBio CCS reads and subreads."""

import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pysam

from ccs_subread_align.alignment import extract_zmw_from_name, parse_cigar_to_reference_map

logger = logging.getLogger(__name__)


def load_reference(fasta_path: str) -> Dict[str, str]:
    """Load reference FASTA as a dictionary of {name: sequence}.

    Args:
        fasta_path: Path to reference FASTA file

    Returns:
        Dictionary mapping sequence names to their sequences
    """
    logger.info(f"Loading reference from: {fasta_path}")
    with pysam.FastaFile(fasta_path) as fasta:
        ref_seqs = {name: fasta.fetch(name) for name in fasta.references}
    logger.info(f"Loaded {len(ref_seqs)} reference sequences: {list(ref_seqs.keys())}")
    return ref_seqs


def load_ccs_reads(
    ccs_bam_path: str, zmw_list: List[int], chrM_length: int
) -> List[Dict]:
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

    with pysam.AlignmentFile(ccs_bam_path, "rb") as bam:
        for read in bam.fetch():
            zmw = extract_zmw_from_name(read.query_name)
            if zmw in zmw_set:
                strand = "rev" if read.is_reverse else "fwd"

                ccs_reads.append(
                    {
                        "zmw": zmw,
                        "strand": strand,
                        "zmw_strand": f"{zmw}_{strand}",
                        "read_name": read.query_name,
                        "sam_flag": read.flag,
                        "reference_start": read.reference_start,
                        "query_sequence": read.query_sequence,
                        "query_length": read.query_length,
                        "cigartuples": read.cigartuples,
                        "quality_array": (
                            np.array(read.query_qualities)
                            if read.query_qualities
                            else np.zeros(read.query_length)
                        ),
                        "mapping_quality": read.mapping_quality,
                        "reference_name": read.reference_name,
                    }
                )

    for ccs in ccs_reads:
        if ccs["cigartuples"]:
            ccs["query_to_ref_map"] = parse_cigar_to_reference_map(
                ccs["cigartuples"], ccs["reference_start"], chrM_length
            )
        else:
            ccs["query_to_ref_map"] = {}

    logger.info(f"Loaded {len(ccs_reads)} CCS reads")
    return ccs_reads


def load_subreads(
    subreads_bam_path: str, zmw_list: List[int]
) -> Dict[int, List[Dict]]:
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

    with pysam.AlignmentFile(subreads_bam_path, "rb", check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            zmw = extract_zmw_from_name(read.query_name)
            if zmw in zmw_set:
                subreads_by_zmw[zmw].append(
                    {
                        "read_name": read.query_name,
                        "zmw": zmw,
                        "query_sequence": read.query_sequence,
                        "query_length": read.query_length,
                    }
                )

    total = sum(len(v) for v in subreads_by_zmw.values())
    logger.info(f"Loaded {total} subreads across {len(subreads_by_zmw)} ZMWs")
    return dict(subreads_by_zmw)
