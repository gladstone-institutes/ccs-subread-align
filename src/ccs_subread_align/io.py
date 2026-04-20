"""I/O functions for loading PacBio CCS reads, subreads, and Parquet data."""

import logging
import resource
import sys
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import pysam
from tqdm import tqdm

from ccs_subread_align.alignment import extract_zmw_from_name, parse_cigar_to_reference_map

logger = logging.getLogger(__name__)


def _peak_rss_mb() -> float:
    """Peak RSS of the current process in MB.

    ru_maxrss is bytes on macOS, kilobytes on Linux.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


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
    ccs_bam_path: str,
    zmw_list: List[int],
    chrM_length: int,
) -> List[Dict]:
    """
    Load CCS reads from BAM file.

    Runs in two phases: (1) iterate the BAM and collect per-read fields,
    (2) build a per-read int32 ``query_to_ref`` array from the CIGAR.

    Args:
        ccs_bam_path: Path to CCS BAM file
        zmw_list: List of ZMWs to load
        chrM_length: Mitochondrial genome length for position normalization

    Returns:
        List of CCS read dictionaries
    """
    logger.info(f"Loading CCS reads from: {ccs_bam_path}")
    zmw_set = set(zmw_list)
    target_n = len(zmw_set)
    ccs_reads = []

    logger.info(f"Phase 1: iterating BAM for {target_n} target ZMWs")
    t0 = time.monotonic()
    with pysam.AlignmentFile(ccs_bam_path, "rb") as bam:
        for read in tqdm(bam.fetch(), desc="Phase 1: scanning BAM", unit="reads"):
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

    k = len(ccs_reads)
    logger.info(
        f"Phase 1 complete: read {k}/{target_n} CCS records in "
        f"{time.monotonic() - t0:.1f}s (peak RSS: {_peak_rss_mb():.0f} MB)"
    )

    logger.info(f"Phase 2: building query→reference maps for {k} CCS reads")
    t1 = time.monotonic()
    for ccs in tqdm(ccs_reads, desc="Phase 2: CIGAR parse"):
        ccs["query_to_ref"] = parse_cigar_to_reference_map(
            ccs["cigartuples"],
            ccs["reference_start"],
            ccs["query_length"],
            chrM_length,
        )

    logger.info(
        f"Phase 2 complete in {time.monotonic() - t1:.1f}s "
        f"(peak RSS: {_peak_rss_mb():.0f} MB)"
    )
    logger.info(f"Loaded {k} CCS reads")
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
    target_n = len(zmw_set)
    subreads_by_zmw = defaultdict(list)

    logger.info(f"Phase 1: iterating BAM for {target_n} target ZMWs")
    t0 = time.monotonic()
    with pysam.AlignmentFile(subreads_bam_path, "rb", check_sq=False) as bam:
        for read in tqdm(
            bam.fetch(until_eof=True),
            desc="Phase 1: scanning subreads BAM",
            unit="reads",
        ):
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
    logger.info(
        f"Phase 1 complete: read {total} subreads across "
        f"{len(subreads_by_zmw)} ZMWs in {time.monotonic() - t0:.1f}s "
        f"(peak RSS: {_peak_rss_mb():.0f} MB)"
    )
    logger.info(f"Loaded {total} subreads across {len(subreads_by_zmw)} ZMWs")
    return dict(subreads_by_zmw)


def write_parquet(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to a Parquet file using pyarrow.

    Args:
        df: DataFrame to write
        path: Output file path
    """
    df.to_parquet(path, engine="pyarrow", index=False)
    logger.info(f"Wrote {len(df)} rows to: {path}")


def read_parquet(path: str) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame using pyarrow.

    Args:
        path: Path to Parquet file

    Returns:
        DataFrame with the Parquet file contents
    """
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info(f"Read {len(df)} rows from: {path}")
    return df
