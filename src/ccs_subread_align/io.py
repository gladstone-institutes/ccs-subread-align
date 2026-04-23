"""I/O functions for loading PacBio CCS reads, subreads, and Parquet data."""

import logging
import resource
import sys
import time
from collections import defaultdict
from typing import Dict, Iterator, List

import numpy as np
import pandas as pd
import pysam
from tqdm import tqdm

from ccs_subread_align.alignment import extract_zmw_from_name

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


def scan_zmw_to_chrom(
    ccs_bam_path: str, zmw_list: List[int]
) -> Dict[int, str]:
    """Light BAM scan that returns the ``{zmw: reference_name}`` mapping.

    Alignment needs this before composition runs, so it's broken out of the
    full CCS streamer. Reads only the fields needed for the mapping, leaving
    sequences, qualities, and CIGAR unparsed.
    """
    logger.info(f"Scanning zmw→chrom from: {ccs_bam_path}")
    zmw_set = set(zmw_list)
    mapping: Dict[int, str] = {}
    t0 = time.monotonic()
    with pysam.AlignmentFile(ccs_bam_path, "rb") as bam:
        for read in tqdm(bam.fetch(), desc="Scan zmw→chrom", unit="reads"):
            zmw = extract_zmw_from_name(read.query_name)
            if zmw in zmw_set and zmw not in mapping:
                mapping[zmw] = read.reference_name
    logger.info(
        f"Scan complete: {len(mapping)} zmws mapped in "
        f"{time.monotonic() - t0:.1f}s"
    )
    return mapping


def stream_ccs_reads(
    ccs_bam_path: str,
    zmw_list: List[int],
    chrM_length: int,
) -> Iterator[Dict]:
    """Yield CCS read dicts from BAM one-at-a-time.

    Replaces the pre-v0.6.0 ``load_ccs_reads`` list return. `query_to_ref`
    is NOT precomputed — each yielded dict carries ``cigartuples``,
    ``reference_start``, and ``query_length`` so the composition worker
    parses the CIGAR lazily (see ``composition.calculate_base_composition``).
    This is what keeps the parent process out of the ~21 GB regime at full
    scale (240k × ~68 kB query_to_ref arrays).

    ``chrM_length`` is carried in each yielded dict so downstream workers
    have it without a closure over this function's scope.
    """
    logger.info(f"Streaming CCS reads from: {ccs_bam_path}")
    zmw_set = set(zmw_list)
    target_n = len(zmw_set)
    t0 = time.monotonic()
    yielded = 0
    with pysam.AlignmentFile(ccs_bam_path, "rb") as bam:
        for read in tqdm(bam.fetch(), desc="Streaming CCS reads", unit="reads"):
            zmw = extract_zmw_from_name(read.query_name)
            if zmw not in zmw_set:
                continue
            strand = "rev" if read.is_reverse else "fwd"
            yield {
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
                "chrM_length": chrM_length,
            }
            yielded += 1
    logger.info(
        f"Streamed {yielded}/{target_n} CCS records in "
        f"{time.monotonic() - t0:.1f}s (peak RSS: {_peak_rss_mb():.0f} MB)"
    )


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
