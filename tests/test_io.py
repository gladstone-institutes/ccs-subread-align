"""Tests for ccs_subread_align.io module."""

from pathlib import Path

import pytest

import numpy as np
import pandas as pd

from ccs_subread_align.io import (
    load_reference,
    load_subreads,
    read_parquet,
    scan_zmw_to_chrom,
    stream_ccs_reads,
    write_parquet,
)

DATA_DIR = Path(__file__).parent / "data"
REF_FASTA = DATA_DIR / "hg38_chrM_circularized_by_doubling.fa"
CCS_BAM = DATA_DIR / "test_cases.bam"
SUBREADS_BAM = DATA_DIR / "test_cases_subreads.bam"

CHRM_LENGTH = 16569


@pytest.fixture
def ccs_zmws():
    """Get ZMW IDs present in the test CCS BAM."""
    import pysam

    zmws = set()
    with pysam.AlignmentFile(str(CCS_BAM), "rb") as bam:
        for read in bam.fetch():
            if read.is_secondary or read.is_supplementary:
                continue
            parts = read.query_name.split("/")
            if len(parts) >= 2:
                try:
                    zmws.add(int(parts[1]))
                except ValueError:
                    pass
    return sorted(zmws)


# --- scan_zmw_to_chrom ---


@pytest.mark.skipif(not CCS_BAM.exists(), reason="Test BAM not available")
def test_scan_zmw_to_chrom_maps_every_requested_zmw(ccs_zmws):
    mapping = scan_zmw_to_chrom(str(CCS_BAM), ccs_zmws)
    assert isinstance(mapping, dict)
    assert set(mapping.keys()) == set(ccs_zmws)
    for zmw, chrom in mapping.items():
        assert isinstance(zmw, int)
        assert isinstance(chrom, str)
        assert len(chrom) > 0


@pytest.mark.skipif(not CCS_BAM.exists(), reason="Test BAM not available")
def test_scan_zmw_to_chrom_empty_zmw_list():
    assert scan_zmw_to_chrom(str(CCS_BAM), []) == {}


# --- stream_ccs_reads ---


@pytest.mark.skipif(not CCS_BAM.exists(), reason="Test BAM not available")
def test_stream_ccs_reads_yields_iterator(ccs_zmws):
    stream = stream_ccs_reads(str(CCS_BAM), ccs_zmws, CHRM_LENGTH)
    # Must be a one-shot iterator, not a list.
    assert not isinstance(stream, list)
    reads = list(stream)
    assert len(reads) > 0


@pytest.mark.skipif(not CCS_BAM.exists(), reason="Test BAM not available")
def test_stream_ccs_reads_structure(ccs_zmws):
    reads = list(stream_ccs_reads(str(CCS_BAM), ccs_zmws, CHRM_LENGTH))
    required_keys = {
        "zmw",
        "strand",
        "zmw_strand",
        "read_name",
        "query_sequence",
        "query_length",
        "cigartuples",
        "reference_start",
        "reference_name",
    }
    for read in reads:
        assert required_keys.issubset(read.keys())
        # query_to_ref is deliberately absent — parsed lazily in composition.
        assert "query_to_ref" not in read
        assert read["strand"] in ("fwd", "rev")
        assert read["query_length"] > 0
        assert len(read["query_sequence"]) == read["query_length"]


@pytest.mark.skipif(not CCS_BAM.exists(), reason="Test BAM not available")
def test_stream_ccs_reads_lazy_query_to_ref_matches_eager(ccs_zmws):
    """Parsing cigartuples in the caller matches the pre-v0.6.0 eager path."""
    from ccs_subread_align.alignment import parse_cigar_to_reference_map

    for read in stream_ccs_reads(str(CCS_BAM), ccs_zmws, CHRM_LENGTH):
        lazy = parse_cigar_to_reference_map(
            read["cigartuples"],
            read["reference_start"],
            read["query_length"],
            CHRM_LENGTH,
        )
        assert isinstance(lazy, np.ndarray)
        assert lazy.dtype == np.int32
        assert lazy.shape == (read["query_length"],)
        valid = lazy[lazy >= 0]
        assert (valid < CHRM_LENGTH).all()


@pytest.mark.skipif(not CCS_BAM.exists(), reason="Test BAM not available")
def test_stream_ccs_reads_empty_zmw_list():
    reads = list(stream_ccs_reads(str(CCS_BAM), [], CHRM_LENGTH))
    assert reads == []


# --- load_subreads ---


@pytest.mark.skipif(not SUBREADS_BAM.exists(), reason="Test BAM not available")
def test_load_subreads_returns_dict(ccs_zmws):
    result = load_subreads(str(SUBREADS_BAM), ccs_zmws)
    assert isinstance(result, dict)
    assert len(result) > 0


@pytest.mark.skipif(not SUBREADS_BAM.exists(), reason="Test BAM not available")
def test_load_subreads_structure(ccs_zmws):
    result = load_subreads(str(SUBREADS_BAM), ccs_zmws)
    for zmw, subreads in result.items():
        assert isinstance(zmw, int)
        assert isinstance(subreads, list)
        for sr in subreads:
            assert "read_name" in sr
            assert "query_sequence" in sr
            assert "query_length" in sr
            assert sr["query_length"] > 0


@pytest.mark.skipif(not SUBREADS_BAM.exists(), reason="Test BAM not available")
def test_load_subreads_empty_zmw_list():
    result = load_subreads(str(SUBREADS_BAM), [])
    assert result == {}


# --- load_reference ---


@pytest.mark.skipif(not REF_FASTA.exists(), reason="Test FASTA not available")
def test_load_reference_returns_dict():
    ref_seqs = load_reference(str(REF_FASTA))
    assert isinstance(ref_seqs, dict)
    assert len(ref_seqs) > 0


@pytest.mark.skipif(not REF_FASTA.exists(), reason="Test FASTA not available")
def test_load_reference_has_sequences():
    ref_seqs = load_reference(str(REF_FASTA))
    for name, seq in ref_seqs.items():
        assert isinstance(name, str)
        assert len(seq) > 0


# --- Parquet read/write ---


@pytest.fixture
def sample_composition_df():
    """Create a sample composition DataFrame matching the package output schema."""
    return pd.DataFrame(
        {
            "zmw": [1, 1, 2],
            "strand": ["fwd", "fwd", "rev"],
            "zmw_strand": ["1_fwd", "1_fwd", "2_rev"],
            "ccs_pos": [0, 1, 0],
            "ref_pos": [100, 101, 200],
            "ccs_base": ["A", "T", "C"],
            "reference_base": ["A", "T", "G"],
            "q_score": np.array([30, 25, 40], dtype=np.int64),
            "A_count": [5, 0, 1],
            "T_count": [0, 5, 0],
            "C_count": [0, 0, 4],
            "G_count": [0, 0, 0],
            "N_count": [0, 0, 0],
            "total_subreads": [5, 5, 5],
            "agreement_fraction": [1.0, 1.0, 0.8],
        }
    )


def test_parquet_round_trip(tmp_path, sample_composition_df):
    path = str(tmp_path / "test.parquet")
    write_parquet(sample_composition_df, path)
    result = read_parquet(path)
    pd.testing.assert_frame_equal(result, sample_composition_df)


def test_write_parquet_nonexistent_dir(sample_composition_df):
    with pytest.raises((FileNotFoundError, OSError)):
        write_parquet(sample_composition_df, "/nonexistent/dir/out.parquet")


def test_read_parquet_nonexistent_file():
    with pytest.raises((FileNotFoundError, OSError)):
        read_parquet("/nonexistent/file.parquet")
