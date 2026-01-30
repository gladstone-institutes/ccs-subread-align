"""Tests for ccs_subread_align.io module."""

from pathlib import Path

import pytest

from ccs_subread_align.io import load_ccs_reads, load_reference, load_subreads

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
            parts = read.query_name.split("/")
            if len(parts) >= 2:
                try:
                    zmws.add(int(parts[1]))
                except ValueError:
                    pass
    return sorted(zmws)


# --- load_ccs_reads ---


@pytest.mark.skipif(not CCS_BAM.exists(), reason="Test BAM not available")
def test_load_ccs_reads_returns_list(ccs_zmws):
    reads = load_ccs_reads(str(CCS_BAM), ccs_zmws, CHRM_LENGTH)
    assert isinstance(reads, list)
    assert len(reads) > 0


@pytest.mark.skipif(not CCS_BAM.exists(), reason="Test BAM not available")
def test_load_ccs_reads_structure(ccs_zmws):
    reads = load_ccs_reads(str(CCS_BAM), ccs_zmws, CHRM_LENGTH)
    required_keys = {
        "zmw",
        "strand",
        "zmw_strand",
        "read_name",
        "query_sequence",
        "query_length",
        "query_to_ref_map",
    }
    for read in reads:
        assert required_keys.issubset(read.keys())
        assert read["strand"] in ("fwd", "rev")
        assert read["query_length"] > 0
        assert len(read["query_sequence"]) == read["query_length"]


@pytest.mark.skipif(not CCS_BAM.exists(), reason="Test BAM not available")
def test_load_ccs_reads_empty_zmw_list():
    reads = load_ccs_reads(str(CCS_BAM), [], CHRM_LENGTH)
    assert reads == []


@pytest.mark.skipif(not CCS_BAM.exists(), reason="Test BAM not available")
def test_load_ccs_reads_query_to_ref_map(ccs_zmws):
    reads = load_ccs_reads(str(CCS_BAM), ccs_zmws, CHRM_LENGTH)
    for read in reads:
        ref_map = read["query_to_ref_map"]
        for qpos, rpos in ref_map.items():
            if rpos is not None:
                assert 0 <= rpos < CHRM_LENGTH


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


# --- CCS reads include reference_name ---


@pytest.mark.skipif(not CCS_BAM.exists(), reason="Test BAM not available")
def test_load_ccs_reads_has_reference_name(ccs_zmws):
    reads = load_ccs_reads(str(CCS_BAM), ccs_zmws, CHRM_LENGTH)
    for read in reads:
        assert "reference_name" in read
        assert isinstance(read["reference_name"], str)
