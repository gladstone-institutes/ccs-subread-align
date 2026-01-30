"""Tests for ccs_subread_align.alignment module."""

import os
from pathlib import Path

import numpy as np
import pysam
import pytest

from ccs_subread_align.alignment import (
    _assign_single_subread,
    assign_subreads_to_strand,
    extract_zmw_from_name,
    parse_cigar_to_reference_map,
    parse_edlib_cigar_to_positions,
    reverse_complement,
)

DATA_DIR = Path(__file__).parent / "data"
REF_FASTA = DATA_DIR / "hg38_chrM_circularized_by_doubling.fa"
SUBREADS_BAM = DATA_DIR / "test_cases_subreads.bam"
CCS_BAM = DATA_DIR / "test_cases.bam"

CHRM_LENGTH = 16569


# --- reverse_complement ---


def test_reverse_complement_basic():
    assert reverse_complement("ATCG") == "CGAT"


def test_reverse_complement_single():
    assert reverse_complement("A") == "T"


def test_reverse_complement_with_n():
    assert reverse_complement("ANCG") == "CGNT"


def test_reverse_complement_empty():
    assert reverse_complement("") == ""


def test_reverse_complement_involution():
    seq = "ATCGATCG"
    assert reverse_complement(reverse_complement(seq)) == seq


# --- extract_zmw_from_name ---


def test_extract_zmw_valid():
    assert extract_zmw_from_name("m64020_200101/12345/ccs") == 12345


def test_extract_zmw_two_parts():
    assert extract_zmw_from_name("movie/999") == 999


def test_extract_zmw_no_slash():
    assert extract_zmw_from_name("noslash") is None


def test_extract_zmw_non_numeric():
    assert extract_zmw_from_name("movie/abc/ccs") is None


# --- parse_cigar_to_reference_map ---


def test_parse_cigar_to_reference_map_simple_match():
    # 5M: 5 matches starting at ref pos 0
    cigartuples = [(0, 5)]  # 5M
    result = parse_cigar_to_reference_map(cigartuples, 0, chrM_length=100)
    assert result[0] == 0
    assert result[4] == 4
    assert len([k for k, v in result.items() if v is not None]) == 5


def test_parse_cigar_to_reference_map_with_insertion():
    # 3M1I2M
    cigartuples = [(0, 3), (1, 1), (0, 2)]
    result = parse_cigar_to_reference_map(cigartuples, 0, chrM_length=100)
    assert result[0] == 0
    assert result[1] == 1
    assert result[2] == 2
    assert result[3] is None  # insertion
    assert result[4] == 3
    assert result[5] == 4


def test_parse_cigar_to_reference_map_with_deletion():
    # 3M2D3M: query has 6 bases, ref has 8 positions
    cigartuples = [(0, 3), (2, 2), (0, 3)]
    result = parse_cigar_to_reference_map(cigartuples, 0, chrM_length=100)
    assert result[0] == 0
    assert result[2] == 2
    assert result[3] == 5  # after 2-base deletion
    assert result[5] == 7


def test_parse_cigar_to_reference_map_normalization():
    # Position wraps around chrM_length
    cigartuples = [(0, 5)]
    result = parse_cigar_to_reference_map(cigartuples, 98, chrM_length=100)
    assert result[0] == 98
    assert result[1] == 99
    assert result[2] == 0  # wraps around
    assert result[3] == 1


def test_parse_cigar_to_reference_map_soft_clip():
    # 2S3M: 2 soft-clipped bases then 3 matches
    cigartuples = [(4, 2), (0, 3)]
    result = parse_cigar_to_reference_map(cigartuples, 10, chrM_length=100)
    # Soft-clipped bases should be None (not aligned to ref)
    assert result[0] is None
    assert result[1] is None
    assert result[2] == 10
    assert result[4] == 12


# --- parse_edlib_cigar_to_positions ---


def test_parse_edlib_cigar_simple_match():
    result = parse_edlib_cigar_to_positions("5=", "ATCGA", 0, chrM_length=100)
    assert len(result) == 5
    np.testing.assert_array_equal(result, [0, 1, 2, 3, 4])


def test_parse_edlib_cigar_with_mismatch():
    result = parse_edlib_cigar_to_positions("3=1X2=", "ATCGAT", 0, chrM_length=100)
    np.testing.assert_array_equal(result, [0, 1, 2, 3, 4, 5])


def test_parse_edlib_cigar_with_insertion():
    result = parse_edlib_cigar_to_positions("3=1I2=", "ATCGAT", 0, chrM_length=100)
    assert result[0] == 0
    assert result[2] == 2
    assert result[3] == -1  # insertion
    assert result[4] == 3
    assert result[5] == 4


def test_parse_edlib_cigar_with_deletion():
    result = parse_edlib_cigar_to_positions("3=2D3=", "ATCGAT", 0, chrM_length=100)
    assert result[0] == 0
    assert result[2] == 2
    assert result[3] == 5  # after deletion
    assert result[5] == 7


def test_parse_edlib_cigar_normalization():
    result = parse_edlib_cigar_to_positions("5=", "ATCGA", 98, chrM_length=100)
    np.testing.assert_array_equal(result, [98, 99, 0, 1, 2])


def test_parse_edlib_cigar_empty():
    result = parse_edlib_cigar_to_positions("", "ATCGA", 0, chrM_length=100)
    np.testing.assert_array_equal(result, [-1, -1, -1, -1, -1])


# --- assign_subreads_to_strand ---


@pytest.fixture
def ref_seq():
    with pysam.FastaFile(str(REF_FASTA)) as fasta:
        return fasta.fetch(fasta.references[0])


def test_assign_strand_fwd(ref_seq):
    """A subread taken from the forward strand should align as fwd."""
    # Use a chunk of the reference as a fake subread (should align perfectly fwd)
    subread = ref_seq[100:200]
    result = assign_subreads_to_strand(subread, ref_seq, CHRM_LENGTH)
    assert result is not None
    assert result["strand"] == "fwd"
    assert result["identity"] == 1.0


def test_assign_strand_rev(ref_seq):
    """RC of a reference chunk should align as rev."""
    subread = reverse_complement(ref_seq[100:200])
    result = assign_subreads_to_strand(subread, ref_seq, CHRM_LENGTH)
    assert result is not None
    assert result["strand"] == "rev"
    assert result["identity"] == 1.0


def test_assign_strand_low_identity(ref_seq):
    """Random sequence should fail identity threshold."""
    subread = "A" * 100
    result = assign_subreads_to_strand(subread, ref_seq, CHRM_LENGTH, min_identity=0.99)
    # Poly-A won't match well enough
    assert result is None


def test_assign_strand_position_map_valid(ref_seq):
    """Position map should contain valid normalized positions."""
    subread = ref_seq[500:600]
    result = assign_subreads_to_strand(subread, ref_seq, CHRM_LENGTH)
    assert result is not None
    pm = result["position_map"]
    assert len(pm) == 100
    assert all(0 <= p < CHRM_LENGTH for p in pm if p >= 0)


# --- _assign_single_subread ---


def test_assign_single_subread_short():
    """Subreads shorter than 25bp should be skipped."""
    result = _assign_single_subread(
        {"zmw": 1, "read_name": "test", "query_sequence": "ATCG", "_ref_seq": "ATCG" * 100},
        chrM_length=100,
        min_identity=0.5,
    )
    assert result is None


def test_assign_single_subread_valid(ref_seq):
    """Valid subread should return full result dict."""
    subread_dict = {
        "zmw": 42,
        "read_name": "movie/42/0_100",
        "query_sequence": ref_seq[200:300],
        "_ref_seq": ref_seq,
    }
    result = _assign_single_subread(
        subread_dict,
        chrM_length=CHRM_LENGTH,
        min_identity=0.5,
    )
    assert result is not None
    assert result["zmw"] == 42
    assert result["strand"] in ("fwd", "rev")
    assert "zmw_strand" in result
    assert "identity" in result


# --- Integration with real BAM data ---


@pytest.mark.skipif(not SUBREADS_BAM.exists(), reason="Test BAM not available")
def test_assign_real_subreads(ref_seq):
    """Test alignment with actual subreads from test BAM."""
    subreads = []
    with pysam.AlignmentFile(str(SUBREADS_BAM), "rb", check_sq=False) as bam:
        for i, read in enumerate(bam.fetch(until_eof=True)):
            if i >= 5:
                break
            subreads.append(read.query_sequence)

    aligned_count = 0
    for seq in subreads:
        if len(seq) >= 25:
            result = assign_subreads_to_strand(seq, ref_seq, CHRM_LENGTH)
            if result is not None:
                aligned_count += 1
                assert result["strand"] in ("fwd", "rev")
                assert 0.0 <= result["identity"] <= 1.0
    # At least some subreads should align successfully
    assert aligned_count > 0
