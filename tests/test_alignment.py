"""Tests for ccs_subread_align.alignment module."""

import os
from pathlib import Path

import numpy as np
import pysam
import pytest

from aligntools import Cigar, CigarActions

from ccs_subread_align.alignment import (
    _assign_single_subread,
    assign_subreads_to_strand,
    extract_zmw_from_name,
    parse_cigar_to_reference_map,
    parse_edlib_cigar_to_positions,
    reverse_complement,
)

# aligntools is a dev-only dependency used here as a ground-truth reference
# for the hand-rolled CIGAR walkers in alignment.py.
_LEGACY_OP_TO_CHAR = {0: "M", 1: "I", 2: "D", 3: "N", 4: "S", 5: "H", 6: "P", 7: "=", 8: "X"}
_LEGACY_OP_TO_ACTION = {
    0: CigarActions.MATCH,
    1: CigarActions.INSERT,
    2: CigarActions.DELETE,
    3: CigarActions.SKIPPED,
    4: CigarActions.SOFT_CLIPPED,
    5: CigarActions.HARD_CLIPPED,
    6: CigarActions.PADDING,
    7: CigarActions.SEQ_MATCH,
    8: CigarActions.MISMATCH,
}


def _aligntools_position_map(cigartuples, ref_start, qlen, chrM_length):
    """Build a position map via aligntools (ground-truth oracle for tests)."""
    out = np.full(qlen, -1, dtype=np.int32)
    if not cigartuples:
        return out
    cigar = Cigar([(length, _LEGACY_OP_TO_ACTION[op]) for op, length in cigartuples])
    for qpos, rpos in cigar.coordinate_mapping.query_to_ref.items():
        if 0 <= qpos < qlen:
            out[qpos] = (rpos + ref_start) % chrM_length
    return out


def _aligntools_edlib_position_map(cigar_str, qlen, ref_start, chrM_length):
    """Build an edlib-CIGAR position map via aligntools."""
    out = np.full(qlen, -1, dtype=np.int32)
    if not cigar_str:
        return out
    parsed = Cigar.coerce(cigar_str)
    for qpos, rpos in parsed.coordinate_mapping.query_to_ref.items():
        if 0 <= qpos < qlen:
            out[qpos] = (rpos + ref_start) % chrM_length
    return out

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
    result = parse_cigar_to_reference_map([(0, 5)], 0, 5, chrM_length=100)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int32
    np.testing.assert_array_equal(result, [0, 1, 2, 3, 4])


def test_parse_cigar_to_reference_map_with_insertion():
    # 3M1I2M, query_length=6
    result = parse_cigar_to_reference_map(
        [(0, 3), (1, 1), (0, 2)], 0, 6, chrM_length=100
    )
    np.testing.assert_array_equal(result, [0, 1, 2, -1, 3, 4])


def test_parse_cigar_to_reference_map_with_deletion():
    # 3M2D3M: query has 6 bases, ref has 8 positions
    result = parse_cigar_to_reference_map(
        [(0, 3), (2, 2), (0, 3)], 0, 6, chrM_length=100
    )
    np.testing.assert_array_equal(result, [0, 1, 2, 5, 6, 7])


def test_parse_cigar_to_reference_map_normalization():
    # Position wraps around chrM_length
    result = parse_cigar_to_reference_map([(0, 5)], 98, 5, chrM_length=100)
    np.testing.assert_array_equal(result, [98, 99, 0, 1, 2])


def test_parse_cigar_to_reference_map_soft_clip():
    # 2S3M: 2 soft-clipped bases then 3 matches
    result = parse_cigar_to_reference_map([(4, 2), (0, 3)], 10, 5, chrM_length=100)
    np.testing.assert_array_equal(result, [-1, -1, 10, 11, 12])


def test_parse_cigar_to_reference_map_empty_cigartuples():
    result = parse_cigar_to_reference_map([], 0, 10, chrM_length=100)
    assert result.shape == (10,)
    assert (result == -1).all()


def test_parse_cigar_to_reference_map_hard_clip():
    # 5H3M: hard-clipped bases are absent from the query (query_length=3)
    result = parse_cigar_to_reference_map(
        [(5, 5), (0, 3)], 0, 3, chrM_length=100
    )
    np.testing.assert_array_equal(result, [0, 1, 2])


# --- Equivalence with aligntools (ground-truth oracle) ---


@pytest.mark.parametrize(
    "cigartuples,ref_start,qlen",
    [
        ([(0, 5)], 0, 5),
        ([(0, 3), (1, 1), (0, 2)], 0, 6),
        ([(0, 3), (2, 2), (0, 3)], 0, 6),
        ([(4, 2), (0, 3)], 10, 5),
        ([(4, 2), (0, 3), (4, 1)], 10, 6),
        ([(5, 5), (0, 3)], 0, 3),
        ([(0, 3), (5, 5)], 0, 3),
        ([(5, 2), (4, 3), (0, 10), (4, 2), (5, 1)], 50, 15),
        ([(0, 10), (1, 2), (0, 5), (2, 3), (0, 8), (4, 3)], 200, 28),
        ([(7, 3), (8, 1), (7, 2)], 0, 6),
        ([(0, 3), (3, 10), (0, 3)], 0, 6),
        ([(0, 5)], 16567, 5),  # wraps chrM
    ],
)
def test_parse_cigar_matches_aligntools_synthetic(cigartuples, ref_start, qlen):
    """Our numpy CIGAR walker must match aligntools on synthetic CIGARs."""
    ours = parse_cigar_to_reference_map(
        cigartuples, ref_start, qlen, chrM_length=CHRM_LENGTH
    )
    oracle = _aligntools_position_map(cigartuples, ref_start, qlen, CHRM_LENGTH)
    np.testing.assert_array_equal(ours, oracle)


@pytest.mark.skipif(not CCS_BAM.exists(), reason="Test BAM not available")
def test_parse_cigar_matches_aligntools_on_real_bam():
    """On every real BAM CIGAR, our output must match the aligntools oracle."""
    with pysam.AlignmentFile(str(CCS_BAM), "rb") as bam:
        for read in bam.fetch():
            if not read.cigartuples:
                continue
            ref_start = read.reference_start
            qlen = read.query_length

            ours = parse_cigar_to_reference_map(
                read.cigartuples, ref_start, qlen, chrM_length=CHRM_LENGTH
            )
            oracle = _aligntools_position_map(
                list(read.cigartuples), ref_start, qlen, CHRM_LENGTH
            )
            np.testing.assert_array_equal(ours, oracle)


@pytest.mark.parametrize(
    "cigar_str,query_seq,ref_start",
    [
        ("5=", "ATCGA", 0),
        ("3=1X2=", "ATCGAT", 0),
        ("3=1I2=", "ATCGAT", 0),
        ("3=2D3=", "ATCGAT", 0),
        ("5=", "ATCGA", 16567),  # wraps chrM
        ("100=5X10I20=", "A" * 135, 500),
        ("50=1D50=", "A" * 100, 1000),
        ("10I100=", "A" * 110, 0),
    ],
)
def test_parse_edlib_cigar_matches_aligntools_synthetic(cigar_str, query_seq, ref_start):
    """Our edlib-CIGAR walker must match aligntools on synthetic strings."""
    ours = parse_edlib_cigar_to_positions(
        cigar_str, query_seq, ref_start, chrM_length=CHRM_LENGTH
    )
    oracle = _aligntools_edlib_position_map(
        cigar_str, len(query_seq), ref_start, CHRM_LENGTH
    )
    np.testing.assert_array_equal(ours, oracle)


@pytest.mark.skipif(not REF_FASTA.exists(), reason="Test FASTA not available")
def test_parse_edlib_cigar_matches_aligntools_on_real_alignments():
    """Run edlib on real reference chunks and compare our parser to aligntools."""
    import edlib

    with pysam.FastaFile(str(REF_FASTA)) as fasta:
        ref = fasta.fetch(fasta.references[0])

    rng = np.random.default_rng(0)
    for _ in range(20):
        start = int(rng.integers(0, len(ref) - 500))
        length = int(rng.integers(80, 400))
        query = ref[start : start + length]
        # Inject a couple of mutations so edlib emits X/I/D operations.
        q_list = list(query)
        for pos in rng.integers(0, len(q_list), size=3):
            q_list[int(pos)] = "N"
        query = "".join(q_list)

        result = edlib.align(query, ref, mode="HW", task="path")
        if not result["cigar"] or not result["locations"]:
            continue
        ref_start = result["locations"][0][0]

        ours = parse_edlib_cigar_to_positions(
            result["cigar"], query, ref_start, chrM_length=CHRM_LENGTH
        )
        oracle = _aligntools_edlib_position_map(
            result["cigar"], len(query), ref_start, CHRM_LENGTH
        )
        np.testing.assert_array_equal(ours, oracle)


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
    assert "edit_distance_margin" not in result

    # With report_margin=True, margin should be present
    result_margin = assign_subreads_to_strand(subread, ref_seq, CHRM_LENGTH, report_margin=True)
    assert result_margin is not None
    assert result_margin["edit_distance_margin"] >= 0


def test_assign_strand_rev(ref_seq):
    """RC of a reference chunk should align as rev."""
    subread = reverse_complement(ref_seq[100:200])
    result = assign_subreads_to_strand(subread, ref_seq, CHRM_LENGTH)
    assert result is not None
    assert result["strand"] == "rev"
    assert result["identity"] == 1.0
    assert "edit_distance_margin" not in result

    result_margin = assign_subreads_to_strand(subread, ref_seq, CHRM_LENGTH, report_margin=True)
    assert result_margin is not None
    assert result_margin["edit_distance_margin"] >= 0


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
    assert "edit_distance_margin" not in result

    result_margin = _assign_single_subread(
        subread_dict,
        chrM_length=CHRM_LENGTH,
        min_identity=0.5,
        report_margin=True,
    )
    assert result_margin is not None
    assert result_margin["edit_distance_margin"] >= 0


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
            result = assign_subreads_to_strand(seq, ref_seq, CHRM_LENGTH, report_margin=True)
            if result is not None:
                aligned_count += 1
                assert result["strand"] in ("fwd", "rev")
                assert 0.0 <= result["identity"] <= 1.0
                assert result["edit_distance_margin"] >= 0
    # At least some subreads should align successfully
    assert aligned_count > 0
