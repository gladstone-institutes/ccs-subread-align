"""Tests for ccs_subread_align.composition module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ccs_subread_align.composition import (
    calculate_all_base_compositions,
    calculate_base_composition,
)

DATA_DIR = Path(__file__).parent / "data"
REF_FASTA = DATA_DIR / "hg38_chrM_circularized_by_doubling.fa"
CCS_BAM = DATA_DIR / "test_cases.bam"
SUBREADS_BAM = DATA_DIR / "test_cases_subreads.bam"

CHRM_LENGTH = 16569

EXPECTED_COLUMNS = {
    "zmw",
    "strand",
    "zmw_strand",
    "ccs_pos",
    "ref_pos",
    "ccs_base",
    "reference_base",
    "q_score",
    "A_count",
    "T_count",
    "C_count",
    "G_count",
    "N_count",
    "total_subreads",
    "agreement_fraction",
}


def _make_ccs_read(seq="ACGT", quality=None, zmw=1, strand="fwd", ref_start=0):
    """Build a minimal CCS read dict for testing."""
    if quality is None:
        quality = np.array([30] * len(seq))
    query_to_ref = {i: (ref_start + i) % CHRM_LENGTH for i in range(len(seq))}
    return {
        "zmw": zmw,
        "strand": strand,
        "zmw_strand": f"{zmw}_{strand}",
        "query_sequence": seq,
        "query_length": len(seq),
        "quality_array": quality,
        "query_to_ref_map": query_to_ref,
    }


def _make_subread(seq, position_map, zmw=1, strand="fwd"):
    """Build a minimal assigned subread dict for testing."""
    return {
        "zmw": zmw,
        "strand": strand,
        "zmw_strand": f"{zmw}_{strand}",
        "aligned_sequence": seq,
        "position_map": np.array(position_map, dtype=np.int32),
        "identity": 0.95,
        "subread_name": f"movie/{zmw}/0_100",
    }


# --- calculate_base_composition ---


def test_base_composition_columns():
    ccs = _make_ccs_read("ACGT")
    sr = _make_subread("ACGT", [0, 1, 2, 3])
    ref_seq = "ACGT" * 5000
    df = calculate_base_composition(ccs, [sr], ref_seq, CHRM_LENGTH)
    assert EXPECTED_COLUMNS == set(df.columns)
    assert len(df) == 4


def test_base_composition_perfect_agreement():
    ccs = _make_ccs_read("ACGT")
    sr1 = _make_subread("ACGT", [0, 1, 2, 3])
    sr2 = _make_subread("ACGT", [0, 1, 2, 3])
    ref_seq = "ACGT" * 5000
    df = calculate_base_composition(ccs, [sr1, sr2], ref_seq, CHRM_LENGTH)
    assert (df["agreement_fraction"] == 1.0).all()
    assert (df["total_subreads"] == 2).all()


def test_base_composition_disagreement():
    ccs = _make_ccs_read("AAAA")
    # Subread has T at first position
    sr = _make_subread("TAAA", [0, 1, 2, 3])
    ref_seq = "A" * 20000
    df = calculate_base_composition(ccs, [sr], ref_seq, CHRM_LENGTH)
    assert df.iloc[0]["agreement_fraction"] == 0.0
    assert df.iloc[0]["T_count"] == 1
    assert df.iloc[0]["A_count"] == 0
    assert df.iloc[1]["agreement_fraction"] == 1.0


def test_base_composition_no_subreads():
    ccs = _make_ccs_read("ACGT")
    ref_seq = "ACGT" * 5000
    df = calculate_base_composition(ccs, [], ref_seq, CHRM_LENGTH)
    assert len(df) == 4
    assert (df["total_subreads"] == 0).all()
    assert (df["agreement_fraction"] == 0.0).all()


def test_base_composition_insertion_positions():
    ccs = _make_ccs_read("ACGT")
    # Override: position 2 is an insertion (None -> -1)
    ccs["query_to_ref_map"] = {0: 0, 1: 1, 2: None, 3: 3}
    sr = _make_subread("ACGT", [0, 1, -1, 3])
    ref_seq = "ACGT" * 5000
    df = calculate_base_composition(ccs, [sr], ref_seq, CHRM_LENGTH)
    assert df.iloc[2]["ref_pos"] == -1
    assert df.iloc[2]["total_subreads"] == 0


# --- calculate_all_base_compositions ---


@pytest.mark.skipif(
    not all(p.exists() for p in [CCS_BAM, SUBREADS_BAM, REF_FASTA]),
    reason="Test data not available",
)
def test_calculate_all_base_compositions_integration():
    from ccs_subread_align.alignment import process_subread_alignment
    from ccs_subread_align.io import load_ccs_reads, load_reference, load_subreads

    ref_seqs = load_reference(str(REF_FASTA))

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
    zmw_list = sorted(zmws)

    ccs_reads = load_ccs_reads(str(CCS_BAM), zmw_list, CHRM_LENGTH)
    subreads_by_zmw = load_subreads(str(SUBREADS_BAM), zmw_list)
    zmw_to_chrom = {ccs["zmw"]: ccs["reference_name"] for ccs in ccs_reads}

    assigned = process_subread_alignment(
        zmw_list,
        subreads_by_zmw,
        ref_seqs,
        zmw_to_chrom,
        CHRM_LENGTH,
        min_identity=0.5,
        n_cores=4,
    )

    df = calculate_all_base_compositions(
        ccs_reads, assigned, ref_seqs, zmw_to_chrom, CHRM_LENGTH, n_cores=4
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert EXPECTED_COLUMNS == set(df.columns)
    # Agreement fraction should be between 0 and 1
    assert (df["agreement_fraction"] >= 0.0).all()
    assert (df["agreement_fraction"] <= 1.0).all()
    # Positions with coverage should have reasonable agreement
    covered = df[df["total_subreads"] > 0]
    assert len(covered) > 0
    assert covered["agreement_fraction"].mean() > 0.8
