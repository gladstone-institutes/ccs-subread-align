"""Tests for ccs_subread_align.composition module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ccs_subread_align.composition import (
    _group_subreads_from_parquet,
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
    query_to_ref = np.array(
        [(ref_start + i) % CHRM_LENGTH for i in range(len(seq))], dtype=np.int32
    )
    return {
        "zmw": zmw,
        "strand": strand,
        "zmw_strand": f"{zmw}_{strand}",
        "query_sequence": seq,
        "query_length": len(seq),
        "quality_array": quality,
        "query_to_ref": query_to_ref,
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
    # Override: position 2 is an insertion (-1)
    ccs["query_to_ref"] = np.array([0, 1, -1, 3], dtype=np.int32)
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
    from ccs_subread_align.io import (
        load_reference,
        load_subreads,
        scan_zmw_to_chrom,
        stream_ccs_reads,
    )

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

    zmw_to_chrom = scan_zmw_to_chrom(str(CCS_BAM), zmw_list)
    subreads_by_zmw = load_subreads(str(SUBREADS_BAM), zmw_list)

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
        stream_ccs_reads(str(CCS_BAM), zmw_list, CHRM_LENGTH),
        assigned,
        ref_seqs,
        zmw_to_chrom,
        CHRM_LENGTH,
        n_cores=4,
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


@pytest.mark.skipif(
    not all(p.exists() for p in [CCS_BAM, SUBREADS_BAM, REF_FASTA]),
    reason="Test data not available",
)
def test_calculate_all_base_compositions_streaming(tmp_path):
    """Streaming output mode must produce identical rows to the in-memory path."""
    from ccs_subread_align.alignment import process_subread_alignment
    from ccs_subread_align.io import (
        load_reference,
        load_subreads,
        scan_zmw_to_chrom,
        stream_ccs_reads,
    )

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

    zmw_to_chrom = scan_zmw_to_chrom(str(CCS_BAM), zmw_list)
    subreads_by_zmw = load_subreads(str(SUBREADS_BAM), zmw_list)

    assigned = process_subread_alignment(
        zmw_list, subreads_by_zmw, ref_seqs, zmw_to_chrom, CHRM_LENGTH,
        min_identity=0.5, n_cores=2,
    )

    df_mem = calculate_all_base_compositions(
        stream_ccs_reads(str(CCS_BAM), zmw_list, CHRM_LENGTH),
        assigned, ref_seqs, zmw_to_chrom, CHRM_LENGTH, n_cores=2,
    )

    out_path = tmp_path / "composition.parquet"
    returned = calculate_all_base_compositions(
        stream_ccs_reads(str(CCS_BAM), zmw_list, CHRM_LENGTH),
        assigned, ref_seqs, zmw_to_chrom, CHRM_LENGTH, n_cores=2,
        output_path=out_path,
    )

    assert returned == out_path
    assert out_path.exists()

    df_stream = pd.read_parquet(out_path)
    assert len(df_stream) == len(df_mem)
    assert set(df_stream.columns) == set(df_mem.columns)

    # Sort both sides to handle non-deterministic pool ordering, then compare values.
    sort_cols = ["zmw", "strand", "ccs_pos"]
    left = df_mem.sort_values(sort_cols).reset_index(drop=True)
    right = df_stream.sort_values(sort_cols).reset_index(drop=True)
    for col in sort_cols + ["ref_pos", "A_count", "T_count", "C_count", "G_count",
                            "N_count", "total_subreads"]:
        assert (left[col].to_numpy() == right[col].to_numpy()).all(), col
    assert np.allclose(
        left["agreement_fraction"].to_numpy(),
        right["agreement_fraction"].to_numpy(),
        equal_nan=True,
    )
    for col in ["ccs_base", "reference_base"]:
        assert (
            left[col].astype(str).to_numpy() == right[col].astype(str).to_numpy()
        ).all(), col


def test_streaming_empty_result(tmp_path):
    """When no CCS reads survive filtering, no parquet file should be created."""
    out_path = tmp_path / "empty.parquet"
    returned = calculate_all_base_compositions(
        ccs_reads=[],
        assigned_subreads=[],
        ref_seqs={"chrM": "ACGT" * 5000},
        zmw_to_chrom={},
        chrM_length=CHRM_LENGTH,
        n_cores=1,
        output_path=out_path,
    )
    assert returned == out_path
    assert not out_path.exists()


def test_calculate_all_base_compositions_accepts_generator(tmp_path):
    """The CCS argument may be a one-shot generator; the function must not
    try to re-iterate it (no pre-count pass)."""
    ccs1 = _make_ccs_read("ACGTACGT", zmw=1, strand="fwd")
    ccs2 = _make_ccs_read("TTTTAAAA", zmw=2, strand="rev")
    sr1 = _make_subread("ACGTACGT", [0, 1, 2, 3, 4, 5, 6, 7], zmw=1, strand="fwd")
    sr2 = _make_subread("TTTTAAAA", [0, 1, 2, 3, 4, 5, 6, 7], zmw=2, strand="rev")
    ref_seq = "ACGT" * 5000

    def _one_shot():
        yield ccs1
        yield ccs2

    out_path = tmp_path / "gen.parquet"
    returned = calculate_all_base_compositions(
        ccs_reads=_one_shot(),
        assigned_subreads=[sr1, sr2],
        ref_seqs={"chrM": ref_seq},
        zmw_to_chrom={1: "chrM", 2: "chrM"},
        chrM_length=CHRM_LENGTH,
        n_cores=1,
        output_path=out_path,
    )
    assert returned == out_path
    df = pd.read_parquet(out_path)
    assert len(df) == 16  # 2 reads × 8 positions


def test_calculate_base_composition_parses_cigar_lazily():
    """When query_to_ref is absent, the worker parses cigartuples on the fly."""
    # Build a CCS dict without query_to_ref, with identity CIGAR (8 matches).
    ccs = {
        "zmw": 1,
        "strand": "fwd",
        "zmw_strand": "1_fwd",
        "query_sequence": "ACGTACGT",
        "query_length": 8,
        "quality_array": np.array([30] * 8, dtype=np.uint8),
        "cigartuples": [(7, 8)],  # op 7 == SEQ_MATCH (=)
        "reference_start": 0,
    }
    sr = _make_subread("ACGTACGT", [0, 1, 2, 3, 4, 5, 6, 7])
    ref_seq = "ACGT" * 5000
    df = calculate_base_composition(ccs, [sr], ref_seq, CHRM_LENGTH)
    assert len(df) == 8
    assert (df["total_subreads"] == 1).all()
    assert (df["agreement_fraction"] == 1.0).all()
    assert df["ref_pos"].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]


def test_streaming_small_synthetic(tmp_path):
    """Streaming path works with n_cores=1 and a handful of reads."""
    ccs1 = _make_ccs_read("ACGTACGT", zmw=1, strand="fwd")
    ccs2 = _make_ccs_read("TTTTAAAA", zmw=2, strand="rev")
    sr1 = _make_subread("ACGTACGT", [0, 1, 2, 3, 4, 5, 6, 7], zmw=1, strand="fwd")
    sr2 = _make_subread("TTTTAAAA", [0, 1, 2, 3, 4, 5, 6, 7], zmw=2, strand="rev")
    ref_seq = "ACGT" * 5000
    out_path = tmp_path / "small.parquet"
    returned = calculate_all_base_compositions(
        ccs_reads=[ccs1, ccs2],
        assigned_subreads=[sr1, sr2],
        ref_seqs={"chrM": ref_seq},
        zmw_to_chrom={1: "chrM", 2: "chrM"},
        chrM_length=CHRM_LENGTH,
        n_cores=1,
        output_path=out_path,
    )
    assert returned == out_path
    assert out_path.exists()
    df = pd.read_parquet(out_path)
    assert EXPECTED_COLUMNS == set(df.columns)
    assert len(df) == 16  # 2 reads × 8 positions


@pytest.mark.skipif(
    not all(p.exists() for p in [CCS_BAM, SUBREADS_BAM, REF_FASTA]),
    reason="Test data not available",
)
def test_calculate_all_base_compositions_from_parquet(tmp_path):
    """Piping streamed alignment parquet into composition must equal the in-memory path."""
    from ccs_subread_align.alignment import process_subread_alignment
    from ccs_subread_align.io import (
        load_reference,
        load_subreads,
        scan_zmw_to_chrom,
        stream_ccs_reads,
    )

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

    zmw_to_chrom = scan_zmw_to_chrom(str(CCS_BAM), zmw_list)
    subreads_by_zmw = load_subreads(str(SUBREADS_BAM), zmw_list)

    # Legacy path: in-memory List[Dict].
    assigned_list = process_subread_alignment(
        zmw_list, subreads_by_zmw, ref_seqs, zmw_to_chrom, CHRM_LENGTH,
        min_identity=0.5, n_cores=2,
    )
    df_from_list = calculate_all_base_compositions(
        stream_ccs_reads(str(CCS_BAM), zmw_list, CHRM_LENGTH),
        assigned_list, ref_seqs, zmw_to_chrom, CHRM_LENGTH, n_cores=2,
    )

    # Streaming path: parquet round-trip.
    aligned_parquet = tmp_path / "aligned.parquet"
    process_subread_alignment(
        zmw_list, subreads_by_zmw, ref_seqs, zmw_to_chrom, CHRM_LENGTH,
        min_identity=0.5, n_cores=2, output_path=aligned_parquet,
    )
    df_from_parquet = calculate_all_base_compositions(
        stream_ccs_reads(str(CCS_BAM), zmw_list, CHRM_LENGTH),
        aligned_parquet, ref_seqs, zmw_to_chrom, CHRM_LENGTH, n_cores=2,
    )

    assert len(df_from_parquet) == len(df_from_list)
    assert set(df_from_parquet.columns) == set(df_from_list.columns)

    sort_cols = ["zmw", "strand", "ccs_pos"]
    left = df_from_list.sort_values(sort_cols).reset_index(drop=True)
    right = df_from_parquet.sort_values(sort_cols).reset_index(drop=True)
    for col in sort_cols + [
        "ref_pos", "A_count", "T_count", "C_count", "G_count", "N_count",
        "total_subreads",
    ]:
        assert (left[col].to_numpy() == right[col].to_numpy()).all(), col
    assert np.allclose(
        left["agreement_fraction"].to_numpy(),
        right["agreement_fraction"].to_numpy(),
        equal_nan=True,
    )


# --- _group_subreads_from_parquet ---


def _legacy_string_schema() -> pa.Schema:
    # Mirrors the pre-v0.5.1 on-disk layout where aligned_sequence was
    # pa.string(). We still want to read these files correctly.
    return pa.schema(
        [
            pa.field("zmw", pa.int64()),
            pa.field("strand", pa.dictionary(pa.int8(), pa.string())),
            pa.field("subread_name", pa.string()),
            pa.field("aligned_sequence", pa.string()),
            pa.field("position_map", pa.list_(pa.int32())),
            pa.field("identity", pa.float32()),
        ]
    )


def _write_alignment_parquet(
    path: Path,
    rows: list,
    schema: pa.Schema,
    row_groups: int = 1,
) -> None:
    """Write a synthetic assigned-subread parquet split into `row_groups` chunks."""
    chunk_size = (len(rows) + row_groups - 1) // row_groups
    writer = pq.ParquetWriter(path, schema, compression="zstd")
    try:
        for start in range(0, len(rows), chunk_size):
            batch = rows[start : start + chunk_size]
            cols = {
                "zmw": pa.array([r["zmw"] for r in batch], type=pa.int64()),
                "strand": pa.array([r["strand"] for r in batch], type=pa.string()).dictionary_encode(),
                "subread_name": pa.array([r["subread_name"] for r in batch], type=pa.string()),
                "aligned_sequence": pa.array(
                    [r["aligned_sequence"] for r in batch],
                    type=schema.field("aligned_sequence").type,
                ),
                "position_map": pa.array(
                    [r["position_map"] for r in batch], type=pa.list_(pa.int32())
                ),
                "identity": pa.array([r["identity"] for r in batch], type=pa.float32()),
            }
            writer.write_table(pa.Table.from_pydict(cols, schema=schema))
    finally:
        writer.close()


def _make_row(
    zmw: int,
    strand: str,
    seq: str,
    idx: int = 0,
    position_map: list = None,
) -> dict:
    return {
        "zmw": zmw,
        "strand": strand,
        "subread_name": f"movie/{zmw}/{idx}_{idx + len(seq)}",
        "aligned_sequence": seq,
        # Default mirrors aligned_sequence length; pass `position_map=[...]`
        # to override when you only need aligned_sequence to be large (the
        # grouper under test never dereferences position_map).
        "position_map": list(range(len(seq))) if position_map is None else position_map,
        "identity": 0.95,
    }


def test_group_subreads_from_parquet_handles_legacy_string_schema(tmp_path):
    """Legacy parquet files (aligned_sequence: pa.string) must still load.

    Pins the read-side cast in _group_subreads_from_parquet: without it, a
    future refactor could drop the cast and silently reintroduce the 2GB
    take() overflow the moment a user has enough total sequence bytes.
    """
    rows = [
        _make_row(1, "fwd", "ACGT"),
        _make_row(1, "fwd", "ACGA"),
        _make_row(2, "rev", "TTTT"),
    ]
    path = tmp_path / "legacy.parquet"
    _write_alignment_parquet(path, rows, _legacy_string_schema())

    on_disk = pq.read_table(path)
    assert on_disk.schema.field("aligned_sequence").type == pa.string()

    groups = _group_subreads_from_parquet(path)
    assert set(groups.keys()) == {(1, "fwd"), (2, "rev")}
    assert groups[(1, "fwd")].num_rows == 2
    assert groups[(2, "rev")].num_rows == 1
    for tbl in groups.values():
        assert tbl.schema.field("aligned_sequence").type == pa.large_string()


def test_group_subreads_from_parquet_groups_correctly(tmp_path):
    """Sanity: grouper partitions rows by (zmw, strand) with no data loss."""
    from ccs_subread_align.alignment import _ASSIGNED_SUBREAD_SCHEMA

    rows = [
        _make_row(1, "fwd", "AAAA", idx=0),
        _make_row(1, "fwd", "AAAT", idx=1),
        _make_row(1, "rev", "CCCC", idx=2),
        _make_row(2, "fwd", "GGGG", idx=3),
        _make_row(2, "rev", "TTTT", idx=4),
        _make_row(3, "fwd", "ACGT", idx=5),
    ]
    path = tmp_path / "current.parquet"
    _write_alignment_parquet(path, rows, _ASSIGNED_SUBREAD_SCHEMA, row_groups=2)

    groups = _group_subreads_from_parquet(path)
    assert set(groups.keys()) == {
        (1, "fwd"), (1, "rev"), (2, "fwd"), (2, "rev"), (3, "fwd"),
    }
    assert groups[(1, "fwd")].num_rows == 2
    assert groups[(1, "rev")].num_rows == 1
    total = sum(tbl.num_rows for tbl in groups.values())
    assert total == len(rows)


def test_group_subreads_from_parquet_no_overflow_over_2gb(tmp_path):
    """Reproduces the user's offset-overflow scenario end-to-end.

    Writes a legacy-schema parquet with ~2.4 GB of aligned_sequence bytes
    split across two <2GB row groups. Without the read-side cast, the
    subsequent take() raises ArrowInvalid with the exact user-facing
    message. With the cast, grouping succeeds and returns large_string
    tables whose row counts match.

    Heavy: allocates several GB of strings and runs ~30-60 s.
    """
    big_seq = "A" * 60_000_000  # 60 MB; zstd compresses the repeats to a tiny parquet
    # Tiny position_map: the grouper never reads it, and a list(range(60M)) per
    # row is ~1.7 GB of Python ints each (×40 rows ≈ 70 GB) that would dwarf
    # the aligned_sequence bytes we actually want to stress.
    rows = [
        _make_row(1, "fwd", big_seq, idx=i, position_map=[0])
        for i in range(40)
    ]  # 40 * 60MB = 2.4GB of aligned_sequence bytes, the column we want to overflow

    path = tmp_path / "overflow.parquet"
    _write_alignment_parquet(path, rows, _legacy_string_schema(), row_groups=2)

    groups = _group_subreads_from_parquet(path)
    assert set(groups.keys()) == {(1, "fwd")}
    assert groups[(1, "fwd")].num_rows == 40
    assert groups[(1, "fwd")].schema.field("aligned_sequence").type == pa.large_string()
