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


def test_base_composition_collapses_duplicate_ref_pos():
    # Concatemer-style CCS: ccs_pos 4..7 revisit canonical ref_pos 0..3.
    ccs = _make_ccs_read("ACGTACGT")
    ccs["query_to_ref"] = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
    sr = _make_subread("ACGT", [0, 1, 2, 3])
    ref_seq = "ACGT" * 5000
    df = calculate_base_composition(ccs, [sr], ref_seq, CHRM_LENGTH)
    assert len(df) == 4
    assert df["ref_pos"].tolist() == [0, 1, 2, 3]
    assert df["ccs_pos"].tolist() == [0, 1, 2, 3]
    assert (df["total_subreads"] == 1).all()
    assert (df["agreement_fraction"] == 1.0).all()


def test_base_composition_no_collapse_keeps_per_ccs_pos():
    ccs = _make_ccs_read("ACGTACGT")
    ccs["query_to_ref"] = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
    sr = _make_subread("ACGT", [0, 1, 2, 3])
    ref_seq = "ACGT" * 5000
    df = calculate_base_composition(
        ccs, [sr], ref_seq, CHRM_LENGTH, collapse_duplicate_positions=False
    )
    assert len(df) == 8
    assert df["ccs_pos"].tolist() == list(range(8))
    assert df["ref_pos"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    # Every ccs_pos in ref_to_ccs[ref_pos] is incremented per subread base,
    # so both passes carry the same count.
    assert (df["total_subreads"] == 1).all()


def test_base_composition_collapse_preserves_insertions():
    # Two insertions and a duplicated ref_pos in the same read.
    ccs = _make_ccs_read("ACGTAC")
    ccs["query_to_ref"] = np.array([0, 1, -1, 3, 0, -1], dtype=np.int32)
    sr = _make_subread("ACT", [0, 1, 3])
    ref_seq = "ACGT" * 5000
    df = calculate_base_composition(ccs, [sr], ref_seq, CHRM_LENGTH)
    # Three canonical ref_pos kept (0, 1, 3) plus two -1 rows preserved.
    assert df["ref_pos"].tolist() == [0, 1, -1, 3, -1]
    assert df["ccs_pos"].tolist() == [0, 1, 2, 3, 5]


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
    # (zmw, strand, ref_pos) is unique on canonical positions under the
    # default collapse_duplicate_positions=True.
    canonical = df[df["ref_pos"] != -1]
    assert not canonical.duplicated(subset=["zmw", "strand", "ref_pos"]).any()


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
                    [r["position_map"] for r in batch],
                    type=schema.field("position_map").type,
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


# --- bucketed read path ---


def _write_bucketed_from_rows(
    root: Path,
    rows: list,
    n_buckets: int,
    schema: pa.Schema = None,
) -> None:
    """Split rows by zmw % n_buckets and write one parquet per bucket, plus
    the completion manifest. Mirrors the production writer's layout so the
    reader tests can use a lightweight synthetic input."""
    import json as _json

    from ccs_subread_align.alignment import _ASSIGNED_SUBREAD_SCHEMA

    if schema is None:
        schema = _ASSIGNED_SUBREAD_SCHEMA
    root.mkdir(parents=True, exist_ok=True)
    by_bucket: dict = {i: [] for i in range(n_buckets)}
    for r in rows:
        by_bucket[r["zmw"] % n_buckets].append(r)
    for i, bucket_rows in by_bucket.items():
        if not bucket_rows:
            continue
        _write_alignment_parquet(root / f"bucket_{i:02d}.parquet", bucket_rows, schema)
    (root / "manifest.json").write_text(
        _json.dumps({"n_buckets": n_buckets, "schema_version": "0.7", "has_margin": False})
    )


def test_calculate_all_base_compositions_bucketed_matches_single(tmp_path):
    """Composition output from a bucketed dir must match the single-file
    output row-for-row on the same synthetic inputs."""
    # Build a handful of CCS reads and subreads across multiple ZMWs so
    # bucketing actually splits the data.
    ccs_reads = [
        _make_ccs_read("ACGTACGT", zmw=z, strand="fwd") for z in range(8)
    ]
    subreads = [
        _make_subread("ACGTACGT", [0, 1, 2, 3, 4, 5, 6, 7], zmw=z, strand="fwd")
        for z in range(8)
    ]
    ref_seq = "ACGT" * 5000
    ref_seqs = {"chrM": ref_seq}
    zmw_to_chrom = {z: "chrM" for z in range(8)}

    # Single-file path: write a single parquet via the writer helper.
    from ccs_subread_align.alignment import _ASSIGNED_SUBREAD_SCHEMA
    single_path = tmp_path / "single.parquet"
    _write_alignment_parquet(
        single_path,
        [
            {
                "zmw": sr["zmw"],
                "strand": sr["strand"],
                "subread_name": sr["subread_name"],
                "aligned_sequence": sr["aligned_sequence"],
                "position_map": sr["position_map"].tolist(),
                "identity": sr["identity"],
            }
            for sr in subreads
        ],
        _ASSIGNED_SUBREAD_SCHEMA,
    )

    # Bucketed path: same rows, split into 4 buckets.
    bucket_dir = tmp_path / "bucketed"
    _write_bucketed_from_rows(
        bucket_dir,
        [
            {
                "zmw": sr["zmw"],
                "strand": sr["strand"],
                "subread_name": sr["subread_name"],
                "aligned_sequence": sr["aligned_sequence"],
                "position_map": sr["position_map"].tolist(),
                "identity": sr["identity"],
            }
            for sr in subreads
        ],
        n_buckets=4,
    )

    df_single = calculate_all_base_compositions(
        ccs_reads=list(ccs_reads),
        assigned_subreads=single_path,
        ref_seqs=ref_seqs,
        zmw_to_chrom=zmw_to_chrom,
        chrM_length=CHRM_LENGTH,
        n_cores=1,
    )

    # Bucketed reader requires a factory (ccs stream replayed per bucket).
    df_bucketed = calculate_all_base_compositions(
        ccs_reads=lambda: iter(ccs_reads),
        assigned_subreads=bucket_dir,
        ref_seqs=ref_seqs,
        zmw_to_chrom=zmw_to_chrom,
        chrM_length=CHRM_LENGTH,
        n_cores=1,
    )

    assert len(df_bucketed) == len(df_single) > 0
    sort_cols = ["zmw", "strand", "ccs_pos"]
    left = df_single.sort_values(sort_cols).reset_index(drop=True)
    right = df_bucketed.sort_values(sort_cols).reset_index(drop=True)
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


def test_calculate_all_base_compositions_bucketed_requires_callable(tmp_path):
    """Passing a plain iterable for ccs_reads when pointed at a bucketed
    directory must raise TypeError — the stream would be exhausted after
    the first bucket."""
    rows = [
        _make_row(1, "fwd", "ACGT", idx=0),
        _make_row(2, "fwd", "ACGT", idx=1),
    ]
    bucket_dir = tmp_path / "bucketed"
    _write_bucketed_from_rows(bucket_dir, rows, n_buckets=2)

    with pytest.raises(TypeError, match="callable"):
        calculate_all_base_compositions(
            ccs_reads=[],  # plain list, not a factory
            assigned_subreads=bucket_dir,
            ref_seqs={"chrM": "ACGT" * 5000},
            zmw_to_chrom={},
            chrM_length=CHRM_LENGTH,
            n_cores=1,
        )


def test_calculate_all_base_compositions_bucketed_missing_manifest_errors(tmp_path):
    """A directory of bucket files without manifest.json is treated as
    incomplete — reader must refuse to proceed, not silently return empty."""
    rows = [_make_row(1, "fwd", "ACGT")]
    bucket_dir = tmp_path / "partial"
    bucket_dir.mkdir()
    from ccs_subread_align.alignment import _ASSIGNED_SUBREAD_SCHEMA
    _write_alignment_parquet(
        bucket_dir / "bucket_00.parquet", rows, _ASSIGNED_SUBREAD_SCHEMA
    )
    # No manifest.json.

    with pytest.raises(ValueError, match="manifest.json"):
        calculate_all_base_compositions(
            ccs_reads=lambda: iter([]),
            assigned_subreads=bucket_dir,
            ref_seqs={"chrM": "ACGT" * 5000},
            zmw_to_chrom={},
            chrM_length=CHRM_LENGTH,
            n_cores=1,
        )


def test_calculate_all_base_compositions_bucketed_skips_empty_buckets(tmp_path):
    """Buckets with no matching ZMWs produce no parquet file; the reader
    must treat missing bucket files as empty, not as an error."""
    # All ZMWs are even, so bucket 1 of n_buckets=2 is empty.
    ccs_reads = [_make_ccs_read("ACGT", zmw=z, strand="fwd") for z in (0, 2, 4)]
    subreads = [
        _make_subread("ACGT", [0, 1, 2, 3], zmw=z, strand="fwd") for z in (0, 2, 4)
    ]
    rows = [
        {
            "zmw": sr["zmw"],
            "strand": sr["strand"],
            "subread_name": sr["subread_name"],
            "aligned_sequence": sr["aligned_sequence"],
            "position_map": sr["position_map"].tolist(),
            "identity": sr["identity"],
        }
        for sr in subreads
    ]
    bucket_dir = tmp_path / "bucketed"
    _write_bucketed_from_rows(bucket_dir, rows, n_buckets=2)
    # Sanity: bucket_01.parquet must not exist (all even zmws).
    assert not (bucket_dir / "bucket_01.parquet").exists()
    assert (bucket_dir / "bucket_00.parquet").exists()

    df = calculate_all_base_compositions(
        ccs_reads=lambda: iter(ccs_reads),
        assigned_subreads=bucket_dir,
        ref_seqs={"chrM": "ACGT" * 5000},
        zmw_to_chrom={0: "chrM", 2: "chrM", 4: "chrM"},
        chrM_length=CHRM_LENGTH,
        n_cores=1,
    )
    # 3 reads × 4 positions each.
    assert len(df) == 12
    assert (df["total_subreads"] == 1).all()


def test_group_subreads_from_parquet_handles_legacy_list_position_map(tmp_path):
    """Legacy parquet with position_map: list<int32> must upgrade to large_list.

    Pins the read-side cast for the list-column family. Without the
    upgrade, production sort_by() raises ArrowInvalid once aggregate
    element count crosses 2^31 (~8 GB of int32 child buffer, observed in
    the 118k-ZMW production run at ~8 B elements). A faithful overflow
    reproducer would require 8+ GB of memory; this small-data test is a
    behavioral pin on the cast logic itself.
    """
    rows = [
        _make_row(1, "fwd", "ACGT", idx=0, position_map=[0, 1, 2, 3]),
        _make_row(1, "fwd", "ACGA", idx=1, position_map=[0, 1, 2, 3]),
        _make_row(2, "rev", "TTTT", idx=2, position_map=[0, 1, 2, 3]),
    ]
    path = tmp_path / "legacy_list.parquet"
    _write_alignment_parquet(path, rows, _legacy_string_schema())

    on_disk = pq.read_table(path)
    assert on_disk.schema.field("position_map").type == pa.list_(pa.int32())
    assert on_disk.schema.field("aligned_sequence").type == pa.string()

    groups = _group_subreads_from_parquet(path)
    assert set(groups.keys()) == {(1, "fwd"), (2, "rev")}
    assert groups[(1, "fwd")].num_rows == 2
    assert groups[(2, "rev")].num_rows == 1
    for tbl in groups.values():
        assert tbl.schema.field("position_map").type == pa.large_list(pa.int32())
        assert tbl.schema.field("aligned_sequence").type == pa.large_string()


# --- zmw_strand high-cardinality regression (v0.7.1) ---


def _write_high_cardinality_composition(path: Path, n_ccs: int) -> None:
    """Stream a composition parquet from n_ccs CCS reads, each with a distinct zmw."""
    ccs_reads = [_make_ccs_read("ACGT", zmw=i, strand="fwd") for i in range(n_ccs)]
    subreads = [
        _make_subread("ACGT", [0, 1, 2, 3], zmw=i, strand="fwd") for i in range(n_ccs)
    ]
    calculate_all_base_compositions(
        ccs_reads=ccs_reads,
        assigned_subreads=subreads,
        ref_seqs={"chrM": "ACGT" * 5000},
        zmw_to_chrom={i: "chrM" for i in range(n_ccs)},
        chrM_length=CHRM_LENGTH,
        n_cores=1,
        output_path=path,
    )


def test_composition_zmw_strand_high_cardinality_round_trip(tmp_path):
    """Pre-v0.7.1 writer emitted dict<int8, string> for zmw_strand; readers overflowed past 127 distinct values per row group."""
    path = tmp_path / "high_cardinality.parquet"
    _write_high_cardinality_composition(path, n_ccs=200)

    pf = pq.ParquetFile(path)
    for _ in pf.iter_batches():
        pass
    pf.read_row_group(0)


def test_composition_parquet_zmw_strand_field_is_plain_string(tmp_path):
    """Structural invariant guarding against a Categorical regression at the per-CCS write site."""
    path = tmp_path / "any_scale.parquet"
    _write_high_cardinality_composition(path, n_ccs=4)

    field_type = pq.ParquetFile(path).schema_arrow.field("zmw_strand").type
    assert field_type in (pa.string(), pa.large_string()), (
        f"zmw_strand serialized as {field_type!r}; must be plain string"
    )


def test_composition_parquet_zmw_strand_distinct_values_recoverable(tmp_path):
    """All 200 distinct zmw_strand values survive the parquet round-trip."""
    n_ccs = 200
    path = tmp_path / "distinct.parquet"
    _write_high_cardinality_composition(path, n_ccs=n_ccs)

    table = pq.read_table(path)
    distinct = set(table.column("zmw_strand").to_pylist())
    assert distinct == {f"{i}_fwd" for i in range(n_ccs)}
