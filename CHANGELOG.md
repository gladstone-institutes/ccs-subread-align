# Changelog

<!--next-version-placeholder-->

## v0.3.0 (19/04/2026)

- **BREAKING**: `query_to_ref_map` (dict) removed from CCS read dicts, replaced with `query_to_ref` (`np.ndarray[int32]`, length = `query_length`, `-1` for insertions/soft-clips/hard-clips). Downstream consumers must switch from dict iteration to ndarray indexing.
- `parse_cigar_to_reference_map` now returns `np.ndarray[int32]` and takes `query_length` as a required positional argument.
- `parse_cigar_to_reference_map` and `parse_edlib_cigar_to_positions` are rewritten as direct numpy walks over the CIGAR (no aligntools intermediary). ~70-140× faster per call in the benchmark; extrapolated 240k × 17kb load drops from ~48 min to ~20 s. Pool-based parallelization is no longer needed, so the `n_cores` kwarg on `load_ccs_reads` is removed.
- `aligntools` moved from runtime to dev dependencies; retained as a ground-truth in the equivalence tests.
- Phase + tqdm progress logging added to `load_ccs_reads` and `load_subreads`, including peak RSS at each phase boundary.

## v0.2.0 (14/04/2026)

- Add Parquet read/write support via pyarrow (`read_parquet`, `write_parquet`)

## v0.1.0 (29/01/2026)

- First release of `ccs_subread_align`!