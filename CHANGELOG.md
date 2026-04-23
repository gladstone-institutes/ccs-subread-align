# Changelog

<!--next-version-placeholder-->

## v0.5.1 (23/04/2026)

- Fix `pyarrow.lib.ArrowInvalid: offset overflow` in `_group_subreads_from_parquet` at full scale. The assigned-subread parquet stored `aligned_sequence` as `pa.string()` (int32 offsets); `pq.read_table` preserved per-row-group chunks under 2 GB, but `table.take()` physically combined them and tipped past the 2 GB offset cap. Reader now casts string columns to `large_string` before `take()`, and the write schema for `aligned_sequence` is widened to `pa.large_string()` so a 100k-row flush buffer of long reads also cannot overflow during `pa.array` construction. `subread_name` stays as `pa.string()` (read names ~50 B, never overflows).

## v0.5.0 (22/04/2026)

- `process_subread_alignment` gains an optional `output_path=` kwarg. When provided, assigned-subread records are streamed to a zstd-compressed Parquet file via `pyarrow.parquet.ParquetWriter` (100k-row flush buffer) and the function returns the path instead of a `List[Dict]`. Schema: `zmw:int64`, `strand:dict<string>`, `subread_name:string`, `aligned_sequence:string`, `position_map:list<int32>`, `identity:float32`, plus `edit_distance_margin:int32` when `report_margin=True`. Avoids materializing the tens-of-GB `results` list that drove the parent process over 30 GB before pool fork.
- `calculate_all_base_compositions` now accepts a Parquet path for `assigned_subreads` (in addition to the legacy `List[Dict]`). The parquet loads as a pyarrow Table and is sliced per `(zmw, strand)` group; groups are converted to `List[Dict]` just-in-time inside the work-items generator so only one group is materialized at a time.
- `calculate_all_base_compositions` no longer builds a full `work_items: List[Tuple]` in the parent. The filter loop is a generator (`_iter_work_items`) fed directly to `pool.imap`, with a cheap pre-count supplying the tqdm total.
- Both pools now use `multiprocessing.get_context("forkserver")` on POSIX (falling back to `spawn` on Windows) with `maxtasksperchild=200`. Workers fork from a tiny bootstrap process rather than the bloated parent, eliminating the ENOMEM failure at pool-creation time on Linux. Introduced `ccs_subread_align._pool.get_pool` as the shared factory.

## v0.4.0 (21/04/2026)

- `calculate_all_base_compositions` gains an optional `output_path=` kwarg. When provided, per-CCS results are streamed to a zstd-compressed Parquet file via `pyarrow.parquet.ParquetWriter` (1M-row flush buffer) and the function returns the path instead of a `DataFrame`. Avoids the in-memory `pd.concat` that OOMs full-scale jobs (~1.69B rows → ~450 GB of which ~340 GB is object-dtype string columns). Omit the kwarg to keep the existing in-memory return.
- Output schema tightened in both return modes: `strand`/`ccs_base`/`reference_base` are `Categorical`, `A/T/C/G/N_count` and `total_subreads` are `uint16`, `ccs_pos` is `int32`, `agreement_fraction` is `float32`. All 15 columns preserved; pandas/arrow auto-promote on read. On the test BAMs this cuts the in-memory DataFrame's `memory_usage(deep=True)` from ~166 MB to ~20 MB (8.2×); the streaming Parquet file is ~3.7 MB (~45× smaller than the baseline DataFrame).
- `calculate_base_composition` hot path rewritten: `ref_pos → ccs_positions` is precomputed once per CCS (was an O(ccs_len) `np.where` inside a nested loop), `reference_base` lookup is vectorized via numpy fancy-indexing, `agreement_counts` uses an ASCII LUT + fancy index. ~5× faster per CCS on the bench workload.
- New `scripts/bench_composition.py` for reproducible pre/post perf measurement (per-CCS micro-benchmark + end-to-end, emits CSV).

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