# Changelog

<!--next-version-placeholder-->

## v0.6.0 (23/04/2026)

- **BREAKING**: `io.load_ccs_reads` is removed. Replaced by two functions with different scopes: `io.scan_zmw_to_chrom(bam, zmw_list)` (light pass that returns only the `{zmw: reference_name}` mapping, for use before alignment) and `io.stream_ccs_reads(bam, zmw_list, chrM_length)` (generator that yields CCS dicts one-at-a-time from the BAM, for use by composition). `query_to_ref` is no longer precomputed at load time; `composition.calculate_base_composition` parses `cigartuples` on demand in the worker (falling back to a precomputed `query_to_ref` when present so existing tests keep working). Removes the parent-side floor of ~21 GB at the 240k-CCS scale (~17 GB of `query_to_ref` arrays + ~4 GB of sequences) that used to sit resident from load through composition completion.
- `composition.calculate_all_base_compositions` accepts `Iterable[Dict]` for `ccs_reads` and iterates it exactly once. The pre-count pass for tqdm's total is removed; the progress bar now shows items-processed and rate. Callers that want a percent-complete can wrap their own tqdm.
- `alignment.process_subread_alignment` no longer builds a full `all_subreads` list before dispatching to the pool. The parent feeds a generator directly to `pool.imap`, and `ref_seqs` is shipped to workers once via a pool `initializer` (module-global `_WORKER_REF_SEQS`) instead of being embedded in every work item. Cuts a ~2 GB parent-side list at full scale and eliminates redundant `ref_seqs` re-pickling per imap chunk.
- `_pool.get_pool` grows `initializer=` and `initargs=` passthrough arguments.

### Migration

```diff
-ccs_reads = io.load_ccs_reads(ccs_bam, zmw_list, chrM_length)
-zmw_to_chrom = {c["zmw"]: c["reference_name"] for c in ccs_reads}
+zmw_to_chrom = io.scan_zmw_to_chrom(ccs_bam, zmw_list)
 alignment.process_subread_alignment(zmw_list, subreads, ref_seqs, zmw_to_chrom, chrM_length, min_identity, output_path=aligned_pq)
-composition.calculate_all_base_compositions(ccs_reads, aligned_pq, ref_seqs, zmw_to_chrom, chrM_length, output_path=comp_pq)
+composition.calculate_all_base_compositions(io.stream_ccs_reads(ccs_bam, zmw_list, chrM_length), aligned_pq, ref_seqs, zmw_to_chrom, chrM_length, output_path=comp_pq)
```

Memory on the in-repo test BAM (dozens of CCSs, all-in-memory dominates): peak RSS was 274 MB pre-v0.6.0 and 272 MB post-v0.6.0, so the delta barely registers at this scale. The design target is the 240k-CCS production scale, where the ~21 GB parent-side floor disappears.

## v0.5.1 (23/04/2026)

- Fix `pyarrow.lib.ArrowInvalid: offset overflow` in `_group_subreads_from_parquet` at full scale. The assigned-subread parquet stored `aligned_sequence` as `pa.string()` (int32 offsets); `pq.read_table` preserved per-row-group chunks under 2 GB, but the grouper's per-group `table.take()` fan-out combined them into one contiguous buffer and tipped past the 2 GB offset cap, *and* held peak memory at ~2× the parquet (one fresh buffer per group). Reader now casts string columns to `large_string`, sorts once by `(zmw, strand)`, then emits zero-copy `table.slice()` views per contiguous run. Peak RSS on a 2.4 GB stress reproducer dropped from 7.5 GB (take fan-out) to 4.5 GB (sort transient). The write schema for `aligned_sequence` is also widened to `pa.large_string()` so a 100k-row flush buffer of long reads cannot overflow during `pa.array` construction. `subread_name` stays as `pa.string()` (read names ~50 B, never overflows).

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