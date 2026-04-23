# ccs_subread_align

Align PacBio subreads to their --by-strand CCS reads. Assigns each subread to forward or reverse strand via competitive edlib alignment, then computes per-position base composition across subreads.

**Currently only mitochondrial (chrM) CCS reads are supported.** The reference genome must be circularized by doubling (concatenating the sequence with itself) to handle reads spanning the circular origin.

## Installation

```bash
pip install ccs_subread_align
```

Or with Poetry:

```bash
poetry install
```

## Usage

```python
from ccs_subread_align import (
    load_reference,
    load_subreads,
    scan_zmw_to_chrom,
    stream_ccs_reads,
    process_subread_alignment,
    calculate_all_base_compositions,
)

chrM_length = 16569
ccs_bam = "ccs.bam"
zmw_list = [12345, 67890]

# Load circularized reference (chrM sequence concatenated with itself)
ref_seqs = load_reference("reference.fasta")

# Light pass over the CCS BAM: returns {zmw: reference_name}, needed before alignment
zmw_to_chrom = scan_zmw_to_chrom(ccs_bam, zmw_list)

# Subreads still load eagerly (smaller; grouped by ZMW for alignment)
subreads_by_zmw = load_subreads("subreads.bam", zmw_list)

# Assign subreads to strands via competitive alignment
assigned = process_subread_alignment(
    zmw_list, subreads_by_zmw, ref_seqs, zmw_to_chrom,
    chrM_length=chrM_length, min_identity=0.5,
)

# Stream CCS reads from the BAM into composition; CIGAR is parsed lazily in the worker
composition_df = calculate_all_base_compositions(
    stream_ccs_reads(ccs_bam, zmw_list, chrM_length),
    assigned, ref_seqs, zmw_to_chrom, chrM_length=chrM_length,
)
```

`stream_ccs_reads` returns a one-shot iterator; pass a fresh call per
consumer if you need to process the reads more than once.

For full-scale jobs the concatenated composition DataFrame can exceed available
memory (the 15-column frame is dominated by repeated string values that bloat to
hundreds of GB across ~10⁹ rows). Pass `output_path=` to stream each per-CCS
result to a zstd-compressed Parquet file instead and get the path back:

```python
out = calculate_all_base_compositions(
    stream_ccs_reads(ccs_bam, zmw_list, chrM_length),
    assigned, ref_seqs, zmw_to_chrom,
    chrM_length=chrM_length,
    output_path="composition.parquet",
)
# read it back lazily on the consumer side
import pandas as pd
df = pd.read_parquet(out)
```

The alignment stage has the same knob: `process_subread_alignment(..., output_path=...)` streams assigned-subread records to Parquet and returns the path instead of a `List[Dict]`. `calculate_all_base_compositions` accepts that Parquet path directly in place of the list, so a full-scale run keeps both the tens-of-GB assignments and the ~10⁹-row composition frame off the heap:

```python
process_subread_alignment(
    zmw_list, subreads_by_zmw, ref_seqs, zmw_to_chrom,
    chrM_length=chrM_length, min_identity=0.5,
    output_path="aligned.parquet",
)
calculate_all_base_compositions(
    stream_ccs_reads(ccs_bam, zmw_list, chrM_length),
    "aligned.parquet", ref_seqs, zmw_to_chrom,
    chrM_length=chrM_length,
    output_path="composition.parquet",
)
```

## License

`ccs_subread_align` was created by Natalie Gill. It is licensed under the terms of the AGPL-3.0 license.
