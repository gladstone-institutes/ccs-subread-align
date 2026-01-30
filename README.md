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
    load_ccs_reads,
    load_subreads,
    process_subread_alignment,
    calculate_all_base_compositions,
)

chrM_length = 16569

# Load circularized reference (chrM sequence concatenated with itself)
ref_seqs = load_reference("reference.fasta")

# Load CCS reads and subreads for a set of ZMWs
zmw_list = [12345, 67890]
ccs_reads = load_ccs_reads("ccs.bam", zmw_list, chrM_length)
subreads_by_zmw = load_subreads("subreads.bam", zmw_list)

# Map each ZMW to its chromosome
zmw_to_chrom = {ccs["zmw"]: ccs["reference_name"] for ccs in ccs_reads}

# Assign subreads to strands via competitive alignment
assigned = process_subread_alignment(
    zmw_list, subreads_by_zmw, ref_seqs, zmw_to_chrom,
    chrM_length=chrM_length, min_identity=0.5,
)

# Compute per-position base composition
composition_df = calculate_all_base_compositions(
    ccs_reads, assigned, ref_seqs, zmw_to_chrom, chrM_length=chrM_length,
)
```

## License

`ccs_subread_align` was created by Natalie Gill. It is licensed under the terms of the AGPL-3.0 license.
