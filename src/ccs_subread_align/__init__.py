from importlib.metadata import version

__version__ = version("ccs_subread_align")

from ccs_subread_align.alignment import (
    assign_subreads_to_strand,
    extract_zmw_from_name,
    parse_cigar_to_reference_map,
    parse_edlib_cigar_to_positions,
    process_subread_alignment,
    reverse_complement,
)
from ccs_subread_align.composition import (
    calculate_all_base_compositions,
    calculate_base_composition,
)
from ccs_subread_align.io import load_ccs_reads, load_reference, load_subreads

__all__ = [
    "reverse_complement",
    "extract_zmw_from_name",
    "parse_cigar_to_reference_map",
    "parse_edlib_cigar_to_positions",
    "assign_subreads_to_strand",
    "process_subread_alignment",
    "calculate_base_composition",
    "calculate_all_base_compositions",
    "load_ccs_reads",
    "load_reference",
    "load_subreads",
]
