"""Resume the frozen numerical experiment with flushed checkpoint I/O only."""
import sys

from commitjl import durable_io


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--report':
        sys.argv.pop(1)
        from analysis import comprehensive_native_report
        comprehensive_native_report.atomic_json=durable_io.atomic_json
        comprehensive_native_report.main()
        return
    from experiments import comprehensive_native_comparison as full
    from experiments import enriched_jl_comparison, paper_compression, laptop_comparison
    # Imported aliases must all use the same replacement. Numerical routines,
    # random states, timings and the experiment's frozen source identity stay
    # unchanged. Evidence I/O remains outside measured compute stages.
    for module in [full,enriched_jl_comparison,paper_compression,laptop_comparison]:
        module.atomic_json=durable_io.atomic_json
        module.save_arrays=durable_io.save_arrays
    full.main()


if __name__=='__main__':main()
