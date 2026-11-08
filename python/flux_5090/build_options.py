"""CPU-only build parallelism shared by the controller and both CUDA libraries."""
import os
import shutil
import subprocess


def resolve_jobs(jobs=None):
    if jobs is None:
        nproc = shutil.which('nproc')
        if nproc:
            result = subprocess.run([nproc], check=True, capture_output=True, text=True, timeout=10)
            jobs = int(result.stdout.strip())
        elif hasattr(os, 'sched_getaffinity'):
            jobs = len(os.sched_getaffinity(0))
        else:
            jobs = os.cpu_count() or 1
    if type(jobs) is not int or jobs < 1:
        raise ValueError('Build jobs must be a positive integer')
    return jobs


def nvcc_parallel_flags(jobs=None):
    jobs = resolve_jobs(jobs)
    # --threads alone does not parallelize a single-architecture compilation.
    # --split-compile also parallelizes eligible device optimization passes.
    return ['--threads', str(jobs), '--split-compile', str(jobs)]
