import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from flux_5090 import build_options as b


class BuildOptionsTests(unittest.TestCase):
    def test_default_uses_actual_nproc_command(self):
        with patch.object(b.shutil, 'which', return_value='/usr/bin/nproc'), \
             patch.object(b.subprocess, 'run', return_value=SimpleNamespace(stdout='24\n')) as run:
            self.assertEqual(b.resolve_jobs(), 24)
            self.assertEqual(run.call_args.args[0], ['/usr/bin/nproc'])

    def test_explicit_jobs_do_not_rediscover_worker_cpu_count(self):
        with patch.object(b.subprocess, 'run', side_effect=AssertionError('rediscovery forbidden')):
            self.assertEqual(b.nvcc_parallel_flags(12), ['--threads', '12', '--split-compile', '12'])

    def test_missing_nproc_respects_cpu_affinity(self):
        with patch.object(b.shutil, 'which', return_value=None), \
             patch.object(b.os, 'sched_getaffinity', return_value={1, 3, 5}):
            self.assertEqual(b.resolve_jobs(), 3)

    def test_bad_job_values_fail_before_compilation(self):
        for value in (0, -1, True, 1.5, '4'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                b.nvcc_parallel_flags(value)


if __name__ == '__main__':
    unittest.main()
