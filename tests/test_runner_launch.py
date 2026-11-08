"""CPU-only routing, detachment and failure-status checks. No worker/GPU launch."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parents[1] / 'tools/5090'
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('init_launch_under_test', HERE / 'init.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class RoutingTests(unittest.TestCase):
    def test_default_and_explicit_place_are_siblings_of_repository(self):
        with mock.patch.object(c, 'REPO', Path('/workspace/TempName-5090')):
            default = c.arguments(['-j', '1'])
            other = c.arguments(['-j', '1', '--place', '60'])
        self.assertEqual(default.place, 61)
        self.assertFalse(default.foreground)
        self.assertEqual(default.out.parent, Path('/workspace/Data-5090/SL3061'))
        self.assertEqual(other.out.parent, Path('/workspace/Data-5090/SL3060'))
        self.assertEqual(c.plan_for(other)['config']['place'], 60)

    def test_out_overrides_place_and_preview_writes_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / 'custom result'
            with mock.patch.object(c, 'start_background', side_effect=AssertionError('no launch')), \
                 mock.patch.object(c, 'execute', side_effect=AssertionError('no execute')), \
                 contextlib.redirect_stdout(io.StringIO()) as text:
                c.main(['--place', '60', '--out', str(target), '-j', '2'])
            preview = json.loads(text.getvalue())
            self.assertEqual(preview['output'], str(target))
            self.assertEqual(preview['config']['place'], 60)
            self.assertEqual(preview['execution_mode'], 'background')
            self.assertFalse(preview['writes_files'])
            self.assertFalse(target.exists())

    def test_foreground_and_report_do_not_detach(self):
        with mock.patch.object(c, 'execute') as execute, mock.patch.object(c, 'start_background') as start:
            c.main(['--execute', '-j', '1'])
            start.assert_called_once()
            execute.assert_not_called()
            start.reset_mock()
            c.main(['--execute', '--foreground', '-j', '1'])
            execute.assert_called_once()
            start.assert_not_called()
            with mock.patch.object(c.reporting, 'replay', return_value={}), \
                 mock.patch.object(c.reporting, 'markdown', return_value='read-only report'), \
                 contextlib.redirect_stdout(io.StringIO()):
                c.main(['--report', '--out', '/tmp/finished-run', '-j', '1'])
            self.assertEqual(execute.call_count, 1)
            start.assert_not_called()


class BackgroundTests(unittest.TestCase):
    def test_detach_preserves_options_and_freezes_output_and_jobs(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / 'literal $(touch injected)'
            argv = ['--execute', '--place', '60', '--trials', '2', '--blocks', '4',
                    '--out', str(target), '-j', '3', '--profile', '--operator-only']
            args = c.arguments(argv)
            with mock.patch.object(c.subprocess, 'Popen', return_value=mock.Mock(pid=12345)) as popen, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(c.start_background(args, argv), 12345)
            command = popen.call_args.args[0]
            child = c.arguments(command[3:])
            self.assertEqual((child.place, child.trials, child.blocks, child.jobs), (60, 2, 4, 3))
            self.assertEqual(child.out, target)
            self.assertTrue(child.execute and child.foreground and child.profile and child.operator_only)
            options = popen.call_args.kwargs
            self.assertTrue(options['start_new_session'] and options['close_fds'])
            self.assertEqual(options['stdin'], subprocess.DEVNULL)
            self.assertEqual(options['stderr'], subprocess.STDOUT)
            self.assertEqual(Path(options['stdout'].name), target / 'controller.log')
            self.assertNotIn('shell', options)
            receipt = c.p.read(target / 'launcher.json')
            self.assertEqual(receipt['token'], child.background_token)
            self.assertEqual(receipt['controller_pid'], 12345)
            self.assertEqual((target / 'controller.pid').read_text(), '12345\n')
            self.assertFalse((Path(root) / 'injected').exists())

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root)
            marker = target / 'data'; marker.write_text('preserve')
            args = c.arguments(['--execute', '--out', root, '-j', '1'])
            with mock.patch.object(c.subprocess, 'Popen') as popen, self.assertRaisesRegex(ValueError, 'already exists'):
                c.start_background(args, [])
            popen.assert_not_called()
            self.assertEqual(marker.read_text(), 'preserve')

    def test_spawn_failure_has_persistent_failure_record(self):
        with tempfile.TemporaryDirectory() as root:
            args = c.arguments(['--execute', '--out', str(Path(root) / 'run'), '-j', '1'])
            with mock.patch.object(c.subprocess, 'Popen', side_effect=OSError('fixture spawn failed')), \
                 self.assertRaisesRegex(OSError, 'fixture spawn failed'):
                c.start_background(args, [])
            state = c.p.read(args.out / 'batch-state.json')
            self.assertEqual(state['status'], 'launch_failed')
            self.assertFalse(state['completed'])

    def test_early_source_failure_is_recorded_and_cannot_be_replayed(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / 'run'; target.mkdir()
            args = c.arguments(['--execute', '--foreground', '--out', str(target), '-j', '1',
                                '--background-token', 'fixture-token'])
            c.p.write(target / 'launcher.json', dict(token='fixture-token', output=str(target)))
            # The parent may be atomically updating its launch receipt at this instant.
            (target / 'launcher.json.tmp').write_text('temporary')
            with mock.patch.object(c.reporting, 'validate_reference_sources', side_effect=RuntimeError('fixture source failed')), \
                 self.assertRaisesRegex(RuntimeError, 'fixture source failed'):
                c.execute(args)
            state = c.p.read(target / 'batch-state.json')
            self.assertEqual(state['status'], 'failed')
            self.assertIn('fixture source failed', state['error'])
            with self.assertRaisesRegex(ValueError, 'already contains'):
                c.execute(args)
            self.assertEqual(c.p.read(target / 'batch-state.json'), state)

    def test_real_detached_cpu_controller_records_worker_failure(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            wrapper = root / 'cpu_controller.py'
            wrapper.write_text(
                'import importlib.util,os,sys\n'
                f'sys.path.insert(0,{str(HERE)!r})\n'
                f'spec=importlib.util.spec_from_file_location("controller_fixture",{str(HERE / "init.py")!r})\n'
                'c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)\n'
                'c.__file__=__file__\n'
                'c.reporting.validate_reference_sources=lambda root: {}\n'
                'c.source_files=lambda: []\n'
                'def fake_launch(*args,**kwargs):\n'
                '    print("CPU_ONLY_SESSION",os.getsid(0),os.getpid(),flush=True)\n'
                '    raise RuntimeError("intentional CPU worker failure; no GPU launched")\n'
                'c.launch=fake_launch\n'
                'c.main()\n')
            target = root / 'run'
            result = subprocess.run([sys.executable, str(wrapper), '--execute', '--place', '60',
                '--blocks', '1', '-j', '1', '--out', str(target)], capture_output=True, text=True,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'), timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Background controller started', result.stdout)
            deadline = time.monotonic() + 15
            state = {}
            while time.monotonic() < deadline:
                path = target / 'batch-state.json'
                if path.exists():
                    state = c.p.read(path)
                    if state['status'] == 'failed':
                        break
                time.sleep(.05)
            self.assertEqual(state.get('status'), 'failed', (state, (target / 'controller.log').read_text()))
            self.assertIn('intentional CPU worker failure', state['error'])
            pid = int((target / 'controller.pid').read_text())
            self.assertEqual(state['controller_pid'], pid)
            self.assertEqual(state['place'], 60)
            self.assertIn(f'CPU_ONLY_SESSION {pid} {pid}', (target / 'controller.log').read_text())


if __name__ == '__main__':
    unittest.main()
