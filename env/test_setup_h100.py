"""CPU-only bootstrap regressions. No installs, downloads, CUDA or GPU queries."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import setup_h100 as setup


class Recorder:
    def __init__(self,root): self.root=Path(root);self.commands=[]
    def run(self,command,env=None,cwd=None):
        self.commands.append(list(map(str,command)))


class Bootstrap(unittest.TestCase):
    def test_default_environment_name_and_plan(self):
        a=setup.arguments([])
        self.assertEqual(a.prefix,setup.REPO.parent/'conda-envs/tempname-h100')
        self.assertEqual(a.megatron,setup.REPO.parent/'Megatron-LM')
        self.assertEqual(setup.plan(a)['python_version'],'3.10.20')
        self.assertFalse(setup.plan(a)['training_started'])

    def test_no_install_or_files_in_dry_run(self):
        with tempfile.TemporaryDirectory() as temp:
            state=Path(temp)/'missing'
            with patch.object(setup,'execute',side_effect=AssertionError('executed')),contextlib.redirect_stdout(io.StringIO()):
                setup.main(['--dry-run','--state-dir',str(state)])
            self.assertFalse(state.exists())
        result=subprocess.run(['bash',str(Path(setup.__file__).with_suffix('.sh')),'--dry-run'],
                              capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('tempname-h100',result.stdout)
        self.assertIn('PLAN ONLY',result.stdout)

    def test_source_repository_cannot_be_environment_prefix(self):
        with self.assertRaises(RuntimeError): setup.arguments(['--prefix',str(setup.REPO/'env/runtime')])
        with self.assertRaises(RuntimeError): setup.arguments(['--state-dir',str(setup.REPO/'cache')])

    def test_bad_installer_checksum_prevents_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);runner=Recorder(root)
            def curl(command,env=None,cwd=None):
                runner.commands.append(command)
                self.assertEqual(command[0],'curl')
                Path(command[command.index('--output')+1]).write_bytes(b'corrupt installer')
            runner.run=curl
            with self.assertRaisesRegex(RuntimeError,'SHA256 mismatch'):
                setup.ensure_miniforge(SimpleNamespace(state_dir=root),runner)
            self.assertEqual(len(runner.commands),1)
            self.assertFalse((root/'miniforge').exists())

    def test_verified_cached_installer_is_used_without_download(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);runner=Recorder(root);cache=root/'downloads';cache.mkdir()
            content=b'CPU installer fixture';(cache/setup.MINIFORGE_FILE).write_bytes(content)
            def install(command,env=None,cwd=None):
                runner.commands.append(command)
                self.assertEqual(command[0],'bash')
                (root/'miniforge/bin').mkdir(parents=True)
                (root/'miniforge/bin/conda').touch()
            runner.run=install
            with patch.object(setup,'MINIFORGE_SHA256',hashlib.sha256(content).hexdigest()):
                self.assertEqual(setup.ensure_miniforge(SimpleNamespace(state_dir=root),runner),root/'miniforge/bin/conda')
            self.assertEqual(len(runner.commands),1)

    def test_unmanaged_environments_are_not_modified(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=SimpleNamespace(state_dir=root,prefix=root/'python')
            (root/'miniforge').mkdir();a.prefix.mkdir()
            runner=Recorder(root)
            with self.assertRaisesRegex(RuntimeError,'not managed'): setup.ensure_miniforge(a,runner)
            with self.assertRaisesRegex(RuntimeError,'not managed'): setup.ensure_python(a,Path('/unused/conda'),runner)
            self.assertEqual(runner.commands,[])

    def test_python_creation_is_exact_and_repeat_is_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=SimpleNamespace(state_dir=root,prefix=root/'python');runner=Recorder(root)
            with patch.object(setup.recovery,'python_inventory',return_value={'python':[3,10,20]}):
                setup.ensure_python(a,Path('/fixture/conda'),runner)
                self.assertIn('python=3.10.20',runner.commands[0])
                self.assertIn('--override-channels',runner.commands[0])
                (a.prefix/'bin').mkdir(parents=True);(a.prefix/'bin/python').touch()
                setup.ensure_python(a,Path('/fixture/conda'),runner)
            self.assertEqual(len(runner.commands),1)

    def test_interrupted_owned_python_is_preserved_and_recreated(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=SimpleNamespace(state_dir=root,prefix=root/'python');runner=Recorder(root)
            a.prefix.mkdir();(a.prefix/'partial-file').write_text('partial')
            setup.write(root/'python-owner.json',dict(protocol=setup.PROTOCOL,prefix=str(a.prefix),version='3.10.20'))
            with patch.object(setup.recovery,'python_inventory',return_value={'python':[3,10,20]}):
                setup.ensure_python(a,Path('/fixture/conda'),runner)
            self.assertEqual((root/'incomplete-python/partial-file').read_text(),'partial')
            self.assertEqual(runner.commands[0][1],'create')

    def test_child_failure_is_not_ignored(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            runner=setup.LiveRunner(Path(temp))
            with self.assertRaisesRegex(RuntimeError,'exited 7'):
                runner.run([sys.executable,'-c','print("fixture failure");raise SystemExit(7)'])
            self.assertIn('fixture failure',(Path(temp)/'install.log').read_text())


class Sources(unittest.TestCase):
    def fixture(self,root):
        repo=root/'megatron';repo.mkdir()
        def git(*args):
            return subprocess.check_output(['git','-C',str(repo),'-c','user.name=CPU Fixture',
                '-c','user.email=fixture@example.invalid',*args],text=True,stderr=subprocess.DEVNULL).strip()
        git('init');(repo/'source.py').write_text('first\n');git('add','source.py');git('commit','-m','first')
        target=git('rev-parse','HEAD')
        (repo/'source.py').write_text('second\n');git('commit','-am','second')
        return repo,target,git

    def test_clean_existing_megatron_is_pinned_without_network(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            repo,target,git=self.fixture(Path(temp));runner=setup.LiveRunner(Path(temp))
            with patch.object(setup.recovery,'MEGATRON_COMMIT',target): setup.ensure_megatron(repo,runner)
            self.assertEqual(git('rev-parse','HEAD'),target)
            self.assertFalse(any('fetch' in r['argv'] for r in runner.commands))

    def test_dirty_megatron_is_preserved_before_any_command(self):
        with tempfile.TemporaryDirectory() as temp:
            repo,target,git=self.fixture(Path(temp));runner=Recorder(temp);before=git('rev-parse','HEAD')
            (repo/'source.py').write_text('user edit\n')
            with patch.object(setup.recovery,'MEGATRON_COMMIT',target),self.assertRaisesRegex(RuntimeError,'Tracked edits'):
                setup.ensure_megatron(repo,runner)
            self.assertEqual(git('rev-parse','HEAD'),before)
            self.assertEqual((repo/'source.py').read_text(),'user edit\n')
            self.assertEqual(runner.commands,[])

    def test_plain_megatron_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'user-file').write_text('keep')
            with self.assertRaisesRegex(RuntimeError,'not its own Git checkout'):
                setup.ensure_megatron(root,Recorder(root))
            self.assertEqual((root/'user-file').read_text(),'keep')


class Pipeline(unittest.TestCase):
    def fixture(self,root,fail=False,gpu=True):
        repo=root/'repo';repo.mkdir();state=root/'state';python=root/'conda-envs/tempname-h100/bin/python'
        a=SimpleNamespace(prefix=python.parent.parent,megatron=root/'Megatron-LM',state_dir=state,
                          skip_gpu_check=not gpu,dry_run=False)
        calls=[]
        class Runner(Recorder):
            def run(self,command,env=None,cwd=None):
                super().run(command,env,cwd);calls.append(self.commands[-1])
                if 'restore_container.py' in str(command[1]):
                    if fail: raise RuntimeError('fixture install failed')
                    setup.write(repo/'outputs/h100/container-recovery/last-report.json',
                        dict(status='completed',software_ready=True,gpu_checked=gpu,runtime_cuda='/cuda124'))
        with patch.object(setup,'REPO',repo),patch.object(setup,'check_host'),patch.object(setup,'LiveRunner',Runner), \
             patch.object(setup,'add_git_trust'),patch.object(setup,'ensure_megatron'),patch.object(setup,'ensure_submodules'), \
             patch.object(setup,'software_sources_check'),patch.object(setup,'ensure_miniforge',return_value=root/'conda'), \
             patch.object(setup,'ensure_python',return_value=python),patch.object(setup.recovery,'install_system'), \
             patch.object(setup.recovery,'package_installed',return_value=False),contextlib.redirect_stdout(io.StringIO()):
            if fail:
                with self.assertRaisesRegex(RuntimeError,'fixture install failed'): setup.execute(a)
            else: setup.execute(a)
        return calls,json.loads((repo/'outputs/h100/env-setup/last-report.json').read_text())

    def test_full_environment_flow_and_final_import_check(self):
        with tempfile.TemporaryDirectory() as temp:
            calls,result=self.fixture(Path(temp))
            self.assertEqual(result['status'],'completed')
            self.assertTrue(result['gpu_checked'])
            self.assertIn('tempname-h100',calls[0][0])
            self.assertIn('--repair-python',calls[0])
            self.assertIn('from pretrain_gpt import model_provider',calls[-1][-1])
            self.assertFalse(result['training_started'])

    def test_software_only_is_explicit_and_failure_does_not_mark_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            calls,result=self.fixture(Path(temp),gpu=False)
            self.assertIn('--skip-gpu-check',calls[0]);self.assertFalse(result['gpu_checked'])
        with tempfile.TemporaryDirectory() as temp:
            _,result=self.fixture(Path(temp),fail=True)
            self.assertEqual(result['status'],'failed');self.assertFalse(result['software_ready'])


if __name__=='__main__': unittest.main()
