"""CPU setup regression tests: no network, package installs, CUDA or GPU execution."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import setup_5090 as setup


def stack():
    return dict(python=[3,12,3], executable=sys.executable, prefix='/existing', base_prefix='/existing',
                torch='2.8.0a0+5228986c39.nv25.06', torch_file='/existing/torch/__init__.py',
                numpy='2.1.0', cuda='12.9', hip=None, distributed=True, nccl_available=True,
                arch_flags='sm_80 sm_90 sm_120 compute_120', profiler=True)


def fixture(root):
    repo, source = root/'repo', root/'source'
    contents = {'LICENSE.txt': b'fixture license\n',
                'include/cutlass/cutlass.h': b'fixture cutlass header\n',
                'tools/util/include/cutlass/util/device_memory.h': b'fixture utility\n'}
    for relative, content in contents.items():
        p=source/relative; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(content)
    lock=dict(source='https://github.com/NVIDIA/cutlass', revision='a'*40,
              scope='CPU fixture', files={name:hashlib.sha256(data).hexdigest() for name,data in contents.items()})
    p=repo/'third_party/cutlass.lock.json'; p.parent.mkdir(parents=True)
    p.write_text(json.dumps(lock,indent=2)+'\n')
    return repo,source,lock


class Recorder:
    def __init__(self, root): self.root,self.commands=Path(root),[]
    def run(self, command, env=None, cwd=None, timeout=900):
        self.commands.append(dict(argv=list(map(str,command)),env=env,cwd=cwd,timeout=timeout))


class Plans(unittest.TestCase):
    def test_uses_current_environment_without_installing(self):
        args=setup.arguments([]); value=setup.plan(args)
        self.assertEqual(args.base_python,Path(sys.executable).absolute())
        self.assertFalse(args.gpu_check)
        self.assertTrue(value['reuse_existing_environment'])
        for key in ('install_python','install_torch','install_cuda','install_packages','create_venv',
                    'requires_megatron','requires_apex','training_started'):
            self.assertFalse(value[key],key)
        self.assertEqual(value['cutlass_revision'],'df8a550d3917b0e97f416b2ed8c2d786f7f686a3')

    def test_preview_does_not_run_subprocess_or_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); out=root/'output'
            with patch.object(setup,'OUTPUT',out), \
                 patch.object(setup,'execute',side_effect=AssertionError('execution forbidden')), \
                 patch.object(setup,'capture',side_effect=AssertionError('query forbidden')), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                setup.main(['--dry-run'])
            self.assertIn('PLAN ONLY',output.getvalue())
            self.assertEqual(list(root.iterdir()),[])
            process=subprocess.run(['bash',str(Path(setup.__file__).with_suffix('.sh')),'--dry-run'],
                capture_output=True,text=True,env=dict(os.environ,RTX5090_BASE_PYTHON=sys.executable))
            self.assertEqual(process.returncode,0,process.stderr)
            self.assertIn('PLAN ONLY',process.stdout)

    def test_python_symlink_keeps_venv_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            python=Path(tmp)/'venv/bin/python'; python.parent.mkdir(parents=True)
            python.symlink_to(sys.executable)
            self.assertNotEqual(python.resolve(),python)
            for option in ('--base-python','--python'):
                self.assertEqual(setup.arguments([option,str(python)]).base_python,python)
            with patch.object(setup.sys,'executable',str(python)):
                self.assertEqual(setup.arguments([]).base_python,python)

    def test_dependencies_only_rejects_gpu_and_skip_flag_keeps_default(self):
        with self.assertRaisesRegex(RuntimeError,'dependencies-only'):
            setup.arguments(['--dependencies-only','--gpu-check'])
        self.assertFalse(setup.arguments(['--skip-gpu-check']).gpu_check)
        self.assertTrue(setup.arguments(['--gpu-check']).gpu_check)

    def test_software_environment_hides_gpu_without_changing_parent(self):
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='4,5',PYTHONPATH='/old/source',
                        PYTHONHOME='/old/home',NCCL_SOCKET_IFNAME='fixture0'):
            env=setup.software_environment('/existing/bin/python','/cuda','/cuda/bin/nvcc')
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')
            self.assertNotIn('PYTHONPATH',env); self.assertNotIn('PYTHONHOME',env)
            self.assertEqual(env['NCCL_SOCKET_IFNAME'],'fixture0')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'4,5')


class CutlassSources(unittest.TestCase):
    def test_valid_lock_and_unsafe_or_malformed_lock_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo,source,original=fixture(Path(tmp)); path=repo/'third_party/cutlass.lock.json'
            self.assertEqual(setup.load_lock(repo),original)
            changes=[dict(source='https://example.invalid/cutlass'),dict(revision='main'),dict(revision=1),
                     dict(files={}),dict(files={'LICENSE.txt':'a'*64}),
                     dict(files={**original['files'],'../outside':'a'*64}),
                     dict(files={**original['files'],'/absolute':'a'*64}),
                     dict(files={**original['files'],'include//extra.h':'a'*64}),
                     dict(files={**original['files'],'manifest.json':'a'*64}),
                     dict(files={**original['files'],'include/extra.h':5})]
            for changed in changes:
                with self.subTest(changed=changed):
                    path.write_text(json.dumps(dict(original,**changed)))
                    with self.assertRaises(RuntimeError): setup.load_lock(repo)

    def test_offline_install_reuses_existing_and_copies_only_locked_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); repo,source,lock=fixture(root)
            (source/'unused-upstream-file').write_text('not a build input')
            runner=Recorder(root); args=setup.arguments(['--cutlass-source',str(source)])
            with patch.object(setup,'capture',side_effect=AssertionError('network forbidden')):
                first=setup.ensure_cutlass(args,runner,repo)
                second=setup.ensure_cutlass(args,runner,repo)
            dest=repo/'third_party/cutlass'
            self.assertEqual((first['action'],second['action']),('installed','verified-existing'))
            self.assertFalse(first['network_used']); self.assertEqual(runner.commands,[])
            self.assertEqual((dest/'manifest.json').read_bytes(),(repo/'third_party/cutlass.lock.json').read_bytes())
            self.assertFalse((dest/'unused-upstream-file').exists())
            self.assertEqual(set(p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()),
                             set(lock['files'])|{'manifest.json'})

    def test_modified_existing_content_is_preserved_without_fetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); repo,source,lock=fixture(root)
            args=setup.arguments(['--cutlass-source',str(source)]); runner=Recorder(root)
            setup.ensure_cutlass(args,runner,repo)
            changed=repo/'third_party/cutlass/include/cutlass/cutlass.h'; changed.write_text('user edit')
            with self.assertRaisesRegex(RuntimeError,'missing/modified'):
                setup.ensure_cutlass(setup.arguments([]),runner,repo)
            self.assertEqual(changed.read_text(),'user edit'); self.assertEqual(runner.commands,[])

    def test_bad_source_never_publishes_partial_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); repo,source,lock=fixture(root)
            (source/'LICENSE.txt').write_text('wrong source revision')
            with self.assertRaisesRegex(RuntimeError,'missing/modified'):
                setup.ensure_cutlass(setup.arguments(['--cutlass-source',str(source)]),Recorder(root),repo)
            self.assertFalse((repo/'third_party/cutlass').exists())
            self.assertEqual(list((repo/'third_party').glob('.cutlass-setup-*')),[])
            self.assertEqual((source/'LICENSE.txt').read_text(),'wrong source revision')

    def test_existing_manifest_must_match_lock_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); repo,source,lock=fixture(root); runner=Recorder(root)
            args=setup.arguments(['--cutlass-source',str(source)])
            setup.ensure_cutlass(args,runner,repo)
            manifest=repo/'third_party/cutlass/manifest.json'
            manifest.write_text(json.dumps(lock))
            with self.assertRaisesRegex(RuntimeError,'manifest differs'):
                setup.ensure_cutlass(args,runner,repo)
            self.assertEqual(manifest.read_text(),json.dumps(lock))

    def test_symlink_inputs_and_destination_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); repo,source,lock=fixture(root)
            license=source/'LICENSE.txt'; target=root/'real-license'; license.rename(target); license.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError,'[Ss]ymlink'):
                setup.validate_cutlass(source,lock)
            dest=repo/'third_party/cutlass'; dest.symlink_to(source,target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError,'real directory'):
                setup.ensure_cutlass(setup.arguments([]),Recorder(root),repo)
            self.assertTrue(dest.is_symlink())

    def test_extra_header_is_allowed_in_upstream_but_rejected_in_installed_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); repo,source,lock=fixture(root)
            (source/'include/foreign.h').write_text('unexpected header')
            setup.validate_cutlass(source,lock)
            with self.assertRaises(RuntimeError): setup.validate_cutlass(source,lock,snapshot=True)

    def test_network_plan_fetches_only_fixed_revision_and_verifies_head(self):
        for correct in (True,False):
            with self.subTest(correct=correct), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); repo,source,lock=fixture(root)
                class FetchRecorder(Recorder):
                    def run(self,command,env=None,cwd=None,timeout=900):
                        super().run(command,env,cwd,timeout)
                        command=list(map(str,command))
                        if 'checkout' in command:
                            checkout=Path(command[command.index('-C')+1])
                            shutil.copytree(source,checkout,dirs_exist_ok=True)
                runner=FetchRecorder(root)
                with patch.object(setup.shutil,'which',return_value='/usr/bin/git'), \
                     patch.object(setup,'capture',return_value=(lock['revision'] if correct else 'b'*40)+'\n'):
                    if correct:
                        result=setup.ensure_cutlass(setup.arguments([]),runner,repo)
                        self.assertTrue(result['network_used'])
                    else:
                        with self.assertRaisesRegex(RuntimeError,'commit differs'):
                            setup.ensure_cutlass(setup.arguments([]),runner,repo)
                        self.assertFalse((repo/'third_party/cutlass').exists())
                fetch=next(c['argv'] for c in runner.commands if 'fetch' in c['argv'])
                self.assertEqual(fetch[-6:],['fetch','--depth','1','--no-tags',lock['source'],lock['revision']])
                self.assertFalse(any('pull' in c['argv'] for c in runner.commands))
                self.assertEqual(list((repo/'third_party').glob('.cutlass-setup-*')),[])


class ExistingSoftware(unittest.TestCase):
    def test_accepts_existing_supported_stack_beyond_one_image(self):
        setup.validate_stack(stack())
        setup.validate_stack(dict(stack(),python=[3,11,10],torch='2.9.0+cu130',cuda='13.0'))
        for changes in (dict(python=[3,9,20]),dict(cuda='12.4'),dict(cuda=None),dict(hip='7.0'),
                        dict(arch_flags='sm_90'),dict(nccl_available=False),dict(distributed=False),dict(profiler=False)):
            with self.subTest(changes=changes),self.assertRaises(RuntimeError):
                setup.validate_stack(dict(stack(),**changes))

    def test_inventory_is_cpu_only_and_requires_one_record(self):
        record=stack()
        with patch.object(setup,'capture',return_value='import log\nRTX5090_JSON:'+json.dumps(record)+'\n') as captured:
            self.assertEqual(setup.inventory('/existing/python','/cuda','/cuda/bin/nvcc'),record)
        argv,env=captured.call_args.args
        self.assertEqual(argv[0],'/existing/python'); self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')
        for data in ('','RTX5090_JSON:{}\nRTX5090_JSON:{}\n'):
            with patch.object(setup,'capture',return_value=data), self.assertRaises(RuntimeError):
                setup.inventory('/python','/cuda','/cuda/bin/nvcc')

    def test_explicit_invalid_nvcc_does_not_use_another_toolkit(self):
        with patch.dict(os.environ,{},clear=True), patch.object(setup.shutil,'which',return_value=None) as finder:
            with self.assertRaisesRegex(RuntimeError,'nvcc not found'):
                setup.find_toolkit(setup.arguments(['--nvcc','/missing/nvcc']))
            self.assertTrue(all(call.args==('/missing/nvcc',) for call in finder.call_args_list))

    def test_gpu_preflight_calls_new_runner_and_restores_requested_visibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); runner=Recorder(root); out=root/'gpu-preflight'; out.mkdir()
            setup.write(out/'batch-state.json',dict(status='preflight_completed',completed=True))
            setup.write(out/'report.json',dict(completed=True,completed_scope='preflight'))
            args=setup.arguments([]); toolkit=dict(root='/cuda',compiler='/cuda/bin/nvcc')
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'):
                setup.verify_gpu(args,toolkit,runner)
            command=runner.commands[0]
            self.assertIn(str(setup.REPO/'tools/5090/init.py'),command['argv'])
            self.assertIn('--preflight-only',command['argv'])
            self.assertIn('--foreground',command['argv'])
            self.assertNotIn('--megatron',command['argv'])
            self.assertEqual(command['env']['CUDA_VISIBLE_DEVICES'],'0,1,2,3,4,5,6,7')


class PipelineAndActivation(unittest.TestCase):
    def pipeline(self,root,flags=(),failure=None):
        repo,source,lock=fixture(root); out=repo/'outputs/5090/env-setup'
        args=setup.arguments([*flags,'--cutlass-source',str(source)])
        toolkit=dict(root='/existing/cuda',compiler='/existing/cuda/bin/nvcc')
        original_plan=setup.plan
        with patch.object(setup,'REPO',repo),patch.object(setup,'OUTPUT',out), \
             patch.object(setup,'plan',side_effect=lambda a:original_plan(a,repo)), \
             patch.object(setup,'find_toolkit',return_value=toolkit) as finder, \
             patch.object(setup,'inventory',return_value=stack()) as inventory, \
             patch.object(setup,'compile_probe',return_value=dict(gpu_used=False),
                          side_effect=RuntimeError('compile failed') if failure=='compile' else None) as compile_probe, \
             patch.object(setup,'verify_gpu',return_value=dict(passed=True),
                          side_effect=RuntimeError('GPU failed') if failure=='gpu' else None) as gpu, \
             contextlib.redirect_stdout(io.StringIO()):
            if failure:
                with self.assertRaisesRegex(RuntimeError,'failed'): setup.execute(args)
            else: setup.execute(args)
        return out,setup.read(out/'last-report.json'),(finder.call_count,inventory.call_count,compile_probe.call_count,gpu.call_count)

    def test_default_only_prepares_software_and_publishes_paths_after_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,record,calls=self.pipeline(Path(tmp))
            self.assertEqual(calls,(1,1,1,0)); self.assertEqual(record['status'],'completed')
            self.assertTrue(record['software_ready'] and record['dependencies_ready'] and record['gpu_checks_pending'])
            self.assertFalse(record['gpu_checked']); self.assertTrue((out/'env.sh').exists())

    def test_dependencies_only_never_calls_software_or_publishes_activation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,record,calls=self.pipeline(Path(tmp),['--dependencies-only'])
            self.assertEqual(calls,(0,0,0,0)); self.assertEqual(record['status'],'dependencies_ready')
            self.assertTrue(record['dependencies_ready']); self.assertFalse(record['software_ready'])
            self.assertFalse((out/'env.sh').exists())

    def test_failure_does_not_publish_activation(self):
        for stage in ('compile','gpu'):
            with self.subTest(stage=stage),tempfile.TemporaryDirectory() as tmp:
                flags=['--gpu-check'] if stage=='gpu' else []
                out,record,calls=self.pipeline(Path(tmp),flags,failure=stage)
                self.assertEqual(record['status'],'failed'); self.assertFalse(record['software_ready'])
                self.assertFalse((out/'env.sh').exists())

    def test_explicit_gpu_check_must_finish_before_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,record,calls=self.pipeline(Path(tmp),['--gpu-check'])
            self.assertEqual(calls,(1,1,1,1)); self.assertTrue(record['gpu_checked'])
            self.assertFalse(record['gpu_checks_pending']); self.assertTrue((out/'env.sh').exists())

    def test_environment_text_quotes_literal_shell_characters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); args=setup.arguments(['--base-python',str(root/'literal$(touch injected)/python')])
            toolkit=dict(root=str(root/"cuda with 'quotes'"),compiler=str(root/'nvcc; touch bad'))
            envfile=root/'env.sh'; envfile.write_text(setup.environment_text(args,toolkit))
            process=subprocess.run(['bash','-c','source "$1"; "$2" -c \'import json,os; print(json.dumps(dict(os.environ)))\'',
                'fixture',str(envfile),sys.executable],cwd=root,capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stderr)
            value=json.loads(process.stdout)
            self.assertEqual(value['RTX5090_PYTHON'],str(args.base_python))
            self.assertEqual(value['CUDA_HOME'],toolkit['root']); self.assertEqual(value['NVCC'],toolkit['compiler'])
            self.assertFalse((root/'injected').exists()); self.assertFalse((root/'bad').exists())

    def test_activation_preserves_pythonpath_nccl_without_venv_or_megatron(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); repo=root/'repo'; envdir=repo/'env'; envdir.mkdir(parents=True)
            script=envdir/'activate_5090.sh'; shutil.copyfile(Path(setup.__file__).with_name('activate_5090.sh'),script)
            cuda=root/'cuda space'; compiler=cuda/'bin/nvcc'; compiler.parent.mkdir(parents=True)
            compiler.write_text('#!/bin/sh\nexit 0\n'); compiler.chmod(0o755)
            args=setup.arguments([]); out=repo/'outputs/5090/env-setup'; out.mkdir(parents=True)
            (out/'env.sh').write_text(setup.environment_text(args,dict(root=str(cuda),compiler=str(compiler))))
            environment=dict(os.environ,PYTHONPATH='/keep/user/source',NCCL_P2P_DISABLE='0',NCCL_SOCKET_IFNAME='fixture0')
            process=subprocess.run(['bash','-c','source "$1" || exit; "$2" -c \'import json,os; print(json.dumps(dict(os.environ)))\'',
                'fixture',str(script),sys.executable],env=environment,cwd=root,capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stderr); value=json.loads(process.stdout)
            for key in ('PYTHONPATH','NCCL_P2P_DISABLE','NCCL_SOCKET_IFNAME'):
                self.assertEqual(value[key],environment[key])
            self.assertEqual(value['RTX5090_PYTHON'],str(args.base_python))
            self.assertEqual(value['CUDA_HOME'],str(cuda))
            self.assertNotIn('RTX5090_MEGATRON',value)


if __name__=='__main__': unittest.main()
