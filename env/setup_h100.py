#!/usr/bin/env python3
"""Fresh-container bootstrap around the existing validated H100 recovery path."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import subprocess
import sys
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'tools/h100/phase6'))
import restore_container as recovery

MINIFORGE_VERSION = '25.3.1-0'
MINIFORGE_FILE = f'Miniforge3-{MINIFORGE_VERSION}-Linux-x86_64.sh'
MINIFORGE_URL = f'https://github.com/conda-forge/miniforge/releases/download/{MINIFORGE_VERSION}/{MINIFORGE_FILE}'
# Published digest of the official release asset, not a moving latest URL.
MINIFORGE_SHA256 = '376b160ed8130820db0ab0f3826ac1fc85923647f75c1b8231166e3d559ab768'
PYTHON_VERSION = '3.10.20'
MEGATRON_URL = 'https://github.com/NVIDIA/Megatron-LM.git'
PROTOCOL = 'h100-fresh-container-environment-v1'
require = recovery.require


class LiveRunner(recovery.Runner):
    """Log commands and stream their progress; the recovery child also logs its builds."""
    def run(self,command,env=None,cwd=None):
        command=list(map(str,command))
        self.commands.append(dict(argv=command,cwd=str(cwd) if cwd else None))
        print('+ '+shlex.join(command),flush=True)
        with (self.root/'install.log').open('a') as log:
            log.write('\nCOMMAND '+shlex.join(command)+'\n');log.flush()
            child=subprocess.Popen(command,cwd=cwd,env=env,stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,text=True,bufsize=1)
            try:
                for line in child.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                code=child.wait()
            except BaseException:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill();child.wait()
                raise
            finally:
                child.stdout.close()
        require(code==0,f'Command exited {code}; see {self.root / "install.log"}')


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def arguments(argv=None):
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--prefix',type=Path,default=REPO.parent/'conda-envs/tempname-h100')
    p.add_argument('--megatron',type=Path,default=REPO.parent/'Megatron-LM')
    p.add_argument('--state-dir',type=Path,default=REPO.parent/'.h100-env')
    p.add_argument('--skip-gpu-check',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    for key in ('prefix','megatron','state_dir'):
        setattr(a,key,getattr(a,key).expanduser().resolve())
    require(a.prefix != a.state_dir and not a.prefix.is_relative_to(REPO) and
            not a.state_dir.is_relative_to(REPO),'Keep downloaded environments/cache outside the source repository')
    require(a.prefix != a.megatron,'Python prefix and Megatron checkout must differ')
    return a


def plan(a):
    return dict(protocol=PROTOCOL,repo=str(REPO),python=str(a.prefix/'bin/python'),
        miniforge=str(a.state_dir/'miniforge'),megatron=str(a.megatron),python_version=PYTHON_VERSION,
        torch='2.6.0+cu124',apex_commit=recovery.APEX_COMMIT,megatron_commit=recovery.MEGATRON_COMMIT,
        cuda_runtime='12.4',cuda_compiler='12.8',gpu_check=not a.skip_gpu_check,
        generated_environment=str(REPO/'outputs/h100/container-recovery/env.sh'),
        training_started=False)


def add_git_trust(paths):
    # Inherited by nested Git operations and the existing native source checks.
    # Never write safe.directory=* or modify the user's global Git settings.
    count=int(os.environ.get('GIT_CONFIG_COUNT','0'))
    for path in paths:
        os.environ[f'GIT_CONFIG_KEY_{count}']='safe.directory'
        os.environ[f'GIT_CONFIG_VALUE_{count}']=str(Path(path).resolve())
        count+=1
    os.environ['GIT_CONFIG_COUNT']=str(count)


def check_clean_checkout(path):
    root=recovery.capture(recovery.git_command(path,'rev-parse','--show-toplevel'))
    require(root.returncode==0 and Path(root.stdout.strip()).resolve()==Path(path).resolve(),
            f'{path} exists but is not its own Git checkout; it will not be overwritten. Select --megatron with a new path.')
    status=recovery.capture(recovery.git_command(path,'status','--porcelain','--untracked-files=no'),timeout=300)
    require(status.returncode==0 and not status.stdout.strip(),
            f'Tracked edits in {path}; preserved unchanged. Commit/stash them or select another --megatron path.')


def ensure_megatron(path,runner):
    path=Path(path)
    if not path.exists():
        recovery.clone_pinned(MEGATRON_URL,path,recovery.MEGATRON_COMMIT,runner,dict(os.environ))
        return
    check_clean_checkout(path)
    current=recovery.capture(recovery.git_command(path,'rev-parse','HEAD')).stdout.strip()
    if current!=recovery.MEGATRON_COMMIT:
        obj=recovery.capture(recovery.git_command(path,'cat-file','-e',recovery.MEGATRON_COMMIT+'^{commit}'))
        if obj.returncode:
            runner.run(recovery.git_command(path,'fetch','--depth','1',MEGATRON_URL,recovery.MEGATRON_COMMIT))
        # No reset --hard, clean, or deletion. Git also protects conflicting untracked files.
        runner.run(recovery.git_command(path,'checkout','--detach',recovery.MEGATRON_COMMIT))
    recovery.check_checkout(path,recovery.MEGATRON_COMMIT)


def ensure_submodules(runner):
    for name in ('cutlass','nccl'):
        relative='3rdparty/'+name;path=REPO/relative
        entry=recovery.capture(recovery.git_command(REPO,'ls-tree','HEAD',relative))
        require(entry.returncode==0 and entry.stdout.startswith('160000 '),'Missing committed submodule: '+relative)
        expected=entry.stdout.split()[2]
        if (path/'.git').exists(): check_clean_checkout(path)
        runner.run(recovery.git_command(REPO,'submodule','update','--init','--recursive','--depth','1','--',relative))
        recovery.check_checkout(path,expected)


def ensure_miniforge(a,runner):
    target=a.state_dir/'miniforge';receipt=a.state_dir/'miniforge-owner.json'
    expected=dict(protocol=PROTOCOL,prefix=str(target),installer_sha256=MINIFORGE_SHA256)
    if target.exists():
        require(receipt.is_file() and json.loads(receipt.read_text())==expected,
                'Existing Miniforge directory is not managed by this script; select another --state-dir')
        if (target/'bin/conda').is_file(): return target/'bin/conda'
        # Preserve an interrupted installation rather than deleting its contents.
        target.rename(runner.root/'incomplete-miniforge')
    cache=a.state_dir/'downloads';cache.mkdir(parents=True,exist_ok=True)
    installer=cache/MINIFORGE_FILE
    if not installer.is_file() or hashlib.sha256(installer.read_bytes()).hexdigest()!=MINIFORGE_SHA256:
        partial=installer.with_suffix('.sh.part')
        runner.run(['curl','--fail','--location','--retry','5','--connect-timeout','30',
                    '--proto','=https','--proto-redir','=https','--output',partial,MINIFORGE_URL])
        require(hashlib.sha256(partial.read_bytes()).hexdigest()==MINIFORGE_SHA256,
                'Miniforge SHA256 mismatch; installer was not executed')
        partial.replace(installer)
    write(receipt,expected)
    runner.run(['bash',installer,'-b','-p',target],dict(os.environ,PYTHONNOUSERSITE='1'))
    require((target/'bin/conda').is_file(),'Miniforge installation incomplete')
    return target/'bin/conda'


def ensure_python(a,conda,runner):
    receipt=a.state_dir/'python-owner.json'
    expected=dict(protocol=PROTOCOL,prefix=str(a.prefix),version=PYTHON_VERSION)
    owned=receipt.is_file() and json.loads(receipt.read_text())==expected
    python=a.prefix/'bin/python'
    if a.prefix.exists():
        require(owned,'Existing Python prefix is not managed by this script; use --prefix with a fresh directory')
        if python.is_file():
            record=recovery.python_inventory(python)
            require(record['python']==[3,10,20],'Existing managed Python version changed; choose a fresh --prefix')
            return python
        a.prefix.rename(runner.root/'incomplete-python')
    write(receipt,expected)
    runner.run([conda,'create','--yes','--prefix',a.prefix,'--override-channels','--channel','conda-forge',
                '--strict-channel-priority','python='+PYTHON_VERSION,'pip=24.3.1',
                'setuptools=75.8.0','wheel=0.45.1'],dict(os.environ,CONDA_AUTO_UPDATE_CONDA='false'))
    require(recovery.python_inventory(python)['python']==[3,10,20],'Requested Python version was not installed')
    return python


def software_sources_check():
    # These are the actual MA11/baseline inputs retained by the slim repository.
    import candidate,ma2_ablation_build,basic_quant_build
    candidate.sources()
    ma2_ablation_build.canonical()
    basic_quant_build.source_identity()


def verify_megatron_imports(a,python,recovered,runner):
    env=recovery.clean_environment(python,Path(recovered['runtime_cuda']))
    env.update(PYTHONPATH=str(a.megatron),CUDA_VISIBLE_DEVICES='')
    source='''import ctypes, torch
import apex_C, amp_C, fused_layer_norm_cuda
from apex.optimizers import FusedAdam
from apex.normalization import FusedLayerNorm
from megatron.training.initialize import initialize_megatron
from megatron.training.training import setup_model_and_optimizer
from pretrain_gpt import model_provider
ctypes.CDLL('libibverbs.so.1')
print('Megatron training imports, Apex extensions and libibverbs: OK (no training launched)')
'''
    runner.run([python,'-c',source],env)


def check_host(a):
    require(os.geteuid()==0,'Run inside the container as root')
    os_release=dict(line.split('=',1) for line in Path('/etc/os-release').read_text().splitlines() if '=' in line)
    require(os_release.get('ID','').strip('"')=='ubuntu' and
            os_release.get('VERSION_ID','').strip('"')=='22.04' and platform.machine()=='x86_64',
            'This script targets Ubuntu22.04 x86_64')
    if not a.skip_gpu_check:
        try: resource.setrlimit(resource.RLIMIT_MEMLOCK,(resource.RLIM_INFINITY,resource.RLIM_INFINITY))
        except (ValueError,OSError) as error:
            raise RuntimeError('Recreate the container with --ulimit memlock=-1, then rerun setup') from error


def execute(a):
    check_host(a)
    a.state_dir.mkdir(parents=True,exist_ok=True)
    lock=(a.state_dir/'setup.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('Another setup process is using '+str(a.state_dir)) from None
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    work=REPO/'outputs/h100/env-setup'/stamp;work.mkdir(parents=True)
    runner=LiveRunner(work)
    result=dict(plan(a),status='running',stage='sources',commands=runner.commands)
    report=work/'report.json'
    print('Setup log:',work/'install.log',flush=True)
    print('Detailed CUDA/PyTorch/Apex logs:',REPO/'outputs/h100/container-recovery/runs',flush=True)
    def stage(name):
        result['stage']=name;write(report,result)
        print('\nSTAGE:',name,flush=True)
    try:
        # Source mistakes are caught before downloading Torch/compiling Apex.
        add_git_trust([REPO,REPO/'3rdparty/cutlass',REPO/'3rdparty/nccl',a.megatron])
        stage('sources')
        ensure_megatron(a.megatron,runner)
        ensure_submodules(runner)
        software_sources_check()
        stage('miniforge')
        conda=ensure_miniforge(a,runner)
        stage('python-3.10.20')
        python=ensure_python(a,conda,runner)
        stage('system-libraries')
        # Include verbs userspace so future DP link tests can load libibverbs.
        packages=('libibverbs1','ibverbs-providers','ibverbs-utils','libnuma1','libnuma-dev')
        recovery.install_system(dict(system_packages=[p for p in packages if not recovery.package_installed(p)],
                                     toolkit_packages=[]),runner)
        stage('cuda-pytorch-apex-validation')
        # Reuse toolkit detection, driver-package exclusion, pinned dependencies,
        # minimal Apex extensions, compile probe, eight-H100/P2P probe and env.sh.
        command=[python,REPO/'tools/h100/phase6/restore_container.py','--execute','--repair-python',
                 '--python',python,'--megatron',a.megatron,
                 '--out',REPO/'outputs/h100/container-recovery']
        if a.skip_gpu_check: command.append('--skip-gpu-check')
        runner.run(command,dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
                               PIP_DISABLE_PIP_VERSION_CHECK='1',PIP_DEFAULT_TIMEOUT='120'))
        recovered=json.loads((REPO/'outputs/h100/container-recovery/last-report.json').read_text())
        require(recovered.get('status')=='completed' and recovered.get('software_ready') is True and
                (a.skip_gpu_check or recovered.get('gpu_checked') is True),'Environment validation incomplete')
        stage('megatron-import-validation')
        verify_megatron_imports(a,python,recovered,runner)
        result.update(status='completed',stage='done',software_ready=True,
                      gpu_checked=recovered['gpu_checked'],recovery_report=recovered)
        print('\nEnvironment configured. Activate this shell with:',flush=True)
        print('source '+shlex.quote(str(REPO/'env/activate.sh')),flush=True)
        print('Existing run_ma*.sh scripts automatically read the generated environment. Training was not started.',flush=True)
    except BaseException as error:
        result.update(status='failed',error=f'{type(error).__name__}: {error}',software_ready=False)
        raise
    finally:
        write(report,result)
        write(REPO/'outputs/h100/env-setup/last-report.json',result)
        print('Setup report:',report,flush=True)
        print('Bootstrap/source log:',work/'install.log',flush=True)
        print('CUDA/PyTorch/Apex logs:',REPO/'outputs/h100/container-recovery/runs',flush=True)
        lock.close()


def main(argv=None):
    a=arguments(argv)
    print(json.dumps(plan(a),ensure_ascii=False,indent=2),flush=True)
    if a.dry_run:
        print('PLAN ONLY: no installation, filesystem changes or GPU queries.')
        return
    execute(a)


if __name__=='__main__': main()
