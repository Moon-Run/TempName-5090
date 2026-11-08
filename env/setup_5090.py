#!/usr/bin/env python3
"""Prepare pinned CUTLASS and validate an existing Python/PyTorch/CUDA environment."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / 'outputs/5090/env-setup'
PROTOCOL = 'rtx5090-existing-environment-cutlass-v2'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class Runner:
    def __init__(self, root):
        self.root, self.commands = Path(root), []

    def run(self, command, env=None, cwd=None, timeout=900):
        command = list(map(str, command))
        self.commands.append(dict(argv=command, cwd=str(cwd) if cwd else None))
        print('+ ' + shlex.join(command), flush=True)
        with (self.root / 'setup.log').open('a') as log:
            log.write('\nCOMMAND ' + shlex.join(command) + '\n')
            log.flush()
            result = subprocess.run(command, env=env, cwd=cwd, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=timeout)
        require(result.returncode == 0,
                f'Command exited {result.returncode}; see {self.root / "setup.log"}')


def capture(command, env=None, timeout=120):
    command = list(map(str, command))
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=timeout)
    require(result.returncode == 0, 'Command failed: ' + shlex.join(command) + '\n' + result.stderr)
    return result.stdout


def load_lock(repo=REPO):
    path = Path(repo) / 'third_party/cutlass.lock.json'
    require(path.is_file(), 'Missing tracked CUTLASS lock: ' + str(path))
    lock = read(path)
    require(lock.get('source') == 'https://github.com/NVIDIA/cutlass', 'Unexpected CUTLASS source in lock')
    require(isinstance(lock.get('revision'), str) and re.fullmatch(r'[0-9a-f]{40}', lock['revision']), 'Invalid CUTLASS commit')
    files = lock.get('files')
    require(isinstance(files, dict) and files and 'LICENSE.txt' in files, 'Invalid CUTLASS file lock')
    require('include/cutlass/cutlass.h' in files, 'CUTLASS headers missing from lock')
    for relative, digest in files.items():
        require(isinstance(relative, str), 'Invalid CUTLASS file name')
        parts = PurePosixPath(relative)
        require(not parts.is_absolute() and '..' not in parts.parts and str(parts) == relative
                and '\\' not in relative and relative != 'manifest.json', 'Unsafe CUTLASS path: ' + relative)
        require(isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest), 'Invalid CUTLASS hash: ' + relative)
    return lock


def validate_cutlass(directory, lock, lock_bytes=None, *, snapshot=False):
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(), 'CUTLASS must be a real directory: ' + str(directory))
    for relative, expected in lock['files'].items():
        path = directory / relative
        require(not any((directory / Path(*Path(relative).parts[:index])).is_symlink()
                        for index in range(1, len(Path(relative).parts) + 1)),
                'Symlinked CUTLASS input is unsupported: ' + str(path))
        require(path.is_file() and sha(path) == expected,
                'CUTLASS missing/modified file: ' + str(path) +
                '; existing files were preserved. Move this directory aside or restore the locked version, then rerun setup.')
    manifest = directory / 'manifest.json'
    if manifest.exists():
        require(manifest.is_file() and not manifest.is_symlink(), 'Invalid CUTLASS manifest')
        require(manifest.read_bytes() == lock_bytes if lock_bytes is not None else read(manifest) == lock,
                'CUTLASS local manifest differs from third_party/cutlass.lock.json; directory preserved')
    if snapshot:
        unexpected = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
        unexpected -= set(lock['files']) | {'manifest.json'}
        require(not unexpected, 'CUTLASS snapshot contains unregistered files: ' + str(sorted(unexpected)[:10]) +
                '; directory preserved. Use a full upstream checkout via --cutlass-source with an absent destination.')
    return dict(revision=lock['revision'], files_verified=len(lock['files']), directory=str(directory))


def ensure_cutlass(args, runner, repo=REPO):
    """Only install an absent directory, and publish it after every locked hash matches."""
    repo = Path(repo)
    lock_path = repo / 'third_party/cutlass.lock.json'
    lock = load_lock(repo)
    locked_bytes = lock_path.read_bytes()
    destination = repo / 'third_party/cutlass'
    if destination.exists() or destination.is_symlink():
        result = validate_cutlass(destination, lock, locked_bytes, snapshot=True)
        return dict(result, action='verified-existing', lock_sha256=sha(lock_path), network_used=False)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    # Both the download and extraction stay in our unique temporary directory;
    # interruption cannot leave a half-installed third_party/cutlass tree.
    with tempfile.TemporaryDirectory(prefix='.cutlass-setup-', dir=parent) as temporary:
        temporary = Path(temporary)
        source = args.cutlass_source
        if source is None:
            require(shutil.which('git'), 'Git is required to download CUTLASS; or use --cutlass-source /existing/cutlass')
            source = temporary / 'checkout'
            runner.run(['git', '-c', 'init.defaultBranch=setup', 'init', source])
            git = ['git', '-C', source, '-c', 'core.hooksPath=/dev/null']
            runner.run([*git, 'fetch', '--depth', '1', '--no-tags', lock['source'], lock['revision']])
            runner.run([*git, '-c', 'advice.detachedHead=false', 'checkout', '--detach', 'FETCH_HEAD'])
            actual = capture([*git, 'rev-parse', 'HEAD']).strip()
            require(actual == lock['revision'], 'Downloaded CUTLASS commit differs from lock')
        validate_cutlass(source, lock)
        prepared = temporary / 'prepared'
        prepared.mkdir()
        for relative in lock['files']:
            target = prepared / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, target)
        (prepared / 'manifest.json').write_bytes(locked_bytes)
        validate_cutlass(prepared, lock, locked_bytes, snapshot=True)
        # Cooperative setup invocations hold one setup lock. Preserve a directory
        # created independently while the download was running as well.
        require(not destination.exists() and not destination.is_symlink(),
                'CUTLASS appeared during setup; preserved, rerun for validation')
        prepared.rename(destination)
    return dict(validate_cutlass(destination, lock, locked_bytes, snapshot=True), action='installed',
                lock_sha256=sha(lock_path), network_used=args.cutlass_source is None,
                source=str(args.cutlass_source) if args.cutlass_source else lock['source'])


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-python', '--python', dest='base_python', type=Path,
                        default=Path(sys.executable), help='Reuse this existing interpreter, including an active venv')
    parser.add_argument('--cuda-home', type=Path, help='Reuse this existing CUDA Toolkit')
    parser.add_argument('--nvcc', help='Explicit existing CUDA compiler')
    parser.add_argument('--cutlass-source', type=Path, help='Offline source checkout/header snapshot; locked file hashes must match')
    parser.add_argument('--dependencies-only', action='store_true', help='Only prepare CUTLASS; no Torch/CUDA checks or activation file')
    gpu = parser.add_mutually_exclusive_group()
    gpu.add_argument('--gpu-check', action='store_true', help='Also run the four-arm runner GPU preflight, without training')
    gpu.add_argument('--skip-gpu-check', action='store_true', help='Compatibility alias for the default software-only mode')
    parser.add_argument('--dry-run', action='store_true', help='Print a plan; no subprocesses, writes or GPU queries')
    args = parser.parse_args(argv)
    # Do not resolve Python symlinks: venv/bin/python must retain its venv identity.
    args.base_python = args.base_python.expanduser().absolute()
    for key in ('cuda_home', 'cutlass_source'):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.expanduser().resolve())
    require(not (args.dependencies_only and args.gpu_check), '--dependencies-only cannot run --gpu-check')
    return args


def plan(args, repo=REPO):
    lock = load_lock(repo)
    return dict(protocol=PROTOCOL, python=str(args.base_python), reuse_existing_environment=True,
                minimum_python='3.10', minimum_cuda='12.8', target_arch='sm_120',
                cuda_home=str(args.cuda_home) if args.cuda_home else 'auto: CUDA_HOME/CUDA_PATH/nvcc',
                nvcc=args.nvcc, cutlass_source=str(args.cutlass_source) if args.cutlass_source else lock['source'],
                cutlass_revision=lock['revision'], cutlass_lock='third_party/cutlass.lock.json',
                cutlass_directory=str(Path(repo) / 'third_party/cutlass'), cutlass_files=len(lock['files']),
                install_python=False, install_torch=False, install_cuda=False, install_packages=False,
                create_venv=False, requires_megatron=False, requires_apex=False,
                dependencies_only=args.dependencies_only, gpu_check=args.gpu_check,
                training_started=False)


def software_environment(python, cuda_home=None, nvcc=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='')
    env.pop('PYTHONHOME', None)
    env.pop('PYTHONPATH', None)
    paths = [str(Path(python).parent)]
    if cuda_home:
        env['CUDA_HOME'] = str(cuda_home)
        paths.append(str(Path(cuda_home) / 'bin'))
    if nvcc:
        env['NVCC'] = str(nvcc)
    env['PATH'] = os.pathsep.join(paths + [env.get('PATH', os.defpath)])
    return env


STACK_PROBE = r'''
import importlib.util, json, sys
import numpy
import torch
print('RTX5090_JSON:'+json.dumps(dict(python=list(sys.version_info[:3]), executable=sys.executable,
    prefix=sys.prefix, base_prefix=sys.base_prefix, torch=str(torch.__version__),
    torch_file=torch.__file__, numpy=str(numpy.__version__), cuda=torch.version.cuda,
    hip=getattr(torch.version, 'hip', None), distributed=torch.distributed.is_available(),
    nccl_available=torch.distributed.is_nccl_available(),
    arch_flags=torch._C._cuda_getArchFlags(), profiler=hasattr(torch, 'profiler'))))
'''


def inventory(python, cuda_home, nvcc):
    output = capture([python, '-B', '-c', STACK_PROBE], software_environment(python, cuda_home, nvcc))
    lines = [line[len('RTX5090_JSON:'):] for line in output.splitlines() if line.startswith('RTX5090_JSON:')]
    require(len(lines) == 1, 'Existing Python environment did not return one inventory record')
    return json.loads(lines[0])


def validate_stack(value):
    require(tuple(value['python'][:2]) >= (3, 10), 'Python 3.10+ is required; select an existing interpreter with --base-python')
    require(value['cuda'] and not value['hip'], 'Existing PyTorch must be a NVIDIA CUDA build')
    version = re.match(r'(\d+)\.(\d+)', value['cuda'])
    require(version and tuple(map(int, version.groups())) >= (12, 8),
            'Existing PyTorch CUDA build must support RTX 5090 (CUDA 12.8+); this script does not replace PyTorch')
    require('sm_120' in (value['arch_flags'] or '').split(), 'Existing PyTorch build lacks SM120 kernels')
    require(value['distributed'] and value['nccl_available'] and value['profiler'],
            'Existing PyTorch lacks required distributed/NCCL/profiler support')


def find_toolkit(args):
    explicit = args.nvcc or os.environ.get('NVCC')
    root = args.cuda_home or (Path(os.environ['CUDA_HOME']) if os.environ.get('CUDA_HOME') else
                             Path(os.environ['CUDA_PATH']) if os.environ.get('CUDA_PATH') else None)
    candidates = [str(Path(args.cuda_home) / 'bin/nvcc')] if args.cuda_home and not args.nvcc else (
        [explicit] if explicit else [str(root / 'bin/nvcc')] if root else ['nvcc', '/usr/local/cuda/bin/nvcc'])
    found = next((shutil.which(str(x)) for x in candidates if shutil.which(str(x))), None)
    require(found, 'Existing nvcc not found; pass --cuda-home /path/to/cuda or --nvcc /path/to/nvcc')
    compiler = Path(found).resolve()
    actual_root = compiler.parent.parent
    if args.cuda_home:
        require((args.cuda_home / 'bin/nvcc').resolve() == compiler,
                '--nvcc and --cuda-home select different toolkits')
    text = capture([compiler, '--version'])
    version = re.search(r'\brelease (\d+)\.(\d+)', text)
    require(version and tuple(map(int, version.groups())) >= (12, 8), 'Existing nvcc must be CUDA 12.8+ for SM120')
    require('sm_120' in capture([compiler, '--list-gpu-code']).split(), 'Existing nvcc does not support SM120')
    require((actual_root / 'include/cuda_runtime.h').is_file() and
            (actual_root / 'lib64/libcudart.so').is_file(), 'Existing CUDA Toolkit headers/runtime are incomplete')
    return dict(root=str(actual_root), compiler=str(compiler), version=text.strip(), compiler_sha256=sha(compiler))


def compile_probe(args, toolkit, runner):
    for key in ('NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS'):
        require(not os.environ.get(key), 'Unset ' + key + '; implicit CUDA flags are unsupported by this experiment')
    source = runner.root / 'sm120-probe.cu'
    source.write_text('#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <cuda_fp8.h>\n'
                      '__global__ void probe(__nv_bfloat16* x) { x[threadIdx.x] = __float2bfloat16(1.f); }\n')
    output = runner.root / 'sm120-probe.o'
    runner.run([toolkit['compiler'], '-std=c++17', '-arch=sm_120', '-c', source, '-o', output],
               software_environment(args.base_python, toolkit['root'], toolkit['compiler']))
    require(output.is_file() and output.stat().st_size, 'SM120 compile probe produced no object')
    return dict(object=str(output), sha256=sha(output), gpu_used=False)


def verify_gpu(args, toolkit, runner):
    out = runner.root / 'gpu-preflight'
    env = software_environment(args.base_python, toolkit['root'], toolkit['compiler'])
    env.pop('CUDA_VISIBLE_DEVICES')
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        env['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
    runner.run([args.base_python, REPO / 'tools/5090/init.py', '--execute', '--foreground', '--preflight-only',
                '--python', args.base_python, '--cuda-home', toolkit['root'], '--nvcc', toolkit['compiler'],
                '--out', out], env, timeout=7500)
    state, report = read(out / 'batch-state.json'), read(out / 'report.json')
    require(state.get('status') == 'preflight_completed' and state.get('completed') is True
            and report.get('completed') is True and report.get('completed_scope') == 'preflight',
            'Four-arm runner GPU preflight did not complete')
    return dict(report=str(out / 'report.json'), sha256=sha(out / 'report.json'))


def environment_text(args, toolkit):
    values = dict(RTX5090_SETUP_PYTHON=str(args.base_python), RTX5090_PYTHON=str(args.base_python),
                  RTX5090_SETUP_CUDA_HOME=toolkit['root'], CUDA_HOME=toolkit['root'],
                  NVCC=toolkit['compiler'], RTX5090_SETUP_PROTOCOL=PROTOCOL)
    return '# Generated after existing-environment validation; no new environment was installed.\n' + ''.join(
        'export ' + name + '=' + shlex.quote(value) + '\n' for name, value in values.items())


def execute(args):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / 'setup.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another setup_5090.sh is running for this checkout') from None
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
        work = OUTPUT / 'runs' / stamp
        work.mkdir(parents=True)
        runner = Runner(work)
        result = dict(plan(args), status='running', software_ready=False, dependencies_ready=False,
                      gpu_checked=False, gpu_checks_pending=True, commands=runner.commands)
        def stage(name):
            result['stage'] = name
            write(work / 'report.json', result)
            print('STAGE:', name, flush=True)
        try:
            if not args.dependencies_only:
                stage('check-existing-python-pytorch-cuda')
                require(args.base_python.is_file(), 'Existing Python not found: ' + str(args.base_python))
                toolkit = find_toolkit(args)
                result['toolkit'] = toolkit
                result['environment'] = inventory(args.base_python, toolkit['root'], toolkit['compiler'])
                validate_stack(result['environment'])
            stage('prepare-locked-cutlass')
            result['cutlass'] = ensure_cutlass(args, runner, REPO)
            result['dependencies_ready'] = True
            if args.dependencies_only:
                result.update(status='dependencies_ready', stage='done')
                print('CUTLASS ready. Python/PyTorch/CUDA and GPU checks were not run.', flush=True)
                return
            stage('sm120-compile-check')
            result['compile_probe'] = compile_probe(args, toolkit, runner)
            if args.gpu_check:
                stage('four-arm-runner-gpu-preflight')
                result['gpu_preflight'] = verify_gpu(args, toolkit, runner)
                result.update(gpu_checked=True, gpu_checks_pending=False)
            stage('save-existing-environment-paths')
            target = OUTPUT / 'env.sh'
            temporary = target.with_suffix('.sh.tmp')
            temporary.write_text(environment_text(args, toolkit))
            temporary.replace(target)
            result.update(status='completed', stage='done', software_ready=True)
            print('Ready: source env/activate_5090.sh', flush=True)
            print('Then: bash tools/5090/run_init.sh --trials 1 --blocks 1 --execute', flush=True)
            if not args.gpu_check:
                print('GPU checks remain pending; run_init.sh performs preflight before the experiments.', flush=True)
        except BaseException as exc:
            result.update(status='failed', software_ready=False, error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            write(work / 'report.json', result)
            write(OUTPUT / 'last-report.json', result)
            print('Setup report:', work / 'report.json', flush=True)


def main(argv=None):
    args = arguments(argv)
    print(json.dumps(plan(args), indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        print('PLAN ONLY: no installs, writes, subprocesses or GPU queries.')
        return
    execute(args)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        print('ERROR:', error, file=sys.stderr)
        raise SystemExit(1)
