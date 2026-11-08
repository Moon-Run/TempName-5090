"""Self-contained Flux/CUTLASS SM120 producer and local stream readiness waits.

No Torch C++ ABI, old build output, NCCL source build or NVSHMEM dependency.
Torch is imported only by GPU-facing methods, so source/mapping checks run on CPU.
"""
from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from .build_options import nvcc_parallel_flags

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "flux-5090-native-v1"
BACKEND = "flux-cutlass-dp-ready-nccl-v1"
FLAGS = ["-O3", "-std=c++17", "-arch=sm_120", "--expt-relaxed-constexpr",
         "--expt-extended-lambda", "-Xcompiler=-fPIC", "-shared", "-cudart=shared"]
LIBRARY = "libflux_5090_gemm.so"
MAX_BLOCKS = 128


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def validate_schedule(m, n, world, chunk_rows, order, interleaved):
    """Validate the host/device contract before allocating GPU resources."""
    for name, value in (("m", m), ("n", n), ("world", world), ("chunk_rows", chunk_rows)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if world > 16 or chunk_rows % 128 or n % 128:
        raise ValueError("world <= 16 and N/chunk_rows multiples of 128 are required")
    if m % world or (m // world) % chunk_rows:
        raise ValueError("Each destination must contain complete macroblocks")
    count = m // chunk_rows
    if count > MAX_BLOCKS or len(order) != count:
        raise ValueError("Invalid macroblock count (maximum 128)")
    if any(type(x) is not int for x in order) or sorted(order) != list(range(count)):
        raise ValueError("order must be a permutation of physical macroblocks")
    if type(interleaved) is not bool:
        raise ValueError("interleaved must be bool")
    per_destination = count // world
    for slot, physical in enumerate(order):
        destination = slot % world if interleaved else slot // per_destination
        if physical // per_destination != destination:
            raise ValueError("A schedule slot cannot change its destination")
    return tuple(order)


def mapping_reference(m, n, world, chunk_rows, order, interleaved):
    """Independent Python specification; native probes use the kernel's C++ map."""
    order = validate_schedule(m, n, world, chunk_rows, order, interleaved)
    tiles_per_macro = (chunk_rows // 128) * (n // 128)
    result = []
    for logical in range((m // 128) * (n // 128)):
        if interleaved:
            within_destination, destination = divmod(logical, world)
            round_index, within = divmod(within_destination, tiles_per_macro)
            slot = round_index * world + destination
        else:
            slot, within = divmod(logical, tiles_per_macro)
        row, column = divmod(within, n // 128)
        result.append((order[slot] * (chunk_rows // 128) + row, column))
    return result


def _compiler(nvcc=None):
    explicit = nvcc or os.environ.get("NVCC")
    candidates = [explicit] if explicit else [
        str(Path(os.environ["CUDA_HOME"]) / "bin/nvcc") if os.environ.get("CUDA_HOME") else None,
        "nvcc", "/usr/local/cuda/bin/nvcc", "/data/apps/cuda/12.8/bin/nvcc"]
    found = next((shutil.which(x) for x in candidates if x and shutil.which(x)), None)
    compiler = Path(found).resolve() if found else None
    if compiler is None:
        raise RuntimeError("CUDA 12.8+ nvcc is required; set CUDA_HOME or pass nvcc")
    result = subprocess.run([str(compiler), "--version"], check=True, text=True, capture_output=True)
    version = result.stdout.strip()
    match = re.search(r"release (\d+)\.(\d+)", version)
    if not match or tuple(map(int, match.groups())) < (12, 8):
        raise RuntimeError("SM120 requires CUDA 12.8 or newer")
    return {"path": str(compiler), "version": version, "sha256": sha256(compiler),
            "environment": {key: os.environ.get(key) for key in
                ("CUDA_HOME", "CUDA_PATH", "NVCC_CCBIN", "CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH")}}


def source_inputs(root=ROOT):
    """Hash every actual compiled header, enforcing the frozen upstream snapshot."""
    root = Path(root)
    inputs = {}
    sources = {}
    for name in ("flux", "cutlass"):
        folder = root / "third_party" / name
        manifest_file = (root / "third_party/cutlass.lock.json" if name == "cutlass"
                         else folder / "manifest.json")
        if name == "cutlass":
            if not folder.is_dir() or not manifest_file.is_file():
                raise RuntimeError("CUTLASS sources/lock are missing; run bash env/setup_5090.sh "
                                   "in the TempName-5090 checkout before building")
            local_manifest = folder / "manifest.json"
            if local_manifest.exists():
                if not local_manifest.is_file() or local_manifest.read_bytes() != manifest_file.read_bytes():
                    raise RuntimeError("CUTLASS manifest differs from third_party/cutlass.lock.json; "
                                       "preserve local changes and configure with bash env/setup_5090.sh")
                inputs[str(local_manifest.relative_to(root))] = sha256(local_manifest)
        manifest = json.loads(manifest_file.read_text())
        for relative, expected in manifest["files"].items():
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise RuntimeError(f"Invalid frozen {name} source path: {relative}")
            path = folder / relative
            if not path.is_file() or sha256(path) != expected:
                hint = "; run bash env/setup_5090.sh after preserving local changes" if name == "cutlass" else ""
                raise RuntimeError(f"Frozen {name} source missing/changed: {relative}{hint}")
            inputs[str(path.relative_to(root))] = expected
        unexpected = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()}
        unexpected -= set(manifest["files"]) | {"manifest.json"}
        if unexpected:
            raise RuntimeError(f"Unregistered {name} files: {sorted(unexpected)}")
        digest = sha256(manifest_file)
        inputs[str(manifest_file.relative_to(root))] = digest
        sources[name] = {"revision": manifest["revision"], "manifest_sha256": digest}
    for relative in ("src/gemm/native.cu", "python/flux_5090/native.py", "python/flux_5090/build_options.py"):
        inputs[relative] = sha256(root / relative)
    return dict(sorted(inputs.items())), sources


def build_native(nvcc=None, build_dir=None, jobs=None):
    """Compile once under a file lock and atomically publish the complete cache."""
    for key in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        if os.environ.get(key):
            raise RuntimeError(f"Unset {key}; compiler flag injection is unsupported")
    inputs, sources = source_inputs()
    compiler = _compiler(nvcc)
    flags = [*FLAGS, *nvcc_parallel_flags(jobs)]
    identity = {"compiler": compiler, "flags": flags, "inputs": inputs}
    key = hashlib.sha256(canonical(identity).encode()).hexdigest()
    cache = Path(build_dir or os.environ.get("FLUX_5090_BUILD_DIR", ROOT / "build/native")).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / key
    with (cache / ".build.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            manifest = json.loads((target / "manifest.json").read_text())
            if (manifest["cache_key"] != key or manifest["inputs"] != inputs
                    or manifest["compiler"] != compiler or manifest["flags"] != flags
                    or sha256(target / LIBRARY) != manifest["library_sha256"]):
                raise RuntimeError(f"Native build cache identity mismatch: {target}")
            manifest["library_path"] = str(target / LIBRARY)
            return manifest
        temporary = Path(tempfile.mkdtemp(prefix=f".{key[:12]}-", dir=cache))
        try:
            include = [ROOT / "third_party/flux/include", ROOT / "third_party/flux/src",
                       ROOT / "third_party/cutlass/include", ROOT / "third_party/cutlass/tools/util/include"]
            cuda_lib = Path(compiler["path"]).parent.parent / "lib64"
            command = [compiler["path"], *flags, *[f"-I{x}" for x in include],
                       str(ROOT / "src/gemm/native.cu"), "-ldl",
                       f"-Xlinker=-rpath,{cuda_lib}", "-o", str(temporary / LIBRARY)]
            with (temporary / "compile.log").open("w") as output:
                output.write(canonical(command) + "\n")
                output.flush()
                result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Native SM120 compilation failed; see {temporary / 'compile.log'}")
            manifest = dict(protocol=PROTOCOL, backend=BACKEND, cache_key=key,
                            library_path=str(target / LIBRARY), library_sha256=sha256(temporary / LIBRARY),
                            compiler=compiler, flags=flags, inputs=inputs, sources=sources,
                            uses_flux_gemm_templates=True, original_flux_kernel_unchanged=False,
                            original_cuda_flux_reduce_scatter=False, cta_level_swizzle=True,
                            native_gemm_family="FluxGemmV2-DP", producer_launches_per_call=1,
                            split_k=False, binary_arch="sm_120", template_arch="Sm80",
                            tile=[128, 128, 32], warp=[64, 64, 32], stages=3,
                            accumulator="float32", output="bfloat16", numerical_mode="complete-K-DP",
                            readiness="per-macroblock release publication after all output stores",
                            wait_api="cuStreamWaitValue32_v2/EQ", gpu_validated=False)
            (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            temporary.rename(target)
            return manifest
        except BaseException:
            # Failed compiler logs stay available for diagnosis; no partial
            # directory ever occupies a valid cache-key destination.
            raise


class NativeGemm:
    def __init__(self, nvcc=None, build_dir=None, jobs=None):
        self.manifest = build_native(nvcc, build_dir, jobs)
        self.library_path = Path(self.manifest["library_path"])
        self._lib = ctypes.CDLL(str(self.library_path))
        integer, pointer = ctypes.c_int, ctypes.c_void_p
        self._lib.fx_last_error.restype = ctypes.c_char_p
        self._lib.fx_abi_version.restype = integer
        if self._lib.fx_abi_version() != 1:
            raise RuntimeError("Wrong native GEMM ABI")
        signatures = {
            "fx_prepare": [integer, integer, integer, integer, integer, pointer, integer, pointer],
            "fx_workspace_bytes": [pointer, pointer], "fx_destroy": [pointer],
            "fx_forward": [pointer, pointer, pointer, pointer, pointer, pointer, ctypes.c_uint32, pointer],
            "fx_wait": [pointer, integer, ctypes.c_uint32, pointer],
            "fx_map_host": [integer, integer, integer, integer, pointer, integer, pointer],
            "fx_map_device": [pointer, pointer, pointer], "fx_probe_wait": [],
            "fx_occupancy": [pointer],
        }
        for name, args in signatures.items():
            function = getattr(self._lib, name)
            function.argtypes, function.restype = args, integer

    def _check(self, result):
        if result != 0:
            raise RuntimeError(self._lib.fx_last_error().decode("utf-8", errors="replace"))

    def host_mapping(self, m, n, world, chunk_rows, order, interleaved):
        order = validate_schedule(m, n, world, chunk_rows, order, interleaved)
        native_order = (ctypes.c_int * len(order))(*order)
        count = (m // 128) * (n // 128)
        output = (ctypes.c_int * (2 * count))()
        self._check(self._lib.fx_map_host(m, n, world, chunk_rows, native_order,
                                          int(interleaved), output))
        return [(output[2 * i], output[2 * i + 1]) for i in range(count)]

    def prepare(self, m, n, k, world, chunk_rows, order, interleaved):
        return NativePlan(self, m, n, k, world, chunk_rows, order, interleaved)

    def probe_wait_value(self):
        self._check(self._lib.fx_probe_wait())
        occupancy = ctypes.c_int()
        self._check(self._lib.fx_occupancy(ctypes.byref(occupancy)))
        return {"wait_value32_supported": True, "api": "cuStreamWaitValue32_v2",
                "functional_probe": "finite already-published value",
                "maximum_active_blocks_per_sm": occupancy.value}


class NativePlan:
    def __init__(self, native, m, n, k, world, chunk_rows, order, interleaved):
        import torch
        self.native = native
        self.order = validate_schedule(m, n, world, chunk_rows, order, interleaved)
        if type(k) is not int or k <= 0 or k % 8:
            raise ValueError("K must be a positive multiple of 8")
        self.m, self.n, self.k = m, n, k
        self.world, self.chunk_rows, self.interleaved = world, chunk_rows, interleaved
        self.blocks = m // chunk_rows
        self.device = torch.cuda.current_device()
        self._handle = ctypes.c_void_p()
        native_order = (ctypes.c_int * self.blocks)(*self.order)
        native._check(native._lib.fx_prepare(m, n, k, world, chunk_rows, native_order,
                                             int(interleaved), ctypes.byref(self._handle)))
        size = ctypes.c_uint64()
        native._check(native._lib.fx_workspace_bytes(self._handle, ctypes.byref(size)))
        self.workspace_bytes = size.value
        self.identity = dict(native_gemm_family="FluxGemmV2-DP", shape=[m, n, k],
                             world=world, chunk_rows=chunk_rows, order=list(self.order),
                             interleaved=interleaved, tile=[128, 128, 32], split_k=False,
                             producer_launches_per_call=1, workspace_bytes=self.workspace_bytes,
                             publication_ctas_per_macroblock=(chunk_rows // 128) * (n // 128),
                             mapping="tile-destination-interleaved" if interleaved else "destination-major",
                             mapping_sha256=hashlib.sha256(canonical(self.mapping()).encode()).hexdigest())

    def close(self):
        if getattr(self, "_handle", None) and self._handle.value:
            self.native._check(self.native._lib.fx_destroy(self._handle))
            self._handle = ctypes.c_void_p()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _stream(self, stream):
        import torch
        if torch.cuda.current_device() != self.device:
            raise ValueError("Prepared native plan belongs to a different CUDA device")
        if stream is None:
            return torch.cuda.current_stream(self.device)
        if type(stream) is int:
            return torch.cuda.ExternalStream(stream, device=self.device)
        if not isinstance(stream, torch.cuda.Stream) or stream.device.index != self.device:
            raise ValueError("stream must be a CUDA Stream on the plan's device or its integer handle")
        return stream

    def _tensor(self, tensor, shape, dtype, name):
        import torch
        if (not isinstance(tensor, torch.Tensor) or not tensor.is_cuda or tensor.device.index != self.device
                or tensor.dtype != dtype or tuple(tensor.shape) != tuple(shape)
                or not tensor.is_contiguous() or tensor.data_ptr() % 16):
            raise ValueError(f"{name} must be aligned contiguous {dtype} {tuple(shape)} on cuda:{self.device}")

    @staticmethod
    def _epoch(epoch):
        if type(epoch) is not int or not 0 < epoch < 2**31:
            raise ValueError("epoch must be an integer in [1, 2^31)")

    def forward(self, input, weight, output, counters, ready, epoch, stream=None):
        import torch
        self._epoch(epoch)
        for tensor, shape, dtype, name in (
                (input, (self.m, self.k), torch.bfloat16, "input"),
                (weight, (self.n, self.k), torch.bfloat16, "weight"),
                (output, (self.m, self.n), torch.bfloat16, "output"),
                (counters, (self.blocks,), torch.int32, "counters"),
                (ready, (self.blocks,), torch.int32, "ready")):
            self._tensor(tensor, shape, dtype, name)
        buffers = [input, weight, output, counters, ready]
        intervals = sorted((x.data_ptr(), x.data_ptr() + x.numel() * x.element_size()) for x in buffers)
        if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
            raise ValueError("Native GEMM buffers may not alias")
        actual_stream = self._stream(stream)
        self.native._check(self.native._lib.fx_forward(self._handle, input.data_ptr(), weight.data_ptr(),
                           output.data_ptr(), counters.data_ptr(), ready.data_ptr(), epoch,
                           actual_stream.cuda_stream))
        for tensor in buffers:
            tensor.record_stream(actual_stream)
        return output

    def wait_ready(self, ready, index, epoch, stream=None):
        import torch
        self._epoch(epoch)
        self._tensor(ready, (self.blocks,), torch.int32, "ready")
        if type(index) is not int or not 0 <= index < self.blocks:
            raise ValueError("Readiness index outside prepared macroblocks")
        actual_stream = self._stream(stream)
        self.native._check(self.native._lib.fx_wait(ready.data_ptr(), index, epoch, actual_stream.cuda_stream))
        ready.record_stream(actual_stream)

    def mapping(self):
        return self.native.host_mapping(self.m, self.n, self.world, self.chunk_rows,
                                        self.order, self.interleaved)

    def probe_mapping(self, stream=None):
        import torch
        actual_stream = self._stream(stream)
        count = (self.m // 128) * (self.n // 128)
        with torch.cuda.stream(actual_stream):
            output = torch.empty((count, 2), device=self.device, dtype=torch.int32)
            self.native._check(self.native._lib.fx_map_device(self._handle, output.data_ptr(),
                                                             actual_stream.cuda_stream))
            result = [tuple(x) for x in output.cpu().tolist()]
        return result
