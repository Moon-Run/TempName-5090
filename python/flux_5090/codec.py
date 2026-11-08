"""CUDA SM120-only ctypes binding for the fixed TACO H128/E4M3FN wire format.

Importing this module is safe without torch or a GPU. NativeCodec construction
compiles/loads CUDA; operation methods import torch lazily. No HIP/CPU fallback.
"""
from __future__ import annotations

import ctypes
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sysconfig
import uuid

ABI_VERSION = 1
GROUP_SIZE = 128
GROUP_BYTES = 136
ROOT = Path(__file__).resolve().parents[2]
_REPOSITORY_SOURCE = ROOT / "src/taco/codec.cu"
SOURCE = (_REPOSITORY_SOURCE if _REPOSITORY_SOURCE.is_file() else
          Path(sysconfig.get_path("data")) / "share/flux-5090/src/taco/codec.cu")
DEFAULT_BUILD_DIR = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "flux-5090/taco"


def packet_bytes(elements: int) -> int:
    if (type(elements) is not int or elements <= 0 or elements % GROUP_SIZE
            or elements // 256 >= 2**31 - 1):
        raise ValueError("TACO requires a positive multiple of 128 elements")
    return elements // GROUP_SIZE * GROUP_BYTES


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compiler(nvcc=None) -> Path:
    candidates = []
    if nvcc:
        candidates.append(str(nvcc))
    elif os.environ.get("NVCC"):
        candidates.append(os.environ["NVCC"])
    else:
        candidates.extend(str(Path(os.environ[key]) / "bin/nvcc")
                          for key in ("CUDA_HOME", "CUDA_PATH") if os.environ.get(key))
        candidates.extend(("nvcc", "/usr/local/cuda/bin/nvcc"))
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return Path(found).resolve()
    raise RuntimeError("CUDA nvcc >=12.8 not found; set NVCC=/path/to/cuda/bin/nvcc")


def _build(nvcc, build_dir: Path, verbose: bool):
    if not SOURCE.is_file():
        raise RuntimeError(f"CUDA TACO source is missing: {SOURCE}; install the package with its share/flux-5090 source data")
    compiler = _compiler(nvcc)
    version = subprocess.run([str(compiler), "--version"], check=True, capture_output=True,
                             text=True, timeout=30)
    version_text = version.stdout + version.stderr
    match = re.search(r"release\s+(\d+)\.(\d+)", version_text)
    if not match or tuple(map(int, match.groups())) < (12, 8):
        raise RuntimeError("RTX 5090 requires nvcc >=12.8 with sm_120 support")
    # Extra flags can alter rounding or target selection. Keep this protocol's
    # compilation fully specified rather than accepting ambient nvcc injection.
    for key in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        if os.environ.get(key):
            raise RuntimeError(f"Unset {key}; compiler flag injection is unsupported")
    flags = ["-arch=sm_120", "-O3", "--fmad=false", "-std=c++17", "--shared", "-Xcompiler=-fPIC"]
    identity = dict(abi=ABI_VERSION, source_sha256=_sha(SOURCE), compiler=str(compiler),
                    compiler_sha256=_sha(compiler), compiler_version=version_text, flags=flags,
                    compiler_environment={key: os.environ.get(key) for key in
                        ("CUDA_HOME", "CUDA_PATH", "NVCC_CCBIN", "CPATH", "CPLUS_INCLUDE_PATH",
                         "LIBRARY_PATH", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")})
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    directory = Path(build_dir).expanduser().resolve() / key
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / "libflux_5090_taco.so"
    manifest_path = directory / "manifest.json"
    with (directory / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = None
        if manifest_path.exists():
            try:
                old = json.loads(manifest_path.read_text())
            except (ValueError, OSError):
                old = None
        valid = (library.is_file() and isinstance(old, dict) and old.get("identity") == identity
                 and old.get("library_sha256") == _sha(library))
        if not valid:
            temporary = directory / (".codec-" + uuid.uuid4().hex + ".so")
            command = [str(compiler), *flags, str(SOURCE), "-o", str(temporary)]
            if verbose:
                print("Compiling CUDA TACO codec:", command, flush=True)
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=900)
                (directory / "compile.log").write_text(result.stdout + result.stderr)
                if result.returncode or not temporary.is_file():
                    raise RuntimeError(f"CUDA codec compilation failed (exit {result.returncode}); see {directory / 'compile.log'}\n{result.stderr[-3000:]}")
                manifest = dict(identity=identity, command=command, library_sha256=_sha(temporary))
                os.replace(temporary, library)
                manifest_temp = directory / (".manifest-" + uuid.uuid4().hex + ".json")
                manifest_temp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
                os.replace(manifest_temp, manifest_path)
                old = manifest
            finally:
                temporary.unlink(missing_ok=True)
    return library, old


class NativeCodec:
    """Finite BF16 input; output storage and async lifetime are caller-owned.

    encode/decode use [payload][FP32 quant scales][FP32 adaptive scales] without
    per-group padding. All buffers must be contiguous CUDA tensors on the current
    device. Calls report launch errors; callers synchronize at validation/timing
    boundaries to catch asynchronous device faults.
    """

    def __init__(self, nvcc=None, build_dir: Path = DEFAULT_BUILD_DIR, verbose=False):
        self.library_path, self.manifest = _build(nvcc, Path(build_dir), bool(verbose))
        self.lib = ctypes.CDLL(str(self.library_path))
        self.lib.flux_5090_codec_abi.argtypes = []
        self.lib.flux_5090_codec_abi.restype = ctypes.c_int
        if self.lib.flux_5090_codec_abi() != ABI_VERSION:
            raise RuntimeError("CUDA TACO library ABI mismatch")
        self.lib.flux_5090_codec_error.argtypes = [ctypes.c_int]
        self.lib.flux_5090_codec_error.restype = ctypes.c_char_p
        signature = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
        for name in ("flux_5090_encode", "flux_5090_decode"):
            function = getattr(self.lib, name)
            function.argtypes = signature
            function.restype = ctypes.c_int
        self.lib.flux_5090_reduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.c_int64, ctypes.c_int, ctypes.c_int,
                                            ctypes.c_void_p]
        self.lib.flux_5090_reduce.restype = ctypes.c_int
        self.lib.flux_5090_decode_reduce.argtypes = self.lib.flux_5090_reduce.argtypes
        self.lib.flux_5090_decode_reduce.restype = ctypes.c_int
        self.lib.flux_5090_diagnostic_delay.argtypes = [ctypes.c_uint64, ctypes.c_void_p]
        self.lib.flux_5090_diagnostic_delay.restype = ctypes.c_int

    @staticmethod
    def _torch():
        torch = importlib.import_module("torch")
        if getattr(torch.version, "hip", None) or not getattr(torch.version, "cuda", None):
            raise RuntimeError("NativeCodec requires NVIDIA CUDA PyTorch, not HIP/CPU")
        if tuple(torch.cuda.get_device_capability()) != (12, 0):
            raise RuntimeError("NativeCodec is built specifically for RTX 5090 SM120")
        return torch

    @staticmethod
    def _check_tensor(torch, tensor, dtype, name):
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype:
            raise TypeError(f"{name} must be a {dtype} tensor")
        if tensor.device.type != "cuda" or not tensor.is_contiguous() or tensor.numel() <= 0:
            raise ValueError(f"{name} must be a nonempty contiguous CUDA device tensor")
        if tensor.device.index != torch.cuda.current_device():
            raise ValueError(f"{name} must be on the current CUDA device")

    @staticmethod
    def _no_overlap(a, b):
        a0, b0 = a.data_ptr(), b.data_ptr()
        a1, b1 = a0 + a.numel() * a.element_size(), b0 + b.numel() * b.element_size()
        if max(a0, b0) < min(a1, b1):
            raise ValueError("Codec input/output buffers must not overlap")

    @staticmethod
    def _stream(torch, stream):
        if stream is None:
            return int(torch.cuda.current_stream().cuda_stream)
        if type(stream) is not int or not 0 <= stream < 2**(8 * ctypes.sizeof(ctypes.c_void_p)):
            raise TypeError("stream must be a valid unsigned CUDA stream pointer integer or None")
        return stream

    def _checked(self, status):
        if status:
            message = self.lib.flux_5090_codec_error(status)
            raise RuntimeError(f"CUDA TACO launch failed ({status}): {message.decode(errors='replace') if message else 'unknown error'}")

    def encode(self, input_bf16_contiguous, packet_uint8_contiguous, stream=None):
        torch = self._torch()
        source, packet = input_bf16_contiguous, packet_uint8_contiguous
        self._check_tensor(torch, source, torch.bfloat16, "input")
        self._check_tensor(torch, packet, torch.uint8, "packet")
        if packet.numel() != packet_bytes(source.numel()):
            raise ValueError("Packet size must be exactly input.numel()/128*136 bytes")
        if packet.ndim != 1:
            raise ValueError("Each packet must be one-dimensional")
        if packet.data_ptr() % 4:
            raise ValueError("Packet storage must be aligned to 4 bytes for FP32 scales")
        self._no_overlap(source, packet)
        self._checked(self.lib.flux_5090_encode(source.data_ptr(), packet.data_ptr(), source.numel(),
                                              self._stream(torch, stream)))
        return packet

    def delay(self, cycles: int, stream=None):
        """Bounded single-thread device delay for separate diagnostic profiles."""
        if type(cycles) is not int or not 0 < cycles <= 100000000:
            raise ValueError('Diagnostic delay requires 0 < cycles <= 100000000')
        torch = self._torch()
        self._checked(self.lib.flux_5090_diagnostic_delay(cycles, self._stream(torch, stream)))

    def decode(self, packet_uint8_contiguous, output_bf16_contiguous, stream=None):
        torch = self._torch()
        packet, output = packet_uint8_contiguous, output_bf16_contiguous
        self._check_tensor(torch, packet, torch.uint8, "packet")
        self._check_tensor(torch, output, torch.bfloat16, "output")
        if packet.numel() != packet_bytes(output.numel()):
            raise ValueError("Packet size must be exactly output.numel()/128*136 bytes")
        if packet.ndim != 1:
            raise ValueError("Each packet must be one-dimensional")
        if packet.data_ptr() % 4:
            raise ValueError("Packet storage must be aligned to 4 bytes for FP32 scales")
        if output.data_ptr() % 8:
            raise ValueError("Decode output storage must be aligned to 8 bytes")
        self._no_overlap(packet, output)
        self._checked(self.lib.flux_5090_decode(packet.data_ptr(), output.data_ptr(), output.numel(),
                                              self._stream(torch, stream)))
        return output

    def reduce(self, local_bf16, remote_bf16, output_bf16_contiguous, rank: int, world: int, stream=None):
        torch = self._torch()
        local, remote, output = local_bf16, remote_bf16, output_bf16_contiguous
        for tensor, name in ((local, "local"), (remote, "remote"), (output, "output")):
            self._check_tensor(torch, tensor, torch.bfloat16, name)
        if type(rank) is not int or type(world) is not int or not 2 <= world <= 128 or not 0 <= rank < world:
            raise ValueError("Require 2<=world<=128 and 0<=rank<world")
        if local.ndim != 2 or output.shape != local.shape or tuple(remote.shape) != (world - 1, *local.shape):
            raise ValueError("Expected local/output[rows,n] and remote[world-1,rows,n], sources ascending excluding local rank")
        packet_bytes(local.numel())
        if local.data_ptr() % 8 or output.data_ptr() % 8 or remote.data_ptr() % 2:
            raise ValueError("Local/output require 8-byte alignment; remote BF16 requires 2")
        self._no_overlap(local, remote)
        self._no_overlap(local, output)
        self._no_overlap(remote, output)
        self._checked(self.lib.flux_5090_reduce(local.data_ptr(), remote.data_ptr(), output.data_ptr(),
                                              local.numel(), rank, world, self._stream(torch, stream)))
        return output

    def decode_reduce(self, local_bf16, packets_uint8, output_bf16_contiguous,
                      rank: int, world: int, stream=None):
        """Fused remote decode + ordered BF16 sum, with no decoded GPU buffer.

        Packets are independent SoA rows, ordered by ascending source rank with
        the local source omitted. Each source decode and each sum round to BF16.
        """
        torch = self._torch()
        local, packets, output = local_bf16, packets_uint8, output_bf16_contiguous
        self._check_tensor(torch, local, torch.bfloat16, "local")
        self._check_tensor(torch, output, torch.bfloat16, "output")
        self._check_tensor(torch, packets, torch.uint8, "packets")
        if type(rank) is not int or type(world) is not int or not 2 <= world <= 128 or not 0 <= rank < world:
            raise ValueError("Require 2<=world<=128 and 0<=rank<world")
        if local.ndim != 2 or output.shape != local.shape or tuple(packets.shape) != (world - 1, packet_bytes(local.numel())):
            raise ValueError("Expected local/output[rows,n] and packets[world-1,packet_bytes], sources ascending excluding local rank")
        if local.data_ptr() % 8 or output.data_ptr() % 8 or packets.data_ptr() % 4:
            raise ValueError("Local/output require 8-byte alignment; packets require 4")
        self._no_overlap(local, packets)
        self._no_overlap(local, output)
        self._no_overlap(packets, output)
        self._checked(self.lib.flux_5090_decode_reduce(local.data_ptr(), packets.data_ptr(), output.data_ptr(),
                                                     local.numel(), rank, world, self._stream(torch, stream)))
        return output
