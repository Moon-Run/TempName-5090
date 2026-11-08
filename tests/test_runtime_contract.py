"""CPU event-DAG checks of the real Python runtime with a recording backend.

The fake separates GPU launch, macroblock publication and GEMM completion.
It records dependencies and allocation identities, not device arithmetic,
NCCL scheduling or measured overlap. Native mapping is independently specified.
"""
import contextlib
import importlib.util
import math
from pathlib import Path
import sys
import struct
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from flux_5090.runtime import Flux5090, initialize_context, shutdown_context
from flux_5090 import reference
from flux_5090.mapping import ARMS
from flux_5090.native import mapping_reference
from flux_5090.planning import make_role
from flux_5090.validation import validate_packet_bytes, packet_tensor_valid


class Device:
    type = "cuda"
    index = 0


class Tensor:
    next_storage = 0
    def __init__(self, shape, dtype, storage=None, offset=0):
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.dtype, self.device, self.offset = dtype, Device(), offset
        self.is_cuda, self.recorded, self.contiguous = True, [], True
        if storage is None:
            Tensor.next_storage += 1
            storage = Tensor.next_storage
        self.storage = storage
    @property
    def ndim(self): return len(self.shape)
    def numel(self): return math.prod(self.shape)
    def element_size(self): return {"uint8": 1, "bf16": 2, "int32": 4}[self.dtype]
    def is_contiguous(self): return self.contiguous
    def data_ptr(self): return self.storage * 2**30 + self.offset * self.element_size()
    def record_stream(self, stream): self.recorded.append(stream)
    def view(self, *shape):
        shape = list(shape)
        if -1 in shape:
            shape[shape.index(-1)] = self.numel() // math.prod(x for x in shape if x != -1)
        assert math.prod(shape) == self.numel()
        return Tensor(shape, self.dtype, self.storage, self.offset)
    def narrow(self, dim, start, length):
        assert dim == 0 and start >= 0 and start + length <= self.shape[0]
        return Tensor((length, *self.shape[1:]), self.dtype, self.storage,
                      self.offset + start * math.prod(self.shape[1:]))
    def __getitem__(self, index):
        assert isinstance(index, int) and 0 <= index < self.shape[0]
        return Tensor(self.shape[1:], self.dtype, self.storage,
                      self.offset + index * math.prod(self.shape[1:]))


class Event:
    def __init__(self, cuda): self.cuda, self.node, self.completed = cuda, None, False
    def record(self, stream=None): self.node = self.cuda.submit("event_record", stream, event=self)
    def query(self): return self.completed
    def elapsed_time(self, other): return float(other.node - self.node)


class Stream:
    def __init__(self, cuda): self.cuda, self.last = cuda, None
    @property
    def cuda_stream(self): return id(self)
    def wait_event(self, event): self.cuda.submit("wait_event", self, extra=[event.node], event=event)
    def wait_stream(self, stream): self.cuda.submit("wait_stream", self, extra=[stream.last], waited_stream=stream)


class FakeCuda:
    def __init__(self):
        self.nodes = []
        self.current = Stream(self)
    def current_device(self): return 0
    def synchronize(self, device=None): pass
    def current_stream(self, device=None): return self.current
    def Stream(self, device=None): return Stream(self)
    def Event(self, **kwargs): return Event(self)
    @contextlib.contextmanager
    def stream(self, value):
        old, self.current = self.current, value
        try: yield
        finally: self.current = old
    def submit(self, kind, stream=None, extra=(), **metadata):
        stream = stream or self.current
        dependencies = [item for item in [stream.last, *extra] if item is not None]
        ancestors = set(dependencies)
        for item in dependencies: ancestors.update(self.nodes[item]["ancestors"])
        node = dict(kind=kind, ancestors=ancestors, stream=stream, **metadata)
        stream.last = len(self.nodes)
        self.nodes.append(node)
        return stream.last


class FakeNativePlan:
    def __init__(self, cuda, m, n, k, world, chunk_rows, order, interleaved):
        self.cuda, self.m, self.n, self.k = cuda, m, n, k
        self.world, self.chunk_rows = world, chunk_rows
        self.order, self.interleaved = list(order), interleaved
        self.workspace_bytes = 0
        self.identity = {"publication_ctas_per_macroblock": chunk_rows // 128 * (n // 128)}
        self.publications, self.forward_calls = {}, []
    def mapping(self):
        return mapping_reference(self.m, self.n, self.world, self.chunk_rows, self.order, self.interleaved)
    def probe_mapping(self): return self.mapping()
    def forward(self, input_, weight, output, counters, ready, epoch, stream=None):
        assert input_.shape == (self.m, self.k) and weight.shape == (self.n, self.k)
        assert output.shape == (self.m, self.n)
        assert counters.shape == ready.shape == (len(self.order),)
        assert len({t.storage for t in (input_, weight, output, counters, ready)}) == 5
        launch = self.cuda.submit("native_launch", epoch=epoch, plan=self)
        published = []
        for physical in self.order:
            # Publication can occur during a GEMM. It must not acquire the
            # producer stream's whole-GEMM-completion dependency by accident.
            node = self.cuda.submit("publication", Stream(self.cuda), extra=[launch],
                                    epoch=epoch, physical=physical, ready=ready)
            self.publications[(epoch, physical)] = node
            published.append(node)
        finish = self.cuda.submit("native_forward", extra=published, source=input_, weight=weight,
                                  output=output, counters=counters, ready=ready, epoch=epoch, plan=self)
        self.forward_calls.append(finish)
    def wait_ready(self, ready, physical, epoch, stream=None):
        publication = self.publications[(epoch, physical)]
        assert self.cuda.nodes[publication]["ready"] is ready
        self.cuda.submit("native_wait_ready", extra=[publication], physical=physical,
                         epoch=epoch, ready=ready, plan=self)


class FakeNative:
    def __init__(self, cuda): self.cuda, self.plans = cuda, []
    def prepare(self, *args):
        result = FakeNativePlan(self.cuda, *args)
        self.plans.append(result)
        return result


class FakeCodec:
    def __init__(self, cuda): self.cuda, self.gemm = cuda, FakeNative(cuda)
    def encode(self, source, packet, stream=None):
        self.cuda.submit("encode", source=source, output=packet)
    def decode(self, packet, output, stream=None):
        raise AssertionError("Runtime must fuse remote decode and reduction")
    def delay(self, cycles, stream=None): self.cuda.submit("diagnostic_delay", cycles=cycles)
    def reduce(self, local, remote, output, rank, world, stream=None):
        self.cuda.submit("reduce", local=local, remote=remote, output=output, rank=rank, world=world)
    def decode_reduce(self, local, packets, output, rank, world, stream=None):
        assert packets.shape == (world - 1, local.numel() // 128 * 136)
        self.cuda.submit("decode_reduce", local=local, remote=packets, output=output, rank=rank, world=world)


class FakeBackend:
    instances = []
    def __init__(self, rank=0):
        self.instances.append(self)
        self.cuda = cuda = FakeCuda()
        self.works, self.allocations = [], []
        self.group = object()
        self.groups, self.destroyed, self.networks = [], [], []
        self.torch = types.ModuleType("torch")
        self.dist = types.ModuleType("torch.distributed")
        self.torch.cuda, self.torch.version = cuda, types.SimpleNamespace(hip=None, cuda="12.9")
        self.torch.bfloat16, self.torch.uint8, self.torch.int32 = "bf16", "uint8", "int32"
        self.torch.autograd = types.SimpleNamespace(profiler=types.SimpleNamespace(
            record_function=lambda name: contextlib.nullcontext()))
        def empty(shape, *, dtype, device):
            tensor = Tensor(shape, dtype)
            self.allocations.append(tensor)
            return tensor
        self.torch.empty = empty
        def zeros(shape, *, dtype, device):
            tensor = empty(shape, dtype=dtype, device=device)
            cuda.submit("zero", output=tensor)
            return tensor
        self.torch.zeros = zeros
        def mm(*args, **kwargs): raise AssertionError("torch.mm fallback is forbidden")
        self.torch.mm = mm
        self.dist.is_initialized = lambda: True
        self.dist.get_backend = lambda group: "nccl"
        self.dist.get_rank = lambda group=None: rank
        self.dist.get_world_size = lambda group=None: 2
        self.dist.get_process_group_ranks = lambda group: [0, 1]
        def new_group(*, ranks, backend):
            assert ranks == [0, 1] and backend == "nccl"
            group = types.SimpleNamespace(lane=len(self.groups), ranks=ranks)
            self.groups.append(group); self.networks.append(Stream(cuda))
            return group
        self.dist.new_group = new_group
        self.dist.destroy_process_group = lambda group: self.destroyed.append(group)
        def exchange(receive, send, output_split_sizes, input_split_sizes, group, async_op):
            assert async_op and sum(output_split_sizes) == receive.numel() and sum(input_split_sizes) == send.numel()
            launch = cuda.submit("exchange", self.networks[group.lane], extra=[cuda.current.last],
                                 send=send, receive=receive, group=group,
                                 sends=list(input_split_sizes), receives=list(output_split_sizes))
            work = types.SimpleNamespace(node=launch, waits=0)
            def wait():
                work.waits += 1
                cuda.submit("work_wait", extra=[launch])
            work.wait = wait
            self.works.append(work)
            return work
        self.dist.all_to_all_single = exchange
        self.torch.distributed = self.dist
        self.codec = FakeCodec(cuda)
    def modules(self): return patch.dict(sys.modules, {"torch": self.torch, "torch.distributed": self.dist})
    def operator(self, arm, plan=None, diagnostic_delay_cycles=0):
        return Flux5090(512, 128, 16, self.group, arm, self.codec, chunk_rows=128,
                       plan=plan, diagnostic_delay_cycles=diagnostic_delay_cycles)
    def inputs(self): return Tensor((512, 16), "bf16"), Tensor((128, 16), "bf16")


class RuntimeContract(unittest.TestCase):
    def setUp(self): FakeBackend.instances = []
    def tearDown(self):
        for backend in FakeBackend.instances:
            with backend.modules(): shutdown_context(backend.group)

    def test_one_whole_native_gemm_and_real_cta_mapping_for_every_arm(self):
        for arm, rows in (("flux", [0, 1, 2, 3]), ("flux_taco", [0, 1, 2, 3]),
                          ("flux_taco_interleaved", [0, 2, 1, 3]),
                          ("flux_taco_interleaved_arrival", [1, 2, 0, 3])):
            backend = FakeBackend()
            with backend.modules():
                plan = make_role(512, 128, 16, 2, 128, rows) if arm == ARMS[-1] else None
                op = backend.operator(arm, plan)
                x, w = backend.inputs(); op.forward(x, w)
            self.assertEqual(len(op.native_plan.forward_calls), 1)
            native = backend.cuda.nodes[op.native_plan.forward_calls[0]]
            self.assertIs(native["source"], x)
            self.assertIs(native["weight"], w)
            self.assertIs(native["output"], op.full_partial)
            self.assertEqual(native["source"].shape, (512, 16))
            self.assertEqual(op.actual_cta_mapping, [(row, 0) for row in rows])
            self.assertEqual(op.state()["producer_launches_per_call"], 1)

    def test_each_block_waits_publication_after_prelaunch_without_full_gemm_dependency(self):
        backend = FakeBackend()
        with backend.modules():
            op = backend.operator("flux_taco_interleaved")
            x, w = backend.inputs(); op.forward(x, w)
        nodes = backend.cuda.nodes
        finish = op.native_plan.forward_calls[0]
        prelaunch = op.start_event.node
        self.assertIn(prelaunch, nodes[finish]["ancestors"])
        waits = [i for i, n in enumerate(nodes) if n["kind"] == "native_wait_ready"]
        self.assertEqual([nodes[i]["physical"] for i in waits], [0, 2, 1, 3])
        for index in waits:
            node = nodes[index]
            publication = op.native_plan.publications[(1, node["physical"])]
            self.assertIn(prelaunch, node["ancestors"])
            self.assertIn(publication, node["ancestors"])
            self.assertNotIn(finish, node["ancestors"])
            self.assertIs(node["stream"], op.communications[op.blocks[node["physical"]]["lane"]])
        exchanges = [i for i, n in enumerate(nodes) if n["kind"] == "exchange"]
        for exchange, wait in zip(exchanges, waits):
            self.assertIn(wait, nodes[exchange]["ancestors"])
            self.assertNotIn(finish, nodes[exchange]["ancestors"])
        for encode in (n for n in nodes if n["kind"] == "encode"):
            physical = encode["source"].offset // (128 * 128)
            needed = next(i for i in waits if nodes[i]["physical"] == physical)
            self.assertIn(needed, encode["ancestors"])

    def test_fused_consumer_output_fence_and_all_lanes_final_join(self):
        backend = FakeBackend(rank=1)
        with backend.modules():
            op = backend.operator("flux_taco_interleaved_arrival", make_role(512, 128, 16, 2, 128))
            x, w = backend.inputs(); output = op.forward(x, w)
        nodes, done = backend.cuda.nodes, op._last_done.node
        exchanges = [i for i,n in enumerate(nodes) if n["kind"] == "exchange"]
        reductions = [i for i,n in enumerate(nodes) if n["kind"] == "decode_reduce"]
        self.assertEqual(len(reductions), 2)
        self.assertFalse(any(n["kind"] in ("decode", "reduce") for n in nodes))
        self.assertTrue(all(block["decoded"] is None for block in op.blocks))
        for index in reductions:
            node = nodes[index]
            self.assertTrue(any(i in node["ancestors"] for i in exchanges))
            self.assertEqual(node["remote"].shape, (1, op.packet_bytes))
            self.assertEqual(node["output"].storage, output.storage)
            self.assertIn(index, nodes[done]["ancestors"])
        self.assertTrue(all(i in nodes[done]["ancestors"] for i in exchanges))
        self.assertIn(op.native_plan.forward_calls[0], nodes[done]["ancestors"])
        self.assertIn(done, nodes[backend.cuda.current.last]["ancestors"])
        join_streams = {n["waited_stream"] for n in nodes if n["kind"] == "wait_stream" and n["stream"] is op.producer}
        self.assertTrue(set((*op.communications, *op.consumers)).issubset(join_streams))
        self.assertTrue(all(work.waits == 1 for work in backend.works))
        self.assertEqual(op._pending[0][1], backend.works)
        self.assertEqual(set(output.recorded), set(op.consumers))

    def test_storage_aliases_only_views_and_quantized_arm_has_no_remote_bf16_allocation(self):
        for arm in ARMS:
            for rank in (0, 1):
                backend = FakeBackend(rank)
                with backend.modules():
                    plan = make_role(512, 128, 16, 2, 128) if arm == ARMS[-1] else None
                    op = backend.operator(arm, plan)
                    x, w = backend.inputs(); output = op.forward(x, w)
                for block in op.blocks:
                    self.assertEqual(block["partial"].storage, op.full_partial.storage)
                    self.assertEqual(block["partial"].offset, block["physical"] * 128 * 128)
                    if block["send"].numel():
                        if arm == "flux": self.assertEqual(block["send"].storage, op.full_partial.storage)
                        else: self.assertNotEqual(block["send"].storage, op.full_partial.storage)
                    if arm == "flux" and block["decoded"] is not None:
                        self.assertEqual(block["decoded"].storage, block["receive"].storage)
                if arm != "flux":
                    bf16 = [tensor for tensor in backend.allocations if tensor.dtype == "bf16"]
                    self.assertEqual([tensor.storage for tensor in bf16], [op.full_partial.storage, output.storage])
                for tensor in (op.full_partial, op.ready, op.counters):
                    self.assertEqual(set(tensor.recorded), {op.producer, *op.communications, *op.consumers})

    def test_reuse_waits_previous_consumption_outputs_and_epochs_are_independent(self):
        backend = FakeBackend()
        with backend.modules():
            op = backend.operator("flux_taco")
            x, w = backend.inputs(); first = op.forward(x, w)
            first_event, first_done = op._last_done, op._last_done.node
            second = op.forward(x, w)
            second_gemm = backend.cuda.nodes[op.native_plan.forward_calls[1]]
            self.assertEqual(len(op._pending), 2)
            first_event.completed = True
            op.forward(x, w)
        self.assertNotEqual(first.storage, second.storage)
        self.assertIn(first_done, second_gemm["ancestors"])
        self.assertEqual(second_gemm["epoch"], 2)
        self.assertEqual(op.epoch, 3)
        self.assertEqual(len(op._pending), 2)
        self.assertIn(op.producer, x.recorded)
        self.assertIn(op.producer, w.recorded)

    def test_submission_failure_poisons_all_operators_sharing_context(self):
        backend = FakeBackend()
        with backend.modules():
            first, second = backend.operator("flux_taco"), backend.operator("flux")
            x,w = backend.inputs()
            def fail(*args, **kwargs): raise RuntimeError("fake transport failure")
            backend.dist.all_to_all_single = fail
            with self.assertRaisesRegex(RuntimeError, "fake transport"):
                first.forward(x,w)
            self.assertTrue(first.failed)
            self.assertTrue(first.context["failed"])
            for op in (first, second):
                with self.assertRaisesRegex(RuntimeError, "[Ff]ailed"):
                    op.forward(x,w)
            self.assertFalse(first._lock.locked())

    def test_four_arms_share_context_groups_and_global_collective_order_on_both_ranks(self):
        traces = []
        for rank in (0, 1):
            backend = FakeBackend(rank)
            with backend.modules():
                context = initialize_context(backend.group)
                operators = [backend.operator(arm, make_role(512, 128, 16, 2, 128) if arm == ARMS[-1] else None) for arm in ARMS]
                for op in operators:
                    self.assertIs(op.context, context)
                    self.assertIs(op.producer, context["producer"])
                    x,w = backend.inputs(); op.forward(x,w)
                self.assertEqual(len(backend.groups), 2)
                self.assertEqual(len({id(x) for x in (context["producer"], *context["communications"], *context["consumers"])}), 5)
                traces.append([n for n in backend.cuda.nodes if n["kind"] == "exchange"])
                shutdown_context(backend.group)
                self.assertEqual(backend.destroyed, list(context["groups"]))
                self.assertTrue(context["closed"])
                x,w = backend.inputs()
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    operators[0].forward(x,w)
        self.assertEqual([n["group"].lane for n in traces[0]], [n["group"].lane for n in traces[1]])
        self.assertEqual(len(traces[0]), 16)
        for zero, one in zip(*traces):
            self.assertEqual(zero["sends"][1], one["receives"][0])
            self.assertEqual(one["sends"][0], zero["receives"][1])

    def test_ready_lane_bypasses_other_lane_but_final_join_covers_everything(self):
        for arm in (ARMS[2], ARMS[3]):
            backend = FakeBackend()
            with backend.modules():
                op = backend.operator(arm, make_role(512, 128, 16, 2, 128) if arm == ARMS[3] else None)
                x,w = backend.inputs(); op.forward(x,w)
            nodes = backend.cuda.nodes
            first_lane0 = next(i for i,n in enumerate(nodes) if n["kind"] == "exchange" and n["group"].lane == 0)
            lane1_reduce = next(n for n in nodes if n["kind"] == "decode_reduce" and n["output"].offset == 128 * 128)
            if arm == ARMS[2]: self.assertIn(first_lane0, lane1_reduce["ancestors"])
            else:
                self.assertNotIn(first_lane0, lane1_reduce["ancestors"])
                self.assertIs(lane1_reduce["stream"], op.consumers[1])
                self.assertEqual(sum(n["kind"] == "wait_event" and n["stream"] in op.consumers for n in nodes), 2)
            done = nodes[op._last_done.node]
            self.assertTrue(all(i in done["ancestors"] for i,n in enumerate(nodes) if n["kind"] in ("exchange", "decode_reduce")))

    def test_arrival_order_changes_native_mapping_without_changing_physical_lanes(self):
        plan = make_role(512, 128, 16, 2, 128, order=[1,3,0,2])
        backend = FakeBackend()
        with backend.modules():
            op = backend.operator(ARMS[-1], plan)
            plan["arrival_order"][:] = [0,2,1,3] # Caller mutation cannot alter frozen runtime plan.
            x,w = backend.inputs(); op.forward(x,w)
        self.assertEqual([b["lane"] for b in op.blocks], [0,1,0,1])
        self.assertEqual([item["lane"] for item in op.order], [1,1,0,0])
        self.assertEqual(op.actual_cta_mapping, [(1,0),(3,0),(0,0),(2,0)])
        self.assertEqual([n["group"].lane for n in backend.cuda.nodes if n["kind"] == "exchange"], [1,1,0,0])
        self.assertTrue(op.state()["arrival_order_changed"])

    def test_diagnostic_delay_is_profile_only_rank0_lane0_once(self):
        for rank in (0, 1):
            backend = FakeBackend(rank)
            with backend.modules():
                op = backend.operator(ARMS[-1], make_role(512, 128, 16, 2, 128), diagnostic_delay_cycles=12345)
                x,w = backend.inputs(); op.forward(x,w)
                self.assertFalse(any(n["kind"] == "diagnostic_delay" for n in backend.cuda.nodes))
                op.profile_enabled = True
                op.forward(x,w)
            delays = [n for n in backend.cuda.nodes if n["kind"] == "diagnostic_delay"]
            self.assertEqual(len(delays), 1 if rank == 0 else 0)
            if rank == 0:
                self.assertIs(delays[0]["stream"], op.communications[0])
                self.assertEqual(op.last_profile_delay, dict(physical=0,lane=0,rank=0,cycles=12345))
            self.assertFalse(op.state()["diagnostic_delay_in_normal_forward"])

    def test_input_contract_epoch_exhaustion_and_concurrent_submission_fail_early(self):
        backend = FakeBackend()
        with backend.modules():
            op = backend.operator("flux")
            x,w = backend.inputs()
            x.contiguous = False
            with self.assertRaisesRegex(ValueError, "contiguous"):
                op.forward(x,w)
            x.contiguous = True
            op.epoch = 2**31 - 1
            with self.assertRaisesRegex(RuntimeError, "epoch"):
                op.forward(x,w)
            op.epoch = 0
            op._lock.acquire()
            try:
                with self.assertRaisesRegex(RuntimeError, "Concurrent"):
                    op.forward(x,w)
            finally: op._lock.release()
            self.assertFalse(op.failed)
            self.assertFalse(op.native_plan.forward_calls)


class PacketValidationContract(unittest.TestCase):
    def test_full_soa_packet_and_fp32_minimum_scales(self):
        packet = reference.encode([0.0] * 256)
        result = validate_packet_bytes(packet, 256)
        self.assertTrue(result['passed'])
        self.assertEqual(result['groups'], 2)
        self.assertEqual(result['bytes'], 272)

    def test_nonfinite_payload_and_corrupt_scales_any_group(self):
        original = bytearray(reference.encode([1.0] * 256))
        for index, byte in ((0, 127), (255, 255)):
            damaged = original.copy(); damaged[index] = byte
            with self.assertRaisesRegex(ValueError, 'payload'):
                validate_packet_bytes(damaged, 256)
        for index, value in ((256, math.nan), (260, math.inf), (260, 0.0),
                             (264, -1.0), (268, 1001.0)):
            damaged = original.copy(); struct.pack_into('<f', damaged, index, value)
            with self.assertRaisesRegex(ValueError, 'scales'):
                validate_packet_bytes(damaged, 256)

    def test_wrong_size_and_group_count(self):
        for packet, elements in ((bytes(135),128),(bytes(272),128),(bytes(136),129)):
            with self.assertRaises(ValueError):
                validate_packet_bytes(packet,elements)

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None,
                         'PyTorch unavailable; stdlib wire gates still tested')
    def test_tensor_packet_checker_on_cpu(self):
        import torch
        raw = reference.encode([0.0] * 256)
        packet = torch.tensor(list(raw),dtype=torch.uint8)
        self.assertTrue(packet_tensor_valid(packet,256).item())
        packet[255] = 255
        self.assertFalse(packet_tensor_valid(packet,256).item())
        packet = torch.tensor(list(raw),dtype=torch.uint8)
        packet[260:264] = torch.tensor(list(struct.pack('<f', math.nan)),dtype=torch.uint8)
        self.assertFalse(packet_tensor_valid(packet,256).item())


if __name__ == "__main__":
    unittest.main()
