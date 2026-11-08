"""CUDA Flux/CUTLASS producer and NCCL block pipeline with two physical transport lanes.

The four arms share one producer, two communication and two consumer streams.
Arrival arms use bounded per-lane ready consumption; this is not a global
dynamic scheduler or a claim of observed device overlap.
"""
import hashlib
import json
import threading

from .mapping import ARMS, ARRIVAL_ARMS, BACKEND, schedule, split_sizes, wire_bytes, cta_mapping

_CONTEXTS = {}
_CONTEXT_LOCK = threading.Lock()


def initialize_context(group):
    """Collectively create two NCCL groups in the same order on every rank.

    Call before constructing operators. Lazy construction remains available,
    but every rank must make that first call in the same process-group order.
    The current single-node experiment requires the parent group to cover all
    distributed ranks. Worker shutdown must destroy lane groups before WORLD.
    """
    import torch
    import torch.distributed as dist
    if getattr(torch.version, 'hip', None) or not getattr(torch.version, 'cuda', None):
        raise RuntimeError('Flux5090 requires CUDA PyTorch; no HIP/CPU fallback')
    if not dist.is_initialized() or str(dist.get_backend(group)) != 'nccl':
        raise RuntimeError('A CUDA PyTorch NCCL group is required')
    device = torch.cuda.current_device()
    key = (device, id(group))
    with _CONTEXT_LOCK:
        if key in _CONTEXTS:
            context = _CONTEXTS[key]
            if context['closed'] or context['failed']:
                raise RuntimeError('This Flux5090 context is closed or previously failed')
            return context
        size = dist.get_world_size(group)
        if size != dist.get_world_size():
            raise ValueError('This single-node context requires a parent group covering all ranks')
        if hasattr(dist, 'get_process_group_ranks'):
            ranks = list(dist.get_process_group_ranks(group))
        else:
            ranks = list(range(size))
        if ranks != list(range(size)):
            raise ValueError('Parent group must use ascending global rank order')
        # All ranks call new_group twice in exactly this order. All arms use
        # the same pair; collective submissions retain one common host order.
        groups = tuple(dist.new_group(ranks=ranks, backend='nccl') for _ in range(2))
        context = dict(group=group, groups=groups, ranks=ranks, device=device,
                       producer=torch.cuda.Stream(device=device),
                       communications=tuple(torch.cuda.Stream(device=device) for _ in range(2)),
                       consumers=tuple(torch.cuda.Stream(device=device) for _ in range(2)),
                       submission_lock=threading.Lock(), pending=[], last_done=None,
                       failed=False, closed=False)
        _CONTEXTS[key] = context
        return context


def shutdown_context(group):
    """Collective, ordered shutdown; invoke on every rank before WORLD teardown."""
    import torch
    import torch.distributed as dist
    key = (torch.cuda.current_device(), id(group))
    with _CONTEXT_LOCK:
        context = _CONTEXTS.get(key)
        if context is None:
            return
        if not context['submission_lock'].acquire(blocking=False):
            raise RuntimeError('Cannot shut down a context during operator submission')
        try:
            torch.cuda.synchronize(context['device'])
            for lane_group in context['groups']:
                dist.destroy_process_group(lane_group)
            context['pending'].clear()
            context['closed'] = True
            del _CONTEXTS[key]
        finally:
            context['submission_lock'].release()


class Flux5090:
    def __init__(self, m, n, k, group, arm, codec, chunk_rows=128, plan=None,
                 diagnostic_delay_cycles=0):
        import torch
        import torch.distributed as dist
        self.torch, self.dist = torch, dist
        self.group, self.codec, self.arm = group, codec, arm
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        self.device = torch.cuda.current_device()
        self.m, self.n, self.k, self.chunk_rows = m, n, k, chunk_rows
        self.plan = json.loads(json.dumps(plan, allow_nan=False)) if plan is not None else None
        self.order = schedule(m,n,k,self.world,chunk_rows,arm,self.plan)
        self.quantized = arm != 'flux'
        self.consumer_policy = 'per_lane_ready' if arm in ARRIVAL_ARMS else 'ordered'
        self.mapping_sha256 = hashlib.sha256(json.dumps(self.order, sort_keys=True).encode()).hexdigest()
        from .planning import role_digest
        self.plan_sha256 = role_digest(self.plan) if self.plan is not None else None
        self.context = initialize_context(group)
        self.producer = self.context['producer']
        self.communications = self.context['communications']
        self.consumers = self.context['consumers']
        self.lane_groups = self.context['groups']
        self._lock, self._pending = self.context['submission_lock'], self.context['pending']
        self.native = getattr(codec, 'gemm', None)
        if self.native is None:
            raise ValueError('The CUDA codec must be bound to a verified NativeGemm; no torch.mm fallback')
        self.native_plan = self.native.prepare(m,n,k,self.world,chunk_rows,
            [item['physical'] for item in self.order], arm in ARMS[2:])
        actual = self.native_plan.mapping()
        expected = cta_mapping(m,n,k,self.world,chunk_rows,arm,self.plan)
        if actual != expected:
            raise RuntimeError('Native Flux CTA mapping disagrees with independent specification')
        self.actual_cta_mapping = actual
        self.blocks, self.events = [], []
        self.calls = self.validation_calls = self.epoch = 0
        self.failed = self.profile_enabled = self.validate_next_call = False
        self._last_done = None
        if type(diagnostic_delay_cycles) is not int or not 0 <= diagnostic_delay_cycles <= 100000000:
            raise ValueError('Invalid bounded diagnostic delay')
        self.diagnostic_delay_cycles = diagnostic_delay_cycles
        self.last_profile_delay = None
        self.full_partial = torch.empty((m,n),dtype=torch.bfloat16,device=self.device)
        self.counters = torch.zeros(len(self.order),dtype=torch.int32,device=self.device)
        self.ready = torch.zeros(len(self.order),dtype=torch.int32,device=self.device)
        elements = chunk_rows*n
        self.packet_bytes = elements//128*136
        per = m//self.world//chunk_rows
        self.start_event = torch.cuda.Event()
        for physical in range(len(self.order)):
            destination,within=divmod(physical,per)
            local = destination == self.rank
            partial = self.full_partial.narrow(0,physical*chunk_rows,chunk_rows)
            dtype = torch.uint8 if self.quantized else torch.bfloat16
            count = self.packet_bytes if self.quantized else elements
            send = (torch.empty(0,dtype=dtype,device=self.device) if local else
                    torch.empty(count,dtype=dtype,device=self.device) if self.quantized else partial.view(-1))
            receive = torch.empty((self.world-1)*count if local else 0,dtype=dtype,device=self.device)
            remote = receive.view(self.world-1,chunk_rows,n) if local and not self.quantized else None
            sends,receives=split_sizes(self.world,self.rank,destination,count)
            self.blocks.append(dict(partial=partial,send=send,receive=receive,decoded=remote,
                sends=sends,receives=receives,quantized=self.quantized,physical=physical,
                destination=destination,within=within,lane=within%2,received=torch.cuda.Event()))
        for tensor in [self.full_partial,self.counters,self.ready]:
            for stream in (self.producer,*self.communications,*self.consumers): tensor.record_stream(stream)
        for block in self.blocks:
            for key in ('send','receive'):
                for stream in (*self.communications,*self.consumers): block[key].record_stream(stream)

    def _check(self,input_,weight):
        t=self.torch
        if self.failed or self.context['failed'] or self.context['closed']:
            raise RuntimeError('Failed/closed operator context cannot be reused')
        for label,value,shape in (('input',input_,(self.m,self.k)),('weight',weight,(self.n,self.k))):
            if tuple(value.shape)!=shape or value.dtype!=t.bfloat16 or not value.is_cuda or value.device.index!=self.device or not value.is_contiguous():
                raise ValueError(f'{label} must be contiguous CUDA BF16 {shape}')
        if self.epoch + 1 >= 2**31:
            raise RuntimeError('Ready epoch exhaustion; create a new operator')

    def forward(self,input_,weight):
        import contextlib
        t=self.torch
        self._check(input_,weight)
        if not self._lock.acquire(blocking=False): raise RuntimeError('Concurrent process-group submission unsupported')
        try:
            caller=t.cuda.current_stream(self.device)
            self.producer.wait_stream(caller)
            if self.context['last_done'] is not None: self.producer.wait_event(self.context['last_done'])
            self._pending[:]=[(event,handles) for event,handles in self._pending if not event.query()]
            input_.record_stream(self.producer); weight.record_stream(self.producer)
            output=t.empty((self.m//self.world,self.n),dtype=t.bfloat16,device=self.device)
            for stream in self.consumers: output.record_stream(stream)
            handles,markers=[],[]
            self.epoch+=1
            scope=(lambda name:t.autograd.profiler.record_function(name)) if self.profile_enabled else (lambda name:contextlib.nullcontext())
            with t.cuda.stream(self.producer),scope('flux5090/single_full_gemm'):
                # Consumers depend on a pre-launch event, never the full-GEMM
                # completion event. The local memory wait observes publication.
                self.start_event.record()
                if self.profile_enabled:
                    start=t.cuda.Event(enable_timing=True); start.record()
                self.native_plan.forward(input_,weight,self.full_partial,self.counters,self.ready,self.epoch)
                if self.profile_enabled:
                    end=t.cuda.Event(enable_timing=True); end.record()
                    markers.append(('producer',-1,-1,start,end))
            delayed=False
            self.last_profile_delay=None
            for item in self.order:
                block=self.blocks[item['physical']]
                lane,local=block['lane'],item['destination']==self.rank
                communication=self.communications[lane]
                with t.cuda.stream(communication),scope(f"flux5090/exchange/l{lane}/p{item['physical']}"):
                    communication.wait_event(self.start_event)
                    self.native_plan.wait_ready(self.ready,item['physical'],self.epoch)
                    if self.profile_enabled:
                        start=t.cuda.Event(enable_timing=True); start.record()
                    if self.profile_enabled and self.rank==0 and lane==0 and self.diagnostic_delay_cycles and not delayed:
                        self.codec.delay(self.diagnostic_delay_cycles)
                        delayed=True
                        self.last_profile_delay=dict(rank=0,lane=0,physical=item['physical'],cycles=self.diagnostic_delay_cycles)
                    if self.quantized and not local:
                        self.codec.encode(block['partial'],block['send'])
                    work=self.dist.all_to_all_single(block['receive'],block['send'],
                        output_split_sizes=block['receives'],input_split_sizes=block['sends'],
                        group=self.lane_groups[lane],async_op=True)
                    work.wait(); handles.append(work)
                    if self.profile_enabled:
                        end=t.cuda.Event(enable_timing=True); end.record()
                        markers.append(('communication',item['physical'],lane,start,end))
                    block['received'].record()
                if self.consumer_policy=='ordered' or local:
                    consumer=self.consumers[0 if self.consumer_policy=='ordered' else lane]
                    with t.cuda.stream(consumer),scope(f"flux5090/consume/l{lane}/p{item['physical']}"):
                        consumer.wait_event(block['received'])
                        if local:
                            if self.profile_enabled:
                                start=t.cuda.Event(enable_timing=True); start.record()
                            target=output.narrow(0,item['within']*self.chunk_rows,self.chunk_rows)
                            if self.quantized:
                                self.codec.decode_reduce(block['partial'],block['receive'].view(self.world-1,self.packet_bytes),
                                                         target,self.rank,self.world)
                            else:
                                self.codec.reduce(block['partial'],block['decoded'],target,self.rank,self.world)
                            if self.profile_enabled:
                                end=t.cuda.Event(enable_timing=True); end.record()
                                markers.append(('consumer',item['physical'],lane,start,end))
            for stream in (*self.communications,*self.consumers): self.producer.wait_stream(stream)
            done=t.cuda.Event(); done.record(self.producer); caller.wait_event(done)
            self._last_done=self.context['last_done']=done
            self._pending.append((done,handles)); self.events=markers; self.calls+=1
            if self.validate_next_call:
                from .validation import validate_operator_buffers
                validate_operator_buffers(self,input_,weight,output)
                self.validate_next_call=False; self.validation_calls+=1
            return output
        except BaseException:
            self.failed=self.context['failed']=True
            raise
        finally:
            self._lock.release()

    def profile_call(self,input_,weight):
        t=self.torch;t.cuda.synchronize()
        origin=t.cuda.Event(enable_timing=True);origin.record()
        self.profile_enabled=True
        try:
            result=self.forward(input_,weight);t.cuda.synchronize()
            rows=[dict(stage=stage,physical=physical,lane=lane,start_ms=origin.elapsed_time(start),end_ms=origin.elapsed_time(end))
                  for stage,physical,lane,start,end in self.events]
        finally: self.profile_enabled=False
        gemm=next(row for row in rows if row['stage']=='producer')
        early=[row['physical'] for row in rows if row['stage']=='consumer' and row['end_ms']<gemm['end_ms']]
        return result,dict(events=rows,transport_lanes=2,consumer_policy=self.consumer_policy,
            consumer_blocks_completed_before_full_gemm=early,
            diagnostic_delay=self.last_profile_delay,
            interval_limitation='Communication intervals include codec and waits; event overlap is not NCCL kernel overlap proof')

    def state(self):
        # Partial views alias one full allocation; BF16 sends alias it too.
        total=self.full_partial.numel()*2+(self.counters.numel()+self.ready.numel())*4
        total+=sum(block['receive'].numel()*block['receive'].element_size()+
                   (block['send'].numel() if self.quantized else 0) for block in self.blocks)
        wires=[wire_bytes(self.m,self.n,self.world,self.quantized)]*self.world
        base=schedule(self.m,self.n,self.k,self.world,self.chunk_rows,'flux_taco_interleaved')
        return dict(backend=BACKEND,arm=self.arm,rank=self.rank,calls=self.calls,
            shape=[self.m,self.n,self.k],chunk_rows=self.chunk_rows,production_schedule=self.order,
            mapping_sha256=self.mapping_sha256,plan_sha256=self.plan_sha256,
            producer_launches_per_call=1,collectives_per_call=len(self.order),
            uses_flux_gemm_templates=True,original_flux_kernel_unchanged=False,
            original_cuda_flux_reduce_scatter=False,cta_level_swizzle=True,native_gemm_family='FluxGemmV2-DP',
            native_gemm=self.native_plan.identity,actual_cta_mapping=self.actual_cta_mapping,
            native_workspace_bytes=self.native_plan.workspace_bytes,
            quantization='all_remote_h128_e4m3' if self.quantized else 'none',
            remote_wire_bytes_by_source=wires,remote_wire_bytes_per_source=wires[self.rank],
            reduction_order=list(range(self.world)),rounding='bf16_after_each_source',scratch_bytes=total,
            shared_streams_per_group=5,transport_lanes=2,consumer_policy=self.consumer_policy,
            validation_calls=self.validation_calls,diagnostic_delay_cycles=self.diagnostic_delay_cycles,
            diagnostic_delay_in_normal_forward=False,
            arrival_order_changed=self.plan is not None and self.order!=base,
            scheduling_granularity='cuda_cta_dp_with_macroblock_publication',
            output_publication='system-release-per-macroblock',wait_api='cuStreamWaitValue32_v2/EQ')
