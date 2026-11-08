# RTX 5090：Flux / TACO / 交替顺序 / 到达优先

更新：2026-11-08。实现位于独立的 `TempName-5090`，承接工作区的 [thinking.md](../../../thinking.md) 与 [ideal.md](../../../ideal.md)。这次实现四组，入口统一为 `tools/5090/run_init.sh`。**代码、CPU 检查与 SM120 编译已完成；八卡正确性、并发行为、显存和性能仍需目标服务器验证。本文没有 GPU 加速结果。**

在仓库根目录执行：

```bash
bash tools/5090/run_init.sh --trials 1 --blocks 1 --execute
```

不加 `--execute` 只输出实验计划，不查询 GPU、不编译、不写结果。默认 TP8，先预检和独立校准，再完成四组算子检查，最后测试 GPT 6.7B 完整训练步。单块是功能短测，不计算置信区间。

## 1. 四组的严格含义

| 组 | 命令内部名称 | 生产 CTA 顺序 | 远端传输 | 接收端消费 |
| --- | --- | --- | --- | --- |
| A：Flux | `flux` | 目的地连续 | BF16 | 全局固定块序 |
| B：Flux + TACO | `flux_taco` | 与 A 相同 | 全部远端 H128 / E4M3FN | 与 A 相同 |
| C：B + 交替顺序 | `flux_taco_interleaved` | CTA 按目的地轮转 | 与 B 相同 | 与 B 相同 |
| D：C + 到达优先 | `flux_taco_interleaved_arrival` | 同一轮转框架，使用目标机独立校准的块优先级 | 与 B 相同 | 两个 lane 各自按就绪依赖消费 |

四组共享 Flux/CUTLASS 主循环、块发布机制、两个 NCCL process group、五条 stream、归约顺序、缓存和训练图。`B/A` 衡量本实现中全部远端压缩的整体代价与收益，`C/B` 只改变生产排程，`D/C` 包含生产优先级与接收端独立消费两部分，不能把它拆读为某一项的单独收益。这里不包含选择性量化。

**A 是面向 5090 重建的 Flux 派生基线。** 使用真实 Flux `GemmV2` 模板与 CUTLASS Tensor Core 主循环，修改了排程与 epilogue 发布，再以 NCCL 传输块；它不是原版 `flux.GemmRS` 的未修改实现，也不是上一版 staging Flux 的原样复现。所有结果均记录 `uses_flux_gemm_templates=true`、`original_flux_kernel_unchanged=false`、`original_cuda_flux_reduce_scatter=false`，不能与旧表格跨批次直接相减。

## 2. 与旧实现相比，改变了什么

`thinking.md` 指出，旧 5090 路径等待完整 producer 之后才发通信，使提前生产的块无法提前进入下游。此次数据流为：

```text
一次完整 Flux GEMM ── 块 p 的全部 CTA 写完 ── 发布 ready[p]=epoch
        │                                      ↓
        │                                 对应 lane 等待该值
        │                                      ↓
        │                                TACO encode（B/C/D）
        │                                      ↓
        └── 仍可生产其他块                 NCCL 块传输
                                               ↓
                                      decode + 有序归约融合
                                               ↓
                                      本地对应输出行写完
所有通信及消费完成 ──────────────────────── 最终 join → 返回
```

生产者仍然是**每次投影一发完整 GEMM**，没有把 16 个块拆成 16 次 `torch.mm`。独立 cuBLAS GEMM 只用于数值参考和单独的计算耗时诊断。生产者写完完整的 `chunk_rows × N` 才能发布该通信块；本次不实现每个 128×128 tile 一次 NCCL。

核心实现：

- [`src/gemm/native.cu`](../../src/gemm/native.cu)：Flux/CUTLASS GEMM、CTA 映射、完成计数、系统域发布和 driver stream wait 的 C ABI。
- [`python/flux_5090/runtime.py`](../../python/flux_5090/runtime.py)：块级依赖、双 lane 通信与消费、输出生命周期及复用顺序。
- [`src/taco/codec.cu`](../../src/taco/codec.cu)：CUDA warp H128、FP8 编解码、融合解码归约。
- [`python/flux_5090/planning.py`](../../python/flux_5090/planning.py) 与 [`calibration.py`](../../python/flux_5090/calibration.py)：目标机测量、独立确认与冻结计划。
- [`python/flux_5090/model.py`](../../python/flux_5090/model.py)：实际 TP/SP 模型、反向、梯度累积和 AdamW 更新。
- [`tools/5090/protocol.py`](../../tools/5090/protocol.py) 与 [`report.py`](../../tools/5090/report.py)：实验身份、完整性验证、逐 rank 证据和配对报告。

上述链接均相对于本文所在的 `docs/design-5090/` 目录。

### 2.1 保留完整 K，避免把排序变化混成算术变化

模板采用 `Sm80` BF16 TensorOp 路径，实际机器码为 `sm_120`，tile=`128×128×32`，warp=`64×64×32`，3 stages，FP32 accumulator、BF16 output。模板中的 `Sm80` 是算法模板标签，实际硬件检查要求 CC 12.0 / 170 SM / RTX 5090。

使用完整 K 的 DP 工作分配，关闭 Stream-K 拆分归约和 cohort/remap。`avail_sms=1` 是使模板选择 DP 的调度参数，不表示仅使用一个物理 SM。相同输入下 B/C/D 的每个输出 tile 做相同运算，源贡献统一按 rank `0…W−1` 相加，每次加法后舍入到 BF16。算子要求 B/C/D 逐位一致，训练要求三组 loss、grad norm 和最终参数摘要一致；发现差异即失败，不能仅凭误差阈值通过就宣布排序组等价。

### 2.2 交替顺序真的进入 CTA 映射

默认 `M=2048,N=4096,W=8,chunk_rows=128`，共有 16 个通信块、512 个 CTA，每块由 32 个 CTA 共同完成。

A/B 的逻辑顺序是 `物理块 → 块内行 tile → N tile`。C/D 是 `轮次 → 块内行 tile → N tile → 目的地`，即在列 tile 层面轮转目的地，不只是改 Python 循环里的块名称。D 只在每个目的地内部选择物理块优先级，保持所有 rank 的目的地槽位和 collective 次序一致，不使用按来源 rank 各自旋转的通信序列。

主机独立公式、原生 C++ 映射和设备端 `mapping_probe` 相互检查。**探针验证的是同一映射函数，不是实际 GEMM CTA 的到达、启动或完成时间。** 物理 GPU 调度仍可改变执行先后，不能用逻辑 index 推导实际波次。

### 2.3 块发布、双 lane 和可复用缓冲

每个 CTA 完成输出写入后执行线程内 memory fence 和 CTA 同步，再更新块的完成计数；最后一个 CTA 用系统域 release 发布 epoch。每次调用清零 counters，ready 保留 epoch；通信 stream 在本地 ready 上使用 `cuStreamWaitValue32_v2/EQ`，不存在远端 GPU flag 的 P2P 访问要求。

四组均建立两套 NCCL group，按物理 `within % 2` 绑定 lane。A/B/C 的消费 stream 仍按全局块序等待；D 的两个消费者只等待所属 lane 的本地输出块，允许另一个 lane 的独立块先完成。**这是一种受限的到达优先，不能绕过同 lane 的队头，也不是任意来源、任意块的全局动态队列。**

每个实例分配一份完整 partial，块视图与 BF16 send 直接别名该存储。量化组接收端不分配完整 BF16 远端解码副本：一个 kernel 对全部远端 packet 解码，再与本地 BF16 按固定源顺序归约。SoA packet 为每个 H128 组 128 字节 FP8、4 字节量化 scale、4 字节 adaptive scale，无额外 AoS↔SoA 全量搬运。

上下文共享 1 producer + 2 communication + 2 consumer stream，每次调用返回独立输出。下一调用等待上次完整 join，避免覆盖还在发送/消费的 partial；保留异步 Work 生命周期并登记跨 stream tensor 使用。它不是跨层异步流水：相邻投影仍受模型依赖与上下文复用约束。

### 2.4 到达计划必须在目标机确认

分别为 attention 输出投影 `(2048,4096,512)` 和 MLP 输出投影 `(2048,4096,2048)` 校准。候选包括 identity、全反转、奇/偶目的地反转和轮转，去除重复序列。每次测量完整的 D 投影，包含 GEMM、编码、通信与消费；取所有 rank 的最大耗时，不将单次到达时间差当成预期收益。

默认每个候选 3 个 AB/BA 配对，每个测量 warmup=2、iterations=5；选出的一个候选再用不重合输入做 3 个确认配对。只有确认中的每一对均超过 0.2% 门槛才接受，否则回退 identity，不在确认集重选。原始 rank 样本、max-rank 值、训练/确认种子、计划 digest 全部保存，报告从样本重放选择过程。

该计划是**角色 shape 级**校准，不是每层真实激活的关键路径预测，也没有证明整步收益。D 即使回退 identity，仍保留每 lane 独立消费，报告会分别说明“生产顺序是否改变”和“人工错峰是否观测到绕过”。

## 3. 环境、依赖和运行方式

运行需要 Linux、Python 3.10+、CUDA PyTorch、可编译 SM120 的 CUDA 12.8+ `nvcc`，默认单机八张 RTX 5090。PyTorch 自身也必须能在该设备上完成 BF16、NCCL、确定性训练和 attention；预检实际执行相关运算。既有 CUDA 容器环境可以继续使用；本入口不安装或切换系统环境，不依赖旧仓库的 Megatron/Apex 或预编译 `.so`。

仓库保留冻结的 Flux `19831ca2d820e3e782ed1d15d8b52d0898b78b26`、TACO `ca717e05b6dd6d21fff4d2617d209c95be63f3ef` 源码，以及旧 OP3 codec 的来源参考。CUTLASS 3.9.0 固定为 `df8a550d3917b0e97f416b2ed8c2d786f7f686a3`：Git 只保留 [`third_party/cutlass.lock.json`](../../third_party/cutlass.lock.json) 中的 835 个文件摘要，`third_party/cutlass/` 头文件目录忽略提交，在新服务器上配置时获取。

```bash
bash env/setup_5090.sh
source env/activate_5090.sh
```

配置脚本复用已有 Python/PyTorch/CUDA，既不创建 venv，也不运行 pip/apt。默认只检查软件、获取或验证 CUTLASS，并做 SM120 小编译检查；需要八卡配置预检时显式加 `--gpu-check`。目录已存在则验证复用，内容不同则保留并报错。无网络时可加 `--cutlass-source /path/to/existing/cutlass`，从匹配锁定摘要的本地源码恢复。配置方式和参数见 [环境说明](../../env/README_5090.md)。

无需访问相邻 `TempName-New` / `flux` / `COCCL_TACO` 才能编译。实验首次执行在本批次 `build/` 下编译两份库，按源码、编译器和参数身份缓存并加锁。**实验源码快照仍包含实际 CUTLASS 头文件和外部锁**，报告继续复核全部编译输入和库文件；Git 忽略头文件不降低实验复核要求。

默认固定：

```text
CUDA_DEVICE_MAX_CONNECTIONS=8
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=0
NCCL_NVLS_ENABLE=0
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

其余已有 `NCCL_*` / `TORCH_NCCL_*` 设置保留并记录。该协议针对 NCCL staging，不要求跨卡 P2P；也不能据此声称使用了原 Flux peer-write 路径。可通过 `--connections 16` 或 `32` 声明另一批配置，不接受 1。不要把不同连接数的批次混成单因素收益。

```bash
# 只看计划；默认训练配置不变。
bash tools/5090/run_init.sh --trials 1 --blocks 1

# 原生编译、设备/通信/codec能力预检。
bash tools/5090/run_init.sh --preflight-only --execute

# 四组算子数值检查及独立 trace；包括独立校准。
bash tools/5090/run_init.sh --trials 1 --blocks 1 --operator-only --profile --execute

# 较小模型先排查完整训练流程；仍是2048行以保留两条lane。
bash tools/5090/run_init.sh --trials 1 --blocks 1 --small-model \
  --warmup-steps 2 --steps 3 --operator-warmup 3 --operator-iters 10 --execute

# 所请求的 GPT6.7B、TP8 完整短测。
bash tools/5090/run_init.sh --trials 1 --blocks 1 --execute

# 完成四次轮转及其反序，共八块的平衡实验。
bash tools/5090/run_init.sh --trials 1 --blocks 8 --execute

# 显式指定环境；脚本本身的Python也可用 RTX5090_PYTHON 设置。
bash tools/5090/run_init.sh --trials 1 --blocks 1 \
  --python /path/to/python --cuda-home /usr/local/cuda --execute

# 已完成结果只读复核，不重新跑GPU。
bash tools/5090/run_init.sh --report --out /path/to/finished-run
```

预检只证明其实际执行的能力项。完整原生 GEMM 的数值/重复输入检查在校准和 operator 阶段进行。`--profile` 的人工 lane 延迟与正常 trace 使用独立调用，排除在校准、正常算子计时和训练计时之外；普通短测也执行到达优先的独立人工错峰诊断，但不据此宣称正常负载加速。

## 4. 完整训练与报告口径

默认 32 层、hidden=4096、FFN=16384、32 heads、sequence=2048、micro_batch=1、梯度累积=4、vocab=33792、TP8/SP，BF16 模型与 FP32 AdamW master/state。embedding 与输出头共享，实际逻辑参数数为 **6,590,963,712**，每 rank **831,905,792**。模型包含 attention、MLP、残差、LayerNorm、词表并行交叉熵、反向传播、复制参数梯度归约、全局 norm、裁剪与权重更新；不能将其称为原 Megatron 完全相同的训练图。

只有每层 attention/MLP 的两个行并行前向输出投影接入这四组，共 64 个实例。反向统一 BF16 STE，微批梯度在 FP32 中累积。默认每窗口先 warmup 10 个真实 optimizer step，再计时 50 步；每步 8192 tokens、256 次目标投影，每个投影实例在正式窗口调用 200 次。初始权重和固定合成微批输入在所有组/窗口一致；测试的是训练吞吐与短轨迹一致性，不是长程收敛或模型质量。

一块执行 **10 个 torchrun worker 窗口**：1 预检 + 1 校准 + 4 算子 + 4 训练；共 200 个正式 optimizer step，各训练窗口独立初始化。八块执行 66 个窗口，其中 32 个训练窗口、1600 个正式 optimizer step。单次算子窗口默认 warmup=100、iterations=200，每个窗口测试两个投影角色。

默认输出在 `logs/5090/init-four-arm-<时间>/`：

```text
plan.json / source-manifest.json / sources/    冻结计划和实际运行源码
build/gemm/ / build/codec/                    本批两份原生库及编译记录
arrival-plan.json                            唯一冻结到达计划
results/<窗口>/config.json / command.json    实际worker配置和命令
results/<窗口>/process.log / rank*.json      全rank原始证据与异常
completed-jobs.json / batch-state.json        完成账本及状态
report.json / report.md                      完整复核后生成的报告
```

报告使用每窗口 **max-rank 墙钟时间**，分别给算子 μs、训练 ms/step、tokens/s、loss、grad norm、显存以及相邻组和总收益。性能下降为正表示更快，负值保留。八块平衡周期后才给按完整块 bootstrap 的区间，不把八个 rank 或窗口内 step 当独立样本；不同 trial 分开报告。重复、缺 rank、缺窗口、轨迹差异、编译输入/库/环境漂移都不能被“completed”标签掩盖。

Torch allocated/reserved 分开报告；块 scratch 已包含在 allocated 中，不重复累加。原生 GEMM 的 `cudaMalloc` workspace 则逐 rank 加到 allocated 再取最大。该值仍不包含 CUDA context 与 NCCL 内部占用，不是整卡总显存。目标卡能否容纳全模型以真实执行为准，发生 OOM 会保留失败窗口，不静默改模型。

## 5. 距离 ideal.md 还有哪些条件要验证

本次提供了“生产未结束时，已发布块可进入通信/消费”的执行接口，但尚未证明目标机实际实现了这种重叠：

1. **driver wait 的平台行为。** 已满足值的 `fx_probe_wait()` 只检测 API 可用；异步 producer 发布与 NCCL 并发需要真实多卡检查。CUDA 官方说明 stream memory 操作的依赖对调度器不可见，使用不当可能死锁。当前先提交 producer、使用 pre-launch event 和至少 8 connections，不能当成普遍无死锁证明；默认 worker 超时 7200 秒，可通过 `--worker-timeout` 缩短，超时终止本次 worker 进程组并保存失败现场。[CUDA Stream Memory Operations](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEMOP.html)
2. **块足够早完成。** 默认交替把多个目的地的 CTA 分散，但一个通信块仍需收齐 32 个 CTA，可能推迟第一块发布。CTA 顺序也可能损伤缓存局部性。收益必须由完整投影的实测决定。
3. **主机提交来得及。** Python 逐块提交 NCCL/codec 有开销，尤其 K=512 的 GEMM 可能在下游真正提交前就结束。`consumer_blocks_completed_before_full_gemm` 和正常 trace 用于观察；零早完成不是测试失败，更不能省略不利结果。
4. **两条 lane 有可并行资源。** 独立 process group/stream 只提供依赖上的机会，NCCL、PCIe/SHM、SM 仍可能争用或串行。通信 event 区间包含编码、通信和等待，区间相交不能代替 NCCL kernel 重叠证据。
5. **压缩收益覆盖 codec 成本。** 默认每源每次投影远端 BF16 为 14,680,064 字节，TACO 为 7,798,784 字节，约原来的 53.125%。这只说明字节预算，不保证 B 快于 A；必须同时观察编码解码成本和完整训练步。
6. **独立块更早完成能缩短最终尾部。** D 可以绕过另一 lane 的等待，但整个算子仍需最后 join。人工错峰证明可绕过也不能替代正常负载的收益；角色级校准还需面对实际层输入的差异。

本次舍弃了旧的“整次 producer 后才通信”、Stream-K 排程引入算术差异、重复 packet 搬运、单 CUDA connection、BF16 到达时间推断压缩收益和跨批次混合归因。保留的约束与剩余开销都进入原始记录，测试应允许结果为零收益或负收益。

## 6. 开发侧验证

2026-11-08 的最终验证运行共 **106 项，103 项通过、3 项 GPU 数值测试明确 skip**。使用 CPU Torch 2.6.0+cu124、`CUDA_VISIBLE_DEVICES=''`，同时启用 CUDA 12.8 编译检查与已编译原生库的 C++ 映射检查。完整运行耗时 110.829 秒；入口预览、shell 语法、Python 语法和文档链接也已检查。

CPU 契约测试覆盖 CTA 双射、完整 K 配置、块发布依赖、双 lane 消费与最终 join、输出复用、FP8 边界与 packet 数值、校准确认回退、完整训练图及结果篡改拒绝。双 rank Gloo 小模型另与独立 dense 模型比较前向和全部参数梯度。原生 GEMM 与 codec 已使用 CUDA 12.8 编译 SM120，编译产物检查到 BF16 Tensor Core 指令；这些都不等同于服务器上的 GPU 数值或性能通过。

```bash
PYTHONPATH=python python3 -m unittest discover -s tests -v

# 有 CPU Torch 环境可执行包括双rank Gloo的完整数学检查；不使用GPU。
CUDA_VISIBLE_DEVICES='' PYTHONPATH=python python3 -m unittest discover -s tests -v

# codec 的原生编译及CUDA头文件host FP8 oracle，不启动GPU。
CUDA_VISIBLE_DEVICES='' FLUX_5090_TEST_NVCC=/path/to/nvcc PYTHONPATH=python \
  python3 -m unittest discover -s tests -p test_codec_reference.py -v

# 在5090服务器上单独验证codec GPU数值及融合归约；主入口也有集成数值门禁。
FLUX_5090_TEST_CUDA=1 PYTHONPATH=python \
  python3 -m unittest discover -s tests -p test_codec_reference.py -v
```

未提供 Torch、nvcc 或显式 GPU 测试开关时，对应项目会明确 skip；不要把 skip 计为 GPU 验证通过。
