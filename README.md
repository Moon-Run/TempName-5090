# TempName-5090

面向 RTX 5090 的独立 Flux/CUTLASS + NCCL 实验实现，提供四组：Flux、Flux + TACO、再加交替顺序、再加到达优先。使用一次完整 GEMM、块级就绪发布、双 lane 通信和融合解码归约；不依赖旧实验的构建产物。

```bash
# 新服务器：复用已有Python/PyTorch/CUDA，获取并校验固定版本CUTLASS。
bash env/setup_5090.sh
source env/activate_5090.sh

# 计划预览，不查询GPU、不写入。
bash tools/5090/run_init.sh --trials 1 --blocks 1

# 后台启动：先做算子检查，再测 GPT 6.7B / TP8 完整训练步。
bash tools/5090/run_init.sh --place 61 --trials 1 --blocks 1 --execute
```

需要 CUDA PyTorch、CUDA 12.8+ nvcc 和默认八张 RTX 5090。CUTLASS 头文件目录不入 Git，配置脚本按仓库中的 `third_party/cutlass.lock.json` 恢复；实验首次执行时编译已校验的冻结源码。单块用于短测；正式平衡实验使用 `--blocks 8`。

默认后台运行并保存到仓库同级 `Data-5090/SL3061/<批次>/`；`--place 60` 改为 `SL3060`，显式 `--out` 优先。启动后会打印 PID 和 `controller.log` 路径，可直接使用给出的 `tail -f` 命令查看日志。需要当前终端等待时加 `--foreground`；未加 `--execute` 仍只预览。

默认编译并行度为 `nproc`，可用 `-j N` / `--jobs N` 覆盖；原生 NVCC 编译启用 threads 和 split-compile。`--blocks 4` 会报告耗时及配对加速率，但按本协议不计算置信区间。

环境复用、离线源码和配置参数见 [环境说明](env/README_5090.md)。实现、其他测试命令、报告口径和待验证边界见 [设计说明](docs/design-5090/init.md)。当前已完成 CPU 检查和 SM120 编译，多卡正确性与性能需要在目标服务器测试。这里的 Flux 为修改了排程/发布及传输的派生基线，非未修改的上游 GemmRS。

第三方来源与许可见 [NOTICE](NOTICE) 和 `third_party/`。
