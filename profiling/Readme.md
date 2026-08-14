# Hardware Acceleration Profiling Suite

This directory contains a specialized benchmarking framework designed to isolate, measure, and analyze the performance characteristics of the AlphaZero algorithm across disparate hardware environments (CPU, NVIDIA CUDA, Apple Silicon MPS).

## Profiling Architecture

The suite operates by decoupling the computational workloads of the AlphaZero algorithm into distinct micro-benchmarks and macro-benchmarks to identify systemic bottlenecks.

### Component Breakdown
- **`core_profiler.py`**: A low-level execution timer utilizing `time.perf_counter`. It instruments precise latency measurements for both purely mathematical operations (JAX neural network inference) and algorithmic state-tracking operations (MCTS traversal and tree expansion).
- **`run_profiling.py`**: The overarching orchestration script. It automates the environment variable injection (`JAX_PLATFORMS`) required to force JAX to compile against specific hardware backends at runtime, bypassing the default static hardware locking mechanism.

## Metrics and Benchmarks

The profiling suite captures and aggregates the following metrics into persistent JSON logs (`cpu_metrics.json`, `gpu_metrics.json`):

1. **JIT Compilation Latency**: The overhead required by the XLA compiler to trace and optimize the computational graph for the target hardware architecture.
2. **Micro-Benchmark (Inference)**: High-frequency batches of raw network forward passes (`@nnx.jit`) to determine the raw FLOP throughput of the hardware.
3. **Macro-Benchmark (Search)**: End-to-end MCTS tree search execution. This exposes the critical dispatch bottleneck where sequential CPU-bound control-flow logic interacts with hardware-accelerated matrix multiplications.

## Execution Guide

To generate a comprehensive performance analysis report, execute the orchestration script. The script will sequentially test the available hardware backends and output a comparative Markdown analysis.

```bash
python run_profiling.py
```

### Analytical Objective
The primary objective of this module is to empirically determine the optimal hardware allocation for AlphaZero. While GPUs drastically accelerate large-batch neural network training (e.g., during the replay buffer optimization phase), sequential MCTS self-play operates on a batch size of 1. On specific architectures (such as Apple Silicon MPS), the asynchronous dispatch overhead of offloading tiny matrices to the GPU can paradoxically render the CPU faster for self-play data generation. This suite quantifies that threshold.
