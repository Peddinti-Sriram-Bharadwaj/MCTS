# Alpha-Zero Hardware Profiling Report

```text

============================================================
 🚀 ALPHA-ZERO HARDWARE PROFILING REPORT 🚀 
============================================================

[Hardware Detected]
  CPU Target: CPU (cpu:0)
  GPU Target: MPS (MPS:0)

[Microbenchmarks: Forward Pass (per batch)]
  Batch Size 1              | CPU:     0.23 ms | GPU:     0.60 ms | GPU is  0.39x faster
  Batch Size 8              | CPU:     0.26 ms | GPU:     0.67 ms | GPU is  0.38x faster
  Batch Size 64             | CPU:     0.25 ms | GPU:     0.68 ms | GPU is  0.37x faster
  Batch Size 256            | CPU:     0.36 ms | GPU:     0.68 ms | GPU is  0.52x faster

[Macrobenchmarks: End-to-End]
  MCTS (50 sims)            | CPU:    62.58 ms | GPU:   349.78 ms | GPU is  0.18x faster
  Self-Play (1 Game)        | CPU:   462.77 ms | GPU:   944.25 ms | GPU is  0.49x faster
  Training Step (B=64)      | CPU:    83.37 ms | GPU:    58.50 ms | GPU is  1.43x faster

============================================================
💡 BOTTLENECK ANALYSIS & SUGGESTION 💡
============================================================

[Self-Play Phase]
-> The CPU is FASTER for Self-Play data generation.
   Reason: MCTS requires highly sequential, unbatched (Batch=1) neural net evaluations.
   GPUs suffer from high dispatch overhead on small batches, wiping out their compute advantage.

[Training Phase]
-> The GPU is moderately faster for Training.

[Final Recommendation]
🌟 ALL-CPU STRATEGY 🌟
Run everything on the CPU. GPU acceleration is not providing a net benefit for this network size.
============================================================

```
