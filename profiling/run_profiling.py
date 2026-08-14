import argparse
import subprocess
import json
import os
import sys

def run_subprocess(game, device, output_file):
    print(f"\n--- Launching {device.upper()} Profiling ---")
    
    # We set JAX_PLATFORMS to force the backend. 
    # For CPU, it's 'cpu'. For GPU, it might be 'mps' on Mac or 'cuda' on Linux.
    env = os.environ.copy()
    if device == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
    else:
        # We try to let JAX decide its best GPU (cuda/mps).
        # We just remove JAX_PLATFORMS if it's set to CPU so it defaults to GPU.
        if "JAX_PLATFORMS" in env:
            del env["JAX_PLATFORMS"]

    cmd = [sys.executable, "core_profiler.py", "--game", game, "--output", output_file]
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError:
        print(f"Error running profiling on {device.upper()}.")
        return False
    return True

def print_report(cpu_metrics, gpu_metrics):
    report_lines = []
    
    def out(line):
        print(line)
        report_lines.append(line)

    out("\n" + "="*60)
    out(" 🚀 ALPHA-ZERO HARDWARE PROFILING REPORT 🚀 ")
    out("="*60)
    
    out(f"\n[Hardware Detected]")
    out(f"  CPU Target: {cpu_metrics['hardware']['backend'].upper()} ({cpu_metrics['hardware']['device']})")
    out(f"  GPU Target: {gpu_metrics['hardware']['backend'].upper()} ({gpu_metrics['hardware']['device']})")

    def compare(metric_name, label):
        c = cpu_metrics[metric_name]
        g = gpu_metrics[metric_name]
        speedup = c / g if g > 0 else 0
        out(f"  {label:<25} | CPU: {c:8.2f} ms | GPU: {g:8.2f} ms | GPU is {speedup:5.2f}x faster")
        return speedup

    out("\n[Microbenchmarks: Forward Pass (per batch)]")
    for b in cpu_metrics["forward_pass"]:
        compare(f"forward_pass_{b}", f"Batch Size {b}")

    out("\n[Macrobenchmarks: End-to-End]")
    mcts_speedup = compare("mcts", "MCTS (50 sims)")
    sp_speedup = compare("self_play", "Self-Play (1 Game)")
    train_speedup = compare("training", "Training Step (B=64)")

    out("\n" + "="*60)
    out("💡 BOTTLENECK ANALYSIS & SUGGESTION 💡")
    out("="*60)
    
    out("\n[Self-Play Phase]")
    if sp_speedup < 1.0:
        out("-> The CPU is FASTER for Self-Play data generation.")
        out("   Reason: MCTS requires highly sequential, unbatched (Batch=1) neural net evaluations.")
        out("   GPUs suffer from high dispatch overhead on small batches, wiping out their compute advantage.")
    else:
        out("-> The GPU is FASTER for Self-Play data generation.")
        
    out("\n[Training Phase]")
    if train_speedup > 2.0:
        out("-> The GPU is MASSIVELY FASTER for Training.")
        out("   Reason: Training operates on large batches (e.g., B=64+). GPUs excel at massively parallel matrix multiplications.")
    elif train_speedup > 1.0:
        out("-> The GPU is moderately faster for Training.")
    else:
        out("-> The CPU is faster for Training. (This is unusual and suggests a very small network or missing GPU drivers).")

    out("\n[Final Recommendation]")
    if sp_speedup < 1.0 and train_speedup > 1.5:
        out("🌟 HYBRID STRATEGY 🌟")
        out("Generate Self-Play games on the CPU (using Python multiprocessing).")
        out("Perform the actual neural network Training on the GPU.")
    elif sp_speedup > 1.0 and train_speedup > 1.0:
        out("🌟 ALL-GPU STRATEGY 🌟")
        out("Your GPU is fast enough to overcome dispatch overheads. Run everything on the GPU!")
    else:
        out("🌟 ALL-CPU STRATEGY 🌟")
        out("Run everything on the CPU. GPU acceleration is not providing a net benefit for this network size.")
    out("="*60 + "\n")
    
    # Save to Markdown file
    with open("profiling_report.md", "w") as f:
        f.write("# Alpha-Zero Hardware Profiling Report\n\n")
        f.write("```text\n")
        f.write("\n".join(report_lines))
        f.write("\n```\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="connect4")
    args = parser.parse_args()

    cpu_out = "cpu_metrics.json"
    gpu_out = "gpu_metrics.json"

    if os.path.exists(cpu_out): os.remove(cpu_out)
    if os.path.exists(gpu_out): os.remove(gpu_out)

    if not run_subprocess(args.game, "cpu", cpu_out):
        return
    if not run_subprocess(args.game, "gpu", gpu_out):
        return

    with open(cpu_out, "r") as f:
        cpu_metrics = json.load(f)
    with open(gpu_out, "r") as f:
        gpu_metrics = json.load(f)

    # Some basic extraction since forward_pass is nested
    def flatten(m):
        flat = {}
        flat["hardware"] = m["hardware"]
        for b, v in m["forward_pass"].items():
            flat[f"forward_pass_{b}"] = v
        flat["mcts"] = m["mcts"]
        flat["self_play"] = m["self_play"]
        flat["training"] = m["training"]
        return flat

    c_flat = flatten(cpu_metrics)
    g_flat = flatten(gpu_metrics)
    
    # Repackage forward pass back into dict to use compare loop cleanly
    c_flat["forward_pass"] = cpu_metrics["forward_pass"]
    g_flat["forward_pass"] = gpu_metrics["forward_pass"]

    print_report(c_flat, g_flat)

if __name__ == "__main__":
    main()
