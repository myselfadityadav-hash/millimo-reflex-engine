import torch
import threading
import time
import statistics
from engine import MillimoReflexEngine
from config import *

def run_benchmark():
    print("=========================================")
    print(" MILLIMO REFLEX ENGINE - PoC BENCHMARK")
    print("=========================================")
    print(f"Device: {DEVICE.upper()}")
    print(f"Target S1 Frequency: {S1_TARGET_HZ} Hz")
    print(f"Target S2 Frequency: {S2_TARGET_HZ} Hz")
    print("Initializing Engine...\n")
    
    engine = MillimoReflexEngine()
    
    # Start System 2 in a background daemon thread
    s2_thread = threading.Thread(target=engine.system_2_loop, daemon=True)
    s2_thread.start()
    
    # Give S2 a moment to generate the first latent vector
    time.sleep(0.5)
    
    # Start System 1 in the main thread for 10 seconds
    print("Starting 10-second High-Speed Control Loop...\n")
    engine.system_1_loop(duration_seconds=10)
    
    # ================= CALCULATE METRICS =================
    s1_latencies = engine.s1_latencies
    s2_latencies = engine.s2_latencies
    
    avg_s1 = statistics.mean(s1_latencies)
    max_s1 = max(s1_latencies)
    avg_s2 = statistics.mean(s2_latencies)
    
    # Calculate actual achieved frequency
    total_s1_steps = len(s1_latencies)
    actual_s1_hz = total_s1_steps / 10.0
    
    print("\n=========================================")
    print(" BENCHMARK RESULTS")
    print("=========================================")
    print(f"System 1 (Reflex) Loops Executed: {total_s1_steps}")
    print(f"System 1 Avg Latency:      {avg_s1:.2f} ms")
    print(f"System 1 Max Latency:      {max_s1:.2f} ms")
    print(f"System 1 Achieved Freq:    {actual_s1_hz:.1f} Hz")
    print("-" * 40)
    print(f"System 2 (Cerebrum) Loops: {len(s2_latencies)}")
    print(f"System 2 Avg Latency:      {avg_s2:.2f} ms")
    print("=========================================")
    
    if avg_s1 < 10.0:
        print("\nSUCCESS: Sub-10ms action latency achieved!")
        print("The robot can react to physical threats faster than human reflexes.")
    else:
        print("\nWARNING: Latency exceeded 10ms. Requires further CUDA optimization.")

if __name__ == "__main__":
    run_benchmark()