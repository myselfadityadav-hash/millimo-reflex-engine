import threading
import time
import statistics
import numpy as np

# Configuration
S2_LATENCY_MS = 80         # Simulated 80ms inference of a 7B INT4 model
S1_LATENCY_MS = 5          # Simulated 5ms inference of 1-step flow matching on a GPU
S1_TARGET_HZ = 120
S2_TARGET_HZ = 10
LATENT_DIM = 512
ACTION_CHUNK_SIZE = 50
ACTION_DIM = 7

class MillimoReflexEngine:
    def __init__(self):
        # Shared memory state (Numpy arrays are memory-efficient)
        self.shared_latent_vector = np.zeros(LATENT_DIM, dtype=np.float32)
        self.lock = threading.Lock()
        
        self.current_action_chunk = np.zeros((ACTION_CHUNK_SIZE, ACTION_DIM), dtype=np.float32)
        self.chunk_step_index = ACTION_CHUNK_SIZE # Forces an update on first run
        
        self.s1_latencies = []
        self.s2_latencies = []

    def system_2_loop(self):
        """Async background thread: Simulates the 7B VLM running at 10 Hz"""
        while True:
            start_time = time.time()
            
            # Simulate heavy compute of a 7B model (takes ~80ms on edge GPU)
            time.sleep(S2_LATENCY_MS / 1000.0)
            
            # Generate a new latent vector
            new_latent = np.random.rand(LATENT_DIM).astype(np.float32)
            
            # Write to shared memory safely
            with self.lock:
                self.shared_latent_vector = new_latent
                
            elapsed_ms = (time.time() - start_time) * 1000
            self.s2_latencies.append(elapsed_ms)
            
            # Sleep to maintain ~10 Hz
            time.sleep(max(0, (1.0 / S2_TARGET_HZ) - (elapsed_ms / 1000.0)))

    def system_1_loop(self, duration_seconds=10):
        """Main control loop: Simulates the 80M Action Expert running at 120 Hz"""
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            loop_start = time.time()
            
            # Simulate 1-step flow matching compute (takes ~5ms on edge GPU)
            time.sleep(S1_LATENCY_MS / 1000.0)
            
            # Read shared memory safely
            with self.lock:
                current_intent = self.shared_latent_vector.copy()
            
            # Real-Time Chunking (RTC) logic
            if self.chunk_step_index >= ACTION_CHUNK_SIZE:
                # Generate new 50-step action chunk
                self.current_action_chunk = np.random.rand(ACTION_CHUNK_SIZE, ACTION_DIM).astype(np.float32)
                self.chunk_step_index = 0
            else:
                # Execute next step in the current chunk
                motor_command = self.current_action_chunk[self.chunk_step_index, :]
                self.chunk_step_index += 1
                
            elapsed_ms = (time.time() - loop_start) * 1000
            self.s1_latencies.append(elapsed_ms)
            
            # Sleep to maintain 120 Hz
            time.sleep(max(0, (1.0 / S1_TARGET_HZ) - (elapsed_ms / 1000.0)))

def run_benchmark():
    print("=========================================")
    print(" MILLIMO REFLEX ENGINE - PoC BENCHMARK")
    print(" (Simulated Edge GPU Timing)")
    print("=========================================")
    
    engine = MillimoReflexEngine()
    
    # Start System 2 in a background daemon thread
    s2_thread = threading.Thread(target=engine.system_2_loop, daemon=True)
    s2_thread.start()
    
    # Give S2 a moment to generate the first latent vector
    time.sleep(0.5)
    
    # Start System 1 in the main thread for 10 seconds
    print("Starting 10-second High-Speed Control Loop...\n")
    engine.system_1_loop(duration_seconds=10)
    
    # Calculate Metrics
    s1_latencies = engine.s1_latencies
    avg_s1 = statistics.mean(s1_latencies)
    max_s1 = max(s1_latencies)
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
    print(f"System 2 (Cerebrum) Loops: {len(engine.s2_latencies)}")
    print(f"System 2 Avg Latency:      {statistics.mean(engine.s2_latencies):.2f} ms")
    print("=========================================")
    
    if avg_s1 < 10.0 and actual_s1_hz >= 100:
        print("\nSUCCESS: Sub-10ms action latency achieved!")
        print("The async pipeline sustains 120Hz without blocking.")
    else:
        print("\nWARNING: Latency exceeded 10ms.")

if __name__ == "__main__":
    run_benchmark()