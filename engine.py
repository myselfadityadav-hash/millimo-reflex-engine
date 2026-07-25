import torch
import threading
import time
import numpy as np
from config import *
from models import System2Cerebrum, System1Cerebellum

class MillimoReflexEngine:
    def __init__(self):
        self.s2_model = System2Cerebrum().to(DEVICE)
        self.s1_model = System1Cerebellum().to(DEVICE)
        
        # Shared Memory State
        self.shared_latent_vector = torch.zeros(1, S2_LATENT_DIM).to(DEVICE)
        self.lock = threading.Lock()
        
        # RTC (Real-Time Chunking) Buffer
        self.current_action_chunk = torch.zeros(1, S1_ACTION_CHUNK_SIZE, S1_ACTION_DIM).to(DEVICE)
        self.chunk_step_index = 0
        
        # Profiling Metrics
        self.s1_latencies = []
        self.s2_latencies = []

    def system_2_loop(self):
        """Async background thread: Runs at 10 Hz"""
        print("[System 2] Cerebrum thread started...")
        while True:
            start_time = time.time()
            
            # Simulate camera frame and language input
            dummy_image = torch.rand(1, IMAGE_DIM * IMAGE_DIM * 3).to(DEVICE)
            dummy_lang = torch.randint(0, 100, (1, 10)).to(DEVICE)
            
            # Inference
            with torch.no_grad():
                latent_vec = self.s2_model(dummy_image, dummy_lang)
            
            # Write to shared memory safely
            with self.lock:
                self.shared_latent_vector = latent_vec
                
            elapsed_ms = (time.time() - start_time) * 1000
            self.s2_latencies.append(elapsed_ms)
            
            # Sleep to maintain 10 Hz
            time.sleep(max(0, (1.0 / S2_TARGET_HZ) - (elapsed_ms / 1000.0)))

    def system_1_loop(self, duration_seconds=10):
        """Main control loop: Runs at 120 Hz"""
        print("[System 1] Cerebellum thread started...")
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            loop_start = time.time()
            
            # Simulate 120 FPS camera frame
            dummy_image = torch.rand(1, IMAGE_DIM, IMAGE_DIM, 3).to(DEVICE)
            
            # Read shared memory safely
            with self.lock:
                current_intent = self.shared_latent_vector.clone()
            
            # Inference (1-Step Flow Matching)
            with torch.no_grad():
                new_action_chunk = self.s1_model(dummy_image, current_intent)
            
            # Real-Time Chunking (RTC) - Temporal Ensembling
            if self.chunk_step_index > S1_ACTION_CHUNK_SIZE // 2:
                alpha = 0.5
                self.current_action_chunk = (alpha * self.current_action_chunk) + ((1 - alpha) * new_action_chunk)
                self.chunk_step_index = 0
            else:
                motor_command = self.current_action_chunk[0, self.chunk_step_index, :]
                self.chunk_step_index += 1
                
            elapsed_ms = (time.time() - loop_start) * 1000
            self.s1_latencies.append(elapsed_ms)
            
            time.sleep(max(0, (1.0 / S1_TARGET_HZ) - (elapsed_ms / 1000.0)))