import torch
import torch.nn as nn
import time
from config import *

# ==========================================
# PRODUCTION ARCHITECTURE TEMPLATES (PyTorch)
# ==========================================
# These models are structural templates for porting the system to real weights.
# The benchmark.py script uses pure Python to bypass PyTorch CPU overhead.

class System2Cerebrum(nn.Module):
    def __init__(self):
        super().__init__()
        # Proxy for OpenVLA 7B INT4. 
        # In production, replace this with: AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b", load_in_4bit=True)
        self.vision_encoder = nn.Linear(IMAGE_DIM * IMAGE_DIM * 3, 1024)
        self.language_encoder = nn.Embedding(100, 1024)
        self.intent_projector = nn.Linear(1024, S2_LATENT_DIM)
        
    def forward(self, image_tensor, lang_tokens):
        # Simulate the heavy compute of a 7B model (takes ~80ms on edge GPU)
        time.sleep(S2_LATENCY_MS / 1000.0)
        
        img_feat = self.vision_encoder(image_tensor)
        lang_feat = self.language_encoder(lang_tokens).mean(dim=1)
        combined = img_feat + lang_feat
        
        # Output: 512-dim Latent Intent Vector
        latent_vector = self.intent_projector(combined)
        return latent_vector

class System1Cerebellum(nn.Module):
    def __init__(self):
        super().__init__()
        # Lightweight Vision Encoder (e.g., ConvNeXt-Tiny)
        self.fast_vision_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 111 * 111, 256)
        )
        
        # Action Expert (Cross-attention transformer proxy)
        self.action_expert = nn.Sequential(
            nn.Linear(256 + S2_LATENT_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, S1_ACTION_CHUNK_SIZE * S1_ACTION_DIM)
        )
        
        # Flow Matching Noise Scheduler (1-step)
        self.noise_scheduler = torch.randn(S1_ACTION_CHUNK_SIZE, S1_ACTION_DIM)

    def forward(self, image_tensor, latent_vector):
        # NO sleep here. This must run as fast as possible.
        img_feat = self.fast_vision_encoder(image_tensor.permute(0, 3, 1, 2))
        
        # Concatenate fast vision features with S2 intent vector
        combined_input = torch.cat((img_feat, latent_vector), dim=1)
        
        # 1-Step Flow Matching (Consistency Model)
        # Instead of 10 denoising steps, we do 1 forward pass.
        noise = self.noise_scheduler.to(combined_input.device)
        action_chunk = self.action_expert(combined_input)
        
        # Reshape to (Batch, 50 steps, 7 DOF)
        action_chunk = action_chunk.view(-1, S1_ACTION_CHUNK_SIZE, S1_ACTION_DIM)
        return action_chunk