import torch

# Hardware config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# System 2 (Cerebrum) Config
S2_TARGET_HZ = 10          # Runs at 10 Hz
S2_LATENCY_MS = 80         # Simulated 80ms inference latency for a 7B INT4 model
S2_LATENT_DIM = 512        # Dimension of the intent vector passed to S1

# System 1 (Cerebellum) Config
S1_TARGET_HZ = 120         # Runs at 120 Hz
S1_INFERENCE_STEPS = 1     # 1-step flow matching (Consistency Distillation)
S1_ACTION_CHUNK_SIZE = 50  # Predicts 50 future actions (0.4s of motion at 120Hz)
S1_ACTION_DIM = 7          # 7-DOF robot arm (e.g., Franka Panda)

# Camera Config
CAMERA_FPS = 120
IMAGE_DIM = 224