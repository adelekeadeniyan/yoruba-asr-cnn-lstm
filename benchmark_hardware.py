import torch
import time
import os
from model import HybridASRModel

def benchmark_model():
    # Evaluate for resource-constrained edge compliance using CPU
    device = torch.device("cpu")  
    model = HybridASRModel(input_dim=80, hidden_dim=256, vocab_size=46).to(device)
    model.eval()

    # 1. Parameter Quantification
    total_params = sum(p.numel() for p in model.parameters())
    
    # 2. Memory Footprint Estimation (MB)
    torch.save(model.state_dict(), "temp_weight.pt")
    model_size_mb = os.path.getsize("temp_weight.pt") / (1024 * 1024)
    os.remove("temp_weight.pt")

    # 3. Inference Latency (Single Audio Sample Stream)
    dummy_input = torch.randn(1, 80, 237)
    
    # Warmup passes
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    start_time = time.time()
    iterations = 100
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(dummy_input)
    end_time = time.time()

    avg_latency_ms = ((end_time - start_time) / iterations) * 1000

    print("=== Edge Hardware Benchmark Results ===")
    print(f"Total Parameters: {total_params:,}")
    print(f"Model Memory Footprint: {model_size_mb:.2f} MB")
    print(f"Average Inference Latency: {avg_latency_ms:.2f} ms")

if __name__ == "__main__":
    benchmark_model()
