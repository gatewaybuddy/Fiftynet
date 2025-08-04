import time
import resource
import torch


def benchmark_module(module: torch.nn.Module, input_tensor: torch.Tensor, iterations: int = 10):
    """Measure throughput (iterations/second) and peak memory usage in MB."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device)
    module.eval()
    input_tensor = input_tensor.to(device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            module(input_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    elapsed = time.time() - start_time
    throughput = iterations / elapsed if elapsed > 0 else float("inf")
    if torch.cuda.is_available():
        mem_bytes = torch.cuda.max_memory_allocated(device)
        mem_mb = mem_bytes / (1024 ** 2)
    else:
        end_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        mem_mb = (end_mem - start_mem) / 1024
    return throughput, mem_mb
