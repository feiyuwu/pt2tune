import time
import torch
from models import add_one, loop_op, elementwise_chain, control_flow, small_conv,  TinyMLP
from typing import Callable
from adaptive_compile import adaptive_compile, default_threshold_fn
from transformers import AutoModel, AutoTokenizer

def benchmark(fn: Callable, x: torch.Tensor, n: int = 10):
    torch.cuda.empty_cache()
    times = []
    with torch.no_grad():
        for _ in range(n):
            start = time.time()
            fn(x)
            times.append(time.time() - start)
    return sum(times) / n

def get_dummy_inputs(tokenizer, device=None):
    # Create inputs for a transformer (adjust as needed)
    dummy_text = "This is a dummy input for benchmarking."
    inputs = tokenizer(dummy_text, return_tensors="pt")
    if device:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs

def run_benchmarks():
    mlp = TinyMLP()
    tests = [
        ("Trivial", add_one, torch.randn(128)),
        ("Medium", loop_op, torch.randn(128)),
        ("Elementwise Chain", elementwise_chain, torch.randn(128)),
        ("Tiny MLP", mlp.forward, torch.randn(64, 128)),
        ("Small Conv", small_conv, torch.randn(8, 3, 32, 32)),
        ("Control Flow", control_flow, torch.randn(64)),
    ]

    for name, fn, input in tests:
        print(f"\n== {name} ==")
        eager_time = benchmark(fn, input)
        compiled_time = benchmark(torch.compile(fn), input)
        adaptive_time = benchmark(adaptive_compile(fn, default_threshold_fn), input)

        print(f"Eager:    {eager_time:.6f}s")
        print(f"Compiled: {compiled_time:.6f}s")
        print(f"Adaptive: {adaptive_time:.6f}s")

        # Create tokenizer and dummy input for PyTorch gpt2
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        dummy_inputs = get_dummy_inputs(tokenizer)
        
        # --- PyTorch with torch.compile ---
        # backends = torch._dynamo.list_backends()
        # print("Available backends:", backends)
        # backends = ["inductor", "eager", 'cudagraphs', 'onnxrt', 'openxla', 'tvm']
        # for backend in backends:
        model_pt = AutoModel.from_pretrained(model_name)
        model_pt.eval()
        eager_time = benchmark(lambda inputs: model_pt(**inputs), dummy_inputs)
        compiled_time = benchmark(lambda inputs: torch.compile(model_pt)(**inputs), dummy_inputs)
        adaptive_time = benchmark(lambda inputs: adaptive_compile(model_pt.forward, default_threshold_fn)(**inputs), dummy_inputs)


        print(f"Eager:    {eager_time:.6f}s")
        print(f"Compiled: {compiled_time:.6f}s")
        print(f"Adaptive: {adaptive_time:.6f}s")

if __name__ == "__main__":
    run_benchmarks()
