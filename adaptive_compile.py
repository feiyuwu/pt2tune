import torch
import dis

def estimate_complexity(fn):
    """Estimate static complexity based on bytecode instructions."""
    ops = list(dis.get_instructions(fn))
    tensor_calls = sum(1 for i in ops if "CALL" in i.opname)
    torch_ops = sum(1 for i in ops if isinstance(i.argval, str) and "torch" in i.argval)
    return tensor_calls + torch_ops

def adaptive_compile(fn, threshold_fn=None):
    """Wraps a function and delays compilation based on estimated complexity."""
    compiled = [None]
    call_count = [0]
    threshold = threshold_fn(fn) if threshold_fn else 2  # fallback threshold

    def wrapper(*args, **kwargs):
        call_count[0] += 1
        if compiled[0]:
            return compiled[0](*args, **kwargs)
        if call_count[0] >= threshold:
            print(f"Compiling function {fn.__name__} after {call_count[0]} calls")
            compiled[0] = torch.compile(fn)
            return compiled[0](*args, **kwargs)
        return fn(*args, **kwargs)
    return wrapper

def default_threshold_fn(fn):
    c = estimate_complexity(fn)
    if c < 5:
        return 3
    elif c < 15:
        return 2
    else:
        return 1
