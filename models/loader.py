"""
Model loader for unsloth/Qwen3.5-4B-GGUF.
Uses llama-cpp-python to load a GGUF quantised checkpoint directly from
HuggingFace Hub. The model is loaded once at startup and reused across
all requests.

Environment variables
─────────────────────
GGUF_REPO      HuggingFace repo id   (default: unsloth/Qwen3.5-4B-GGUF)
GGUF_FILE      Filename inside repo   (default: Qwen3.5-4B-Q4_K_M.gguf)
HF_TOKEN       HF access token        (optional, for private repos)
N_GPU_LAYERS   Layers offloaded to GPU; -1 = all, 0 = CPU-only (default: -1)
N_CTX          Context window size    (default: 8192)
"""

import os
from llama_cpp import Llama

GGUF_REPO  = os.getenv("GGUF_REPO",     "unsloth/Qwen3.5-4B-GGUF")
GGUF_FILE  = os.getenv("GGUF_FILE",     "Qwen3.5-4B-Q4_K_M.gguf")
HF_TOKEN   = os.getenv("HF_TOKEN",      None)
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "-1"))   # -1 = all layers on GPU
N_CTX        = int(os.getenv("N_CTX",   "8192"))

# Kept for backwards-compat so routes can still import MODEL_ID for display
MODEL_ID = f"{GGUF_REPO}/{GGUF_FILE}"

_llm: Llama | None = None


def load_model():
    """
    Download (cached) and load the GGUF file via llama-cpp-python.
    Called once at startup via FastAPI lifespan.
    """
    global _llm

    print(f"[loader] Loading {GGUF_FILE} from {GGUF_REPO} ...")
    print(f"[loader] n_gpu_layers={N_GPU_LAYERS}  n_ctx={N_CTX}")

    _llm = Llama.from_pretrained(
        repo_id=GGUF_REPO,
        filename=GGUF_FILE,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        verbose=False,
        token=HF_TOKEN,
    )
    print("[loader] Model ready.")


def get_llm() -> Llama:
    if _llm is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _llm


# ── Legacy shims so existing import sites don't break ────────────────────────
def get_tokenizer():
    raise RuntimeError("GGUF mode: use get_llm() instead of get_tokenizer().")

def get_model():
    raise RuntimeError("GGUF mode: use get_llm() instead of get_model().")