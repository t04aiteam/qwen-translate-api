"""The GGUF loader must start from the local HF cache with no network."""

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REVISION = "0123456789abcdef0123456789abcdef01234567"

PROBE = textwrap.dedent(
    """
    import json, socket, sys
    import llama_cpp

    def no_network(*args, **kwargs):
        raise AssertionError("network used")

    socket.socket.connect = no_network
    socket.create_connection = no_network

    calls = []

    class RecordingLlama(llama_cpp.Llama):
        def __init__(self, **kwargs):
            calls.append(kwargs)

    import models.loader as loader
    loader.Llama = RecordingLlama
    try:
        loader.load_model()
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(0)
    print(json.dumps({"calls": calls}))
    """
)


class OfflineLoadTest(unittest.TestCase):
    """load_model() resolves the cached GGUF and hands its path to Llama."""

    def test_loads_cached_file_offline(self):
        """With HF_HUB_OFFLINE=1 the cached snapshot path reaches Llama."""
        import json

        with tempfile.TemporaryDirectory() as home:
            repo_dir = Path(home, "hub", "models--unsloth--Qwen3.5-4B-GGUF")
            snap = repo_dir / "snapshots" / REVISION
            snap.mkdir(parents=True)
            gguf = snap / "Qwen3.5-4B-Q4_K_M.gguf"
            gguf.write_bytes(b"GGUF fake")
            (repo_dir / "refs").mkdir()
            (repo_dir / "refs" / "main").write_text(REVISION)
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith(("HF_", "GGUF_", "N_GPU", "N_CTX"))
            }
            env.update(
                HF_HOME=home,
                HF_HUB_OFFLINE="1",
                N_GPU_LAYERS="0",
                N_CTX="4096",
                PYTHONPATH=str(REPO),
            )
            out = subprocess.run(
                [sys.executable, "-c", PROBE],
                cwd=REPO,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(out.returncode, 0, out.stderr)
            result = json.loads(out.stdout.strip().splitlines()[-1])
            self.assertNotIn("error", result, result.get("error"))
            self.assertEqual(len(result["calls"]), 1)
            kwargs = result["calls"][0]
            self.assertEqual(Path(kwargs["model_path"]).resolve(), gguf.resolve())
            self.assertEqual(kwargs["n_gpu_layers"], 0)
            self.assertEqual(kwargs["n_ctx"], 4096)


if __name__ == "__main__":
    unittest.main()
