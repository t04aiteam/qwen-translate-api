"""c09's start command sets env vars and passes the service's .env with --env-file.

The explicit values must win over the .env ones, or a .env that says
N_GPU_LAYERS=-1 or HF_HUB_OFFLINE=0 undoes the start script's choice.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

PROBE = textwrap.dedent(
    """
    import json, os, sys
    import uvicorn
    uvicorn.Config(app="main:app", env_file=sys.argv[1])
    keys = ("N_GPU_LAYERS", "HF_HUB_OFFLINE", "CUDA_VISIBLE_DEVICES", "FROM_FILE")
    print(json.dumps({k: os.environ.get(k) for k in keys}))
    """
)


class EnvFileTest(unittest.TestCase):
    """uvicorn --env-file fills only the variables the process does not have."""

    def test_explicit_env_beats_env_file(self):
        """Explicit values, also an empty one, survive; other .env keys load."""
        with tempfile.TemporaryDirectory() as d:
            env_file = Path(d, ".env")
            env_file.write_text(
                "N_GPU_LAYERS=-1\nHF_HUB_OFFLINE=0\nCUDA_VISIBLE_DEVICES=0\n"
                "FROM_FILE=yes\n"
            )
            env = {
                **{k: v for k, v in os.environ.items() if k != "FROM_FILE"},
                "N_GPU_LAYERS": "0",
                "HF_HUB_OFFLINE": "1",
                "CUDA_VISIBLE_DEVICES": "",
            }
            out = subprocess.run(
                [sys.executable, "-c", PROBE, str(env_file)],
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
        got = json.loads(out.stdout)
        self.assertEqual(
            got,
            {
                "N_GPU_LAYERS": "0",
                "HF_HUB_OFFLINE": "1",
                "CUDA_VISIBLE_DEVICES": "",
                "FROM_FILE": "yes",
            },
        )


if __name__ == "__main__":
    unittest.main()
