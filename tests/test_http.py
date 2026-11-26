"""HTTP behavior of the translate app with a fake llama.cpp model."""

import asyncio
import os
import threading
import time
import unittest
from unittest import mock

import httpx

os.environ["MODEL_IDLE_SECONDS"] = "0"

import main  # noqa: E402
from models import loader  # noqa: E402



class FakeLlama:
    """Record how many chat completions overlap."""

    def __init__(self):
        """Start with no calls."""
        self.running = 0
        self.most = 0
        self.lock = threading.Lock()

    def create_chat_completion(self, **kwargs):
        """Hold the call open briefly so overlapping calls would show.

        Args:
            **kwargs: What the route passed.

        Returns:
            A llama.cpp-shaped completion.
        """
        with self.lock:
            self.running += 1
            self.most = max(self.most, self.running)
        time.sleep(0.05)
        with self.lock:
            self.running -= 1
        return {"choices": [{"message": {"content": "xin chao"}}]}


class TranslateHttpTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.llm = FakeLlama()
        self.llm.close = lambda: None
        self.enterContext(
            mock.patch.object(loader, "hf_hub_download", lambda **_: "fake.gguf")
        )
        self.enterContext(mock.patch.object(loader, "Llama", lambda **_: self.llm))
        self.lifespan = main.app.router.lifespan_context(main.app)
        await self.lifespan.__aenter__()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://qwen"
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.lifespan.__aexit__(None, None, None)

    async def test_overlapping_requests_reach_llama_cpp_one_at_a_time(self):
        """llama.cpp is not safe for concurrent calls into one Llama object."""
        body = {
            "text": "hello",
            "input_language": "English",
            "target_language": "Vietnamese",
        }
        responses = await asyncio.gather(
            *(self.client.post("/translate", json=body) for _ in range(4))
        )
        self.assertEqual([r.status_code for r in responses], [200] * 4)
        self.assertEqual(self.llm.most, 1)


if __name__ == "__main__":
    unittest.main()
