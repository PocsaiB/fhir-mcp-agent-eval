"""Model adapters: one interface over the Anthropic, OpenAI and Ollama APIs.

Each adapter keeps its own native message format and returns a Turn:
  Turn(text, tool_calls=[{"id","name","args"}], input_tokens, output_tokens)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class Turn:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""


class AnthropicAdapter:
    def __init__(self, model: str, system: str, tools: list[dict], max_tokens: int = 4096):
        import anthropic
        self.client = anthropic.AsyncAnthropic()
        self.model, self.system, self.max_tokens = model, system, max_tokens
        # Thinking is adaptive by default on the 5.5 models and temperature cannot be set, so the effort level is fixed
        # for both (output is non-deterministic). Older models (e.g. Haiku 4.5) reject the effort parameter.
        self.effort = os.environ.get("CLAUDE_EFFORT", "medium") if model.endswith("-5-5") else None
        self.tools = [{"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
                      for t in tools]
        self.messages: list[dict] = []

    def start(self, question: str) -> None:
        self.messages = [{"role": "user", "content": question}]

    async def step(self) -> Turn:
        extra = {"output_config": {"effort": self.effort}} if self.effort else {}
        r = await self.client.messages.create(model=self.model, system=self.system, tools=self.tools,
                                              messages=self.messages, max_tokens=self.max_tokens, **extra)
        self.messages.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in r.content]})
        return Turn(
            text="".join(b.text for b in r.content if b.type == "text"),
            tool_calls=[{"id": b.id, "name": b.name, "args": b.input} for b in r.content if b.type == "tool_use"],
            input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens,
            stop_reason=r.stop_reason or "")

    def add_tool_results(self, results: list[tuple[str, str]]) -> None:
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": cid, "content": content} for cid, content in results]})


class OpenAIAdapter:
    def __init__(self, model: str, system: str, tools: list[dict], max_tokens: int = 1024):
        import openai
        self.client = openai.AsyncOpenAI()
        self.model, self.system, self.max_tokens = model, system, max_tokens
        self.tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                        "parameters": t["input_schema"]}} for t in tools]
        self.messages: list[dict] = []

    def start(self, question: str) -> None:
        self.messages = [{"role": "system", "content": self.system}, {"role": "user", "content": question}]

    async def step(self) -> Turn:
        r = await self.client.chat.completions.create(model=self.model, messages=self.messages, tools=self.tools,
                                                      max_completion_tokens=self.max_tokens)
        m = r.choices[0].message
        self.messages.append(m.model_dump(exclude_none=True))
        calls = []
        for tc in m.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparsable_arguments": tc.function.arguments}
            calls.append({"id": tc.id, "name": tc.function.name, "args": args})
        return Turn(text=m.content or "", tool_calls=calls,
                    input_tokens=r.usage.prompt_tokens, output_tokens=r.usage.completion_tokens,
                    stop_reason=r.choices[0].finish_reason or "")

    def add_tool_results(self, results: list[tuple[str, str]]) -> None:
        for cid, content in results:
            self.messages.append({"role": "tool", "tool_call_id": cid, "content": content})


class OllamaAdapter:
    """Local model via the native Ollama /api/chat endpoint (no API key, no cost)."""

    def __init__(self, model: str, system: str, tools: list[dict], max_tokens: int = 1024):
        import httpx
        self.http = httpx.AsyncClient(timeout=900, base_url=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"))
        self.model, self.system, self.max_tokens = model, system, max_tokens
        self.tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                        "parameters": t["input_schema"]}} for t in tools]
        self.messages: list[dict] = []
        self._names: dict[str, str] = {}

    def start(self, question: str) -> None:
        self.messages = [{"role": "system", "content": self.system}, {"role": "user", "content": question}]

    async def step(self) -> Turn:
        r = await self.http.post("/api/chat", json={
            "model": self.model, "messages": self.messages, "tools": self.tools, "stream": False,
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": self.max_tokens}})
        r.raise_for_status()
        d = r.json()
        m = d["message"]
        self.messages.append(m)
        calls = []
        for i, tc in enumerate(m.get("tool_calls") or []):
            cid = f"call_{len(self.messages)}_{i}"
            self._names[cid] = tc["function"]["name"]
            calls.append({"id": cid, "name": tc["function"]["name"], "args": tc["function"].get("arguments") or {}})
        return Turn(text=m.get("content") or "", tool_calls=calls, input_tokens=d.get("prompt_eval_count", 0),
                    output_tokens=d.get("eval_count", 0), stop_reason=d.get("done_reason", ""))

    def add_tool_results(self, results: list[tuple[str, str]]) -> None:
        for cid, content in results:
            self.messages.append({"role": "tool", "tool_name": self._names.get(cid, ""), "content": content})


def make_adapter(provider: str, model: str, system: str, tools: list[dict]):
    if provider == "ollama":
        return OllamaAdapter(model, system, tools)
    if provider == "anthropic":
        return AnthropicAdapter(model, system, tools)
    if provider == "openai":
        return OpenAIAdapter(model, system, tools)
    raise ValueError(f"Ismeretlen provider: {provider}")
