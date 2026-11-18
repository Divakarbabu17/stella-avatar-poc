"""TEST DOUBLE ONLY - NOT vLLM and never part of the application.

Mimics the parts of vLLM's OpenAI-compatible server that the orchestrator's
VllmQwenClient uses, so the client can be tested on the CPU laptop:

    GET  /health               200 with an EMPTY body (like vLLM), never protected
    GET  /v1/models            requires "Authorization: Bearer <key>"
    POST /v1/chat/completions  requires the key; rejects requests without
                               chat_template_kwargs.enable_thinking == false;
                               stream=true -> SSE chunks (+ usage chunk, [DONE])
    GET  /metrics              a few vLLM Prometheus gauges (not protected)

The reply echoes what was received (message count, system prompt, last user message).

Run on the CPU Compose network (from the repo root):
    docker compose --profile cli run -d --name stella-vllm-stub \
      -v ./tests:/opt/tests:ro -w /opt/tests cli \
      uvicorn vllm_contract_stub:app --host 0.0.0.0 --port 8002

Then exercise the real client against it:
    docker compose --profile cli run --rm cli python -c "
from app.orchestrator.clients import VllmQwenClient
c = VllmQwenClient('qwen', 'http://stella-vllm-stub:8002', 30, model='stub-model', api_key='stub-key',
                   system_prompt='You are Stella.', max_new_tokens=32)
print(c.health())
print(c.generate_with_metrics('Hi', history=[{'role': 'user', 'content': 'Earlier'}, {'role': 'assistant', 'content': 'Reply'}]))
print(c.engine_metrics())"
"""

import json
import os
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, Response, StreamingResponse

API_KEY = os.environ.get("STUB_API_KEY", "stub-key")
MODEL = os.environ.get("STUB_MODEL", "stub-model")
app = FastAPI(title="vLLM contract stub (TEST ONLY)")


def _auth(request: Request) -> None:
    if request.headers.get("authorization") != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health():
    return Response(status_code=200)  # empty body, as vLLM does


@app.get("/metrics")
def metrics():
    return PlainTextResponse(
        "# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.\n"
        "# TYPE vllm:kv_cache_usage_perc gauge\n"
        f'vllm:kv_cache_usage_perc{{engine="0",model_name="{MODEL}"}} 0.125\n'
        f'vllm:num_requests_running{{engine="0",model_name="{MODEL}"}} 0.0\n'
        f'vllm:num_requests_waiting{{engine="0",model_name="{MODEL}"}} 0.0\n'
    )


@app.get("/v1/models")
def models(request: Request):
    _auth(request)
    return {"object": "list", "data": [{"id": MODEL, "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    _auth(request)
    body = await request.json()
    if body.get("model") != MODEL:
        raise HTTPException(status_code=404, detail=f"The model `{body.get('model')}` does not exist.")
    if (body.get("chat_template_kwargs") or {}).get("enable_thinking") is not False:
        raise HTTPException(status_code=400, detail="stub: expected chat_template_kwargs.enable_thinking=false")
    messages = body.get("messages") or []
    roles = [m.get("role") for m in messages]
    if not roles or roles[0] != "system" or roles[-1] != "user":
        raise HTTPException(status_code=400, detail=f"stub: unexpected message roles {roles}")
    reply = (
        f"<think></think>stub reply: {len(messages)} messages, roles={roles}, "
        f"system={messages[0]['content']!r}, last={messages[-1]['content']!r}, "
        f"max_completion_tokens={body.get('max_completion_tokens')}, top_k={body.get('top_k')}"
    )
    if body.get("stream"):
        words = reply.split(" ")
        include_usage = (body.get("stream_options") or {}).get("include_usage")

        def events():
            time.sleep(0.2)  # "prompt processing" before the first token
            for index, word in enumerate(words):
                delta = {"content": word if index == 0 else f" {word}"}
                finish = "stop" if index == len(words) - 1 else None
                yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]})}\n\n"
                time.sleep(0.01)
            if include_usage:
                usage = {"prompt_tokens": 10, "completion_tokens": len(words), "total_tokens": 10 + len(words)}
                yield f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")
    return {
        "id": "stub",
        "object": "chat.completion",
        "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22},
    }
