"""
Mu OpenAI API Server
====================

FastAPI server providing OpenAI-compatible REST API.

Endpoints:
- POST /v1/completions - Text completion
- POST /v1/chat/completions - Chat completion
- GET /v1/models - List models
- GET /health - Health check
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Optional, List, Dict, Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from mu_inference.core.config import SamplingParams, EngineConfig
from mu_inference.serving.engine import MuEngine

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models (OpenAI format)
# ============================================================================

class CompletionRequest(BaseModel):
    """OpenAI Completion API request."""
    model: str
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    n: int = Field(default=1, ge=1, le=1)  # Only n=1 supported
    stream: bool = False
    stop: Optional[List[str]] = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=2.0)
    user: Optional[str] = None


class ChatMessage(BaseModel):
    """Chat message."""
    role: str  # "system", "user", "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI Chat Completion API request."""
    model: str
    messages: List[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    n: int = Field(default=1, ge=1, le=1)
    stream: bool = False
    stop: Optional[List[str]] = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=2.0)
    user: Optional[str] = None


class CompletionChoice(BaseModel):
    """Completion choice."""
    index: int
    text: str
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    """Chat completion choice."""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class UsageInfo(BaseModel):
    """Token usage info."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    """OpenAI Completion API response."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo


class ChatCompletionResponse(BaseModel):
    """OpenAI Chat Completion API response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class ModelInfo(BaseModel):
    """Model info."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "mu-inference"


class ModelList(BaseModel):
    """List of models."""
    object: str = "list"
    data: List[ModelInfo]


# ============================================================================
# Server
# ============================================================================

class MuServer:
    """
    OpenAI-compatible API server for Mu inference.
    """

    def __init__(
        self,
        engine: MuEngine,
        model_name: str = "mu-model",
    ):
        self.engine = engine
        self.model_name = model_name

        # Create FastAPI app
        self.app = FastAPI(
            title="Mu Inference API",
            description="OpenAI-compatible API powered by Mu dynamics",
            version="1.0.0",
        )

        # Add CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register routes
        self._register_routes()

    def _register_routes(self):
        """Register API routes."""

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy", "model": self.model_name}

        @self.app.get("/v1/models")
        async def list_models():
            """List available models."""
            return ModelList(
                data=[
                    ModelInfo(
                        id=self.model_name,
                        created=int(time.time()),
                    )
                ]
            )

        @self.app.get("/v1/models/{model_id}")
        async def get_model(model_id: str):
            """Get model info."""
            if model_id != self.model_name:
                raise HTTPException(status_code=404, detail="Model not found")
            return ModelInfo(
                id=self.model_name,
                created=int(time.time()),
            )

        @self.app.post("/v1/completions")
        async def create_completion(request: CompletionRequest):
            """Create text completion."""
            return await self._handle_completion(request)

        @self.app.post("/v1/chat/completions")
        async def create_chat_completion(request: ChatCompletionRequest):
            """Create chat completion."""
            return await self._handle_chat_completion(request)

        @self.app.get("/stats")
        async def get_stats():
            """Get engine statistics."""
            return self.engine.get_stats()

    async def _handle_completion(
        self,
        request: CompletionRequest,
    ) -> CompletionResponse:
        """Handle completion request."""
        request_id = f"cmpl-{uuid.uuid4().hex[:24]}"

        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
        )

        if request.stream:
            return StreamingResponse(
                self._stream_completion(request, request_id, sampling_params),
                media_type="text/event-stream",
            )

        # Non-streaming
        output = await self.engine.generate(
            prompt=request.prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        )

        return CompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=self.model_name,
            choices=[
                CompletionChoice(
                    index=0,
                    text=output.text,
                    finish_reason=output.finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=output.usage["prompt_tokens"],
                completion_tokens=output.usage["completion_tokens"],
                total_tokens=output.usage["total_tokens"],
            ),
        )

    async def _stream_completion(
        self,
        request: CompletionRequest,
        request_id: str,
        sampling_params: SamplingParams,
    ) -> AsyncIterator[str]:
        """Stream completion response."""
        async for chunk in self.engine.generate_stream(
            prompt=request.prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            data = {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "text": chunk.text,
                        "finish_reason": chunk.finish_reason,
                    }
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"

            if chunk.finished:
                yield "data: [DONE]\n\n"
                break

    async def _handle_chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Handle chat completion request."""
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        # Format messages into prompt
        prompt = self._format_chat_prompt(request.messages)

        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
        )

        if request.stream:
            return StreamingResponse(
                self._stream_chat_completion(request, request_id, prompt, sampling_params),
                media_type="text/event-stream",
            )

        # Non-streaming
        output = await self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        )

        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=self.model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=output.text,
                    ),
                    finish_reason=output.finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=output.usage["prompt_tokens"],
                completion_tokens=output.usage["completion_tokens"],
                total_tokens=output.usage["total_tokens"],
            ),
        )

    async def _stream_chat_completion(
        self,
        request: ChatCompletionRequest,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> AsyncIterator[str]:
        """Stream chat completion response."""
        async for chunk in self.engine.generate_stream(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": chunk.text,
                        } if not chunk.finished else {},
                        "finish_reason": chunk.finish_reason,
                    }
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"

            if chunk.finished:
                yield "data: [DONE]\n\n"
                break

    def _format_chat_prompt(self, messages: List[ChatMessage]) -> str:
        """
        Format chat messages into a prompt string.

        Uses simple format. Override for model-specific templates.
        """
        prompt_parts = []

        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}\n")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}\n")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}\n")

        # Add prompt for assistant response
        prompt_parts.append("Assistant:")

        return "".join(prompt_parts)


def create_app(
    model_path: str,
    config: Optional[EngineConfig] = None,
    model_name: Optional[str] = None,
) -> FastAPI:
    """
    Create FastAPI app with Mu engine.

    Args:
        model_path: Path to model
        config: Engine configuration
        model_name: Model name for API

    Returns:
        FastAPI app
    """
    config = config or EngineConfig()

    # Create engine
    engine = MuEngine(config=config, model_path=model_path)

    # Derive model name from path if not provided
    if model_name is None:
        model_name = model_path.split("/")[-1]

    # Create server
    server = MuServer(engine=engine, model_name=model_name)

    # Add startup event to initialize engine
    @server.app.on_event("startup")
    async def startup():
        await engine.initialize(model_path)

    @server.app.on_event("shutdown")
    async def shutdown():
        await engine.shutdown()

    return server.app
