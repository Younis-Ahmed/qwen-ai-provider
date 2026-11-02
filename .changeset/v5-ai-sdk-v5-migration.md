---
"qwen-ai-provider": major
---

Migrate provider to AI SDK v5 (`@ai-sdk/provider` 2.x): implement `LanguageModelV2`/`EmbeddingModelV2`/`ProviderV2`, remove the legacy completion model (DashScope exposes no `/completions` endpoint), and update request/response mapping for v5 usage and stream parts. Includes reasoning content, multimodal input, tool calling, structured outputs, and embeddings support.
