# qwen-ai-provider-v5

## 2.1.3

### Patch Changes

- 177d8f1: Improve Qwen 3.5 compatibility and reorganize the provider internals without changing the main public API.

  This release also removes noisy stream debug logging and keeps the local rerank script aligned with the refactored source layout.

## 2.1.2

### Patch Changes

- dcc41cf: Fix reasoning content parts in assistant messages breaking multi-turn conversations (fixes [#8](https://github.com/bolechen/qwen-ai-provider-v5/issues/8))
  - Silently ignore `reasoning` and `file` parts in assistant messages instead of throwing `UnsupportedFunctionalityError`
  - Models like GLM-4.7 return reasoning parts that were causing dialogue interruption when passed back in subsequent API calls

## 2.1.1

### Patch Changes

- 1bc9fbc: Stabilize streaming tool-calls and reasoning
  - Add retry logic for Qwen streaming API transient 500 errors (`list index out of range`)
  - Support `reasoning` field alongside `reasoning_content` for compatibility
  - Fix incomplete tool-call assembly during streaming by buffering and reassembling fragmented chunks
  - Improve in-stream error detection and recovery for known Qwen boundary failures

## 2.1.0

### Minor Changes

- **Reranking API Improvements** - Support both `qwen3-rerank` and `gte-rerank` models
  - `qwen3-rerank`: Uses OpenAI-compatible API (`/compatible-api/v1/reranks`)
  - `gte-rerank-v2`: Uses DashScope native API (`/api/v1/services/rerank/...`)
  - Add `instruct` setting for task-specific ranking (qwen3-rerank only)
  - Update model capabilities documentation

### Patch Changes

- Fix endpoint for qwen3-rerank from `/compatible-mode/v1/rerank` to `/compatible-api/v1/reranks`

## 2.0.2

### Patch Changes

- Fix reranking API endpoint and request format
  - Use DashScope native API endpoint `/api/v1/services/rerank/text-rerank/text-rerank` instead of OpenAI compatible mode
  - Update request body format to `{ model, input: { query, documents }, parameters: { top_n } }`
  - Update response parsing for `{ output: { results: [...] }, request_id }` format
  - Add documentation for Chinese users using domestic endpoint

## 2.0.1

### Patch Changes

- 7f8dcd2: ### Improvements
  - **CI/CD**: Configure automated npm publishing via GitHub Actions
    - Add Changesets-based release workflow for automatic versioning
    - Trigger releases automatically when changes are merged to main
    - Simplify pre-release workflow for dry-run testing
  - **Reranking Model**: Add support for Qwen reranking models
    - New `QwenRerankingModel` for document reranking tasks
    - Support for `gte-rerank` model family

## 2.0.0

### Major Changes

- **🚀 AI SDK v6 Support** - Upgrade to Vercel AI SDK v6 (specification v3)
  - Updated to `@ai-sdk/provider@^3.0.0` (was ^2.0.0)
  - Updated to `@ai-sdk/provider-utils@^4.0.0` (was ^3.0.0)
  - Migrated `QwenChatLanguageModel` to `LanguageModelV3` interface
  - Migrated `QwenCompletionLanguageModel` to `LanguageModelV3` interface
  - Migrated `QwenEmbeddingModel` to `EmbeddingModelV3` interface

### Breaking Changes

- **Minimum AI SDK Version**: Now requires AI SDK v6 (`ai@^6.0.0`)
- **FinishReason Format**: Changed from string to object `{ unified: string, raw: string | undefined }`
- **Usage Format**: Changed to nested structure with `inputTokens: { total, noCache, cacheRead, cacheWrite }` and `outputTokens: { total, text, reasoning }`
- **Warnings Format**: Changed from `type: "unsupported-setting"` to `type: "unsupported"` with `feature` property

### Migration

If you are using AI SDK v5, please continue using `qwen-ai-provider-v5@^1.0.0`.

---

## 1.0.2

### Patch Changes

- **Tool Call Compatibility**:
  - Fixed `QwenChatLanguageModel` to emit `tool-call` stream parts with string `input` payloads, matching the AI SDK v5 `LanguageModelV2ToolCall` contract
  - Ensured non-streaming `doGenerate` also returns `tool-call` content with stringified JSON `input`
  - This resolves `toolCall.input.trim is not a function` errors when using `qwen-ai-provider-v5` together with ai-sdk v5 tool execution

- **Testing**:
  - Updated chat model tests to assert string `input` for tool calls in both streaming and non-streaming modes

## 1.0.1

### Patch Changes

- **Package Metadata**: Updated package information for npm publication
  - Changed package name to `qwen-ai-provider-v5`
  - Updated repository URL to https://github.com/bolechen/qwen-ai-provider
  - Updated author information

- **Documentation**: Enhanced README with AI SDK v5 focus
  - Added prominent notice about AI SDK v5 requirement
  - Linked to original `qwen-ai-provider` package for v4 users
  - Updated all code examples to use `qwen-ai-provider-v5` imports
  - Added acknowledgment to original author

## 1.0.0

### Major Changes

- **🎉 AI SDK v5 Migration** - Complete migration to Vercel AI SDK v5 (specification v2)
  - Migrated `QwenChatLanguageModel` to `LanguageModelV2` interface
  - Migrated `QwenCompletionLanguageModel` to `LanguageModelV2` interface
  - Migrated `QwenEmbeddingModel` to `EmbeddingModelV2` interface
  - Updated `QwenProvider` interface to v2 specifications

- **Breaking Changes**:
  - Updated to `@ai-sdk/provider@^2.0.0` (was ^1.0.7)
  - Updated to `@ai-sdk/provider-utils@^3.0.0` (was ^2.1.6)
  - Changed response format from `{text, toolCalls}` to `{content: []}` array
  - Usage fields now use `inputTokens`/`outputTokens` instead of `promptTokens`/`completionTokens`
  - Parameters changed: `maxOutputTokens` instead of `maxTokens`, `providerOptions` instead of `providerMetadata`
  - Removed `mode` parameter; tools and toolChoice are now direct parameters
  - Stream parts updated to V2 format with `text-start`/`text-delta`/`text-end` pattern

### Minor Changes

- **Enhanced Tool Handling**:
  - Improved tool call error handling and validation
  - Better tool response conversion in chat messages
  - Enhanced tool streaming support with proper delta handling

- **Improved Error Handling**:
  - Better error messages for QwenChat and QwenCompletion models
  - Refined system message content part conversion
  - Improved handling of unsupported content types

### Patch Changes

- **Testing**:
  - Reorganized test files into `__tests__` directory
  - Migrated all 95+ tests to V2 API format
  - Implemented custom fetch mocking pattern to avoid DataCloneError
  - Added MSW (Mock Service Worker) dependency for testing
  - Complete test coverage maintained: chat model (44 tests), completion model (22 tests), embedding model (6 tests)

- **Zod Compatibility**:
  - Added support for Zod v4 alongside v3
  - Peer dependency: `zod@^3.25.76 || ^4.1.8`
  - Updated Zod response schemas

- **Code Quality**:
  - Added semicolons to all TypeScript files for consistency
  - Tightened peer dependency ranges
  - Added prepack script to ensure built artifacts

- **Build**:
  - Verified TypeScript compilation with strict mode
  - Dual format output: CommonJS and ESM
  - Source maps and type declarations included

---

## Previous Versions (qwen-ai-provider)

The following versions are from the original `qwen-ai-provider` package for AI SDK v4:

## 0.1.1

### Patch Changes

- 06feb5c: - Fixed the onFinish hook token usage response
  - Updated the zod response scheme
  - Handle unsupported content parts in assistant messages

## 0.1.0

### Minor Changes

- This release introduces several improvements and new features across the project:
  - Added support for chat, completion, and text embedding models with the introduction of `QwenChatLanguageModel`, `QwenCompletionLanguageModel`, and `QwenEmbeddingModel`.
  - Enabled the provider to construct and configure these models dynamically based on request parameters.
  - Integrated provider utilities to manage API key loading.
  - For testing, the `DASHSCOPE_API_KEY` environment variable is mocked to ensure tests run reliably without exposing real credentials.
  - Introduced a JSON error response handler (`createJsonErrorResponseHandler`) to streamline error processing.
  - Improved error messaging and handling in the provider modules, including detailed error responses consistent with API validation.
  - Expanded test coverage for all model classes to cover various configurations and edge cases.
  - Applied comprehensive mocking of relevant modules (including provider utilities and language model classes) to ensure tests run in isolation.
  - Added the ability to override environment variables in tests using `vi.stubEnv`.
  - Updated GitHub Actions workflows for continuous integration and publishing.
  - Configured a dedicated GitHub workflow for releases using changesets.
  - Enhanced npm publishing automation with proper authentication setup (using `NPM_TOKEN`).
  - Fixed issues where missing or misconfigured API keys were causing test failures.
  - Resolved errors related to unauthorized npm publishing by enforcing proper npm authentication.
  - Addressed several configuration inconsistencies in the test harness that led to unexpected behavior across language model tests.
  - Updated changelogs automatically via changesets.
  - The release is fully documented with comprehensive commit logs and automated changelog generation to track updates in functionality and configuration.

  Enjoy the new features and improvements! 🎉

## 0.0.2

### Patch Changes

- Initial release of qwen-ai-provider:
  - Added QwenChatLanguageModel for chat completions
  - Added QwenCompletionLanguageModel for text completions
  - Added QwenEmbeddingModel for text embeddings
  - Added provider utilities and error handling
  - Added comprehensive test coverage
  - Added GitHub Actions workflows for CI/CD
  - Added npm package configuration
