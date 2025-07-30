# qwen-ai-provider

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
