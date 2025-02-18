import { loadApiKey } from "@ai-sdk/provider-utils"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { QwenChatLanguageModel } from "./qwen-chat-language-model"
import { QwenCompletionLanguageModel } from "./qwen-completion-language-model"
import { QwenEmbeddingModel } from "./qwen-embedding-model"
import { createQwen } from "./qwen-provider"

vi.stubEnv("DASHSCOPE_API_KEY", "test-api-key-123")

// Mock the model classes
vi.mock("./qwen-chat-language-model", () => ({
  QwenChatLanguageModel: vi.fn(),
}))

vi.mock("./qwen-completion-language-model", () => ({
  QwenCompletionLanguageModel: vi.fn(),
}))

vi.mock("./qwen-embedding-model", () => ({
  QwenEmbeddingModel: vi.fn(),
}))

vi.mock("@ai-sdk/provider-utils", () => ({
  loadApiKey: vi.fn().mockReturnValue("test-api-key-123"),
  withoutTrailingSlash: vi.fn((url: string) => url),
  createJsonErrorResponseHandler: vi.fn().mockImplementation((options) => {
    return async (response: Response) => {
      const error = await response.json()
      return {
        message: options.errorToMessage(error),
        cause: error,
      }
    }
  }),
}))

describe("qwenProvider", () => {
  let provider: ReturnType<typeof createQwen>

  beforeEach(() => {
    vi.clearAllMocks()
    provider = createQwen()
  })

  describe("createQwen", () => {
    it("should set API key using loadApiKey with default options", () => {
      // Create a model with a provider
      provider("test-model", {})
      // Get the constructor call for the Chat Language Model
      const constructorCall = vi.mocked(QwenChatLanguageModel).mock.calls[0]
      const config = constructorCall[2]
      // Invoke headers if needed by the configuration
      config.headers && config.headers()

      expect(loadApiKey).toHaveBeenCalledWith({
        apiKey: undefined,
        environmentVariableName: "DASHSCOPE_API_KEY",
        description: "Qwen API key",
      })
    })

    it("should create a chat model when called as a function", () => {
      provider("chat-model", { user: "test-user" })
      expect(QwenChatLanguageModel).toHaveBeenCalled()
    })
  })

  describe("chatModel", () => {
    it("should construct a chat model with correct configuration", () => {
      const settings = { user: "foo-user" }
      provider.chatModel("qwen-chat-model", settings)
      expect(QwenChatLanguageModel).toHaveBeenCalledWith(
        "qwen-chat-model",
        settings,
        expect.objectContaining({
          provider: "qwen.chat",
          defaultObjectGenerationMode: "tool",
        }),
      )
    })
  })

  describe("completion", () => {
    it("should construct a completion model with correct configuration", () => {
      const settings = { user: "foo-user" }
      provider.completion("qwen-turbo", settings)
      expect(QwenCompletionLanguageModel).toHaveBeenCalledWith(
        "qwen-turbo",
        settings,
        expect.objectContaining({
          provider: "qwen.completion",
        }),
      )
    })
  })

  describe("textEmbeddingModel", () => {
    it("should construct a text embedding model with correct configuration", () => {
      const settings = { user: "foo-user" }
      provider.textEmbeddingModel("qwen-vl-plus", settings)
      expect(QwenEmbeddingModel).toHaveBeenCalledWith(
        "qwen-vl-plus",
        settings,
        expect.objectContaining({
          provider: "qwen.embedding",
        }),
      )
    })
  })

  describe("languageModel alias", () => {
    it("should return a chat model when called via languageModel", () => {
      provider.languageModel("qwen-chat-model", { user: "alias" })
      expect(QwenChatLanguageModel).toHaveBeenCalled()
    })
  })
})
