import {
  OpenAICompatibleChatLanguageModel,
  OpenAICompatibleCompletionLanguageModel,
  OpenAICompatibleEmbeddingModel,
} from "@ai-sdk/openai-compatible"
import { loadApiKey } from "@ai-sdk/provider-utils"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { createQwen } from "./qwen-provider"

// Mocks
vi.mock("@ai-sdk/openai-compatible", () => ({
  OpenAICompatibleChatLanguageModel: vi.fn(),
  OpenAICompatibleCompletionLanguageModel: vi.fn(),
  OpenAICompatibleEmbeddingModel: vi.fn(),
}))

vi.mock("@ai-sdk/provider-utils", () => ({
  loadApiKey: vi.fn().mockReturnValue("mock-api-key"),
  withoutTrailingSlash: vi.fn((url: string) => url),
}))

describe("qwenProvider", () => {
  let provider: ReturnType<typeof createQwen>

  beforeEach(() => {
    vi.clearAllMocks()
    provider = createQwen()
  })

  describe("createQwen", () => {
    it("should set API key using loadApiKey with default options", () => {
      provider("test-model", {})
      const call = vi.mocked(loadApiKey).mock.calls[0][0]
      expect(call).toEqual({
        apiKey: undefined,
        environmentVariableName: "DASHSCOPE_API_KEY",
        description: "Qwen API key",
      })
    })

    it("should create a chat model when called as a function", () => {
      provider("chat-model", { user: "test-user" })
      expect(OpenAICompatibleChatLanguageModel).toHaveBeenCalled()
    })
  })

  describe("chatModel", () => {
    it("should construct a chat model with correct configuration", () => {
      const settings = { user: "foo-user" }
      provider.chatModel("qwen-chat-model", settings)
      expect(OpenAICompatibleChatLanguageModel).toHaveBeenCalledWith(
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
      expect(OpenAICompatibleCompletionLanguageModel).toHaveBeenCalledWith(
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
      expect(OpenAICompatibleEmbeddingModel).toHaveBeenCalledWith(
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
      expect(OpenAICompatibleChatLanguageModel).toHaveBeenCalled()
    })
  })
})
