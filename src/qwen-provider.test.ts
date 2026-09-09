import { describe, expect, it, vi } from "vitest"
import { createQwen } from "./qwen-provider"

vi.stubEnv("DASHSCOPE_API_KEY", "test-api-key-123")

describe("qwenProvider", () => {
  describe("registry", () => {
    const chatFactory = vi.fn()
    const completionFactory = vi.fn()
    const embeddingFactory = vi.fn()
    const imageFactory = vi.fn()

    const provider = createQwen({}, {
      chat: chatFactory as any,
      completion: completionFactory as any,
      embedding: embeddingFactory as any,
      image: imageFactory as any,
    })

    it("should create a chat model when called as a function", () => {
      provider("chat-model", { user: "test-user" })

      expect(chatFactory).toHaveBeenCalledWith(
        "chat-model",
        { user: "test-user" },
        expect.objectContaining({
          provider: "qwen.chat",
        }),
      )
    })

    it("should route the default provider call to the chat registry entry", () => {
      provider("chat-model")

      expect(chatFactory).toHaveBeenCalled()
    })

    it("should construct a chat model via chatModel", () => {
      const settings = { user: "foo-user" }
      provider.chatModel("qwen-chat-model", settings)

      expect(chatFactory).toHaveBeenCalledWith(
        "qwen-chat-model",
        settings,
        expect.objectContaining({
          provider: "qwen.chat",
        }),
      )
    })

    it("should construct a completion model via completion", () => {
      const settings = { user: "foo-user" }
      provider.completion("qwen-turbo", settings)

      expect(completionFactory).toHaveBeenCalledWith(
        "qwen-turbo",
        settings,
        expect.objectContaining({
          provider: "qwen.completion",
        }),
      )
    })

    it("should construct a text embedding model via textEmbeddingModel", () => {
      const settings = { user: "foo-user" }
      provider.textEmbeddingModel("text-embedding-v3", settings)

      expect(embeddingFactory).toHaveBeenCalledWith(
        "text-embedding-v3",
        settings,
        expect.objectContaining({
          provider: "qwen.embedding",
        }),
      )
    })

    it("should return a chat model when called via languageModel", () => {
      provider.languageModel("qwen-chat-model", { user: "alias" })

      expect(chatFactory).toHaveBeenCalled()
    })

    it("should return the image factory from the registry", () => {
      provider.imageModel("qwen-vl-max")

      expect(imageFactory).toHaveBeenCalled()
    })
  })

  describe("defaults", () => {
    it("should use the DASHSCOPE_API_KEY environment variable for the Authorization header", () => {
      const provider = createQwen()
      const chat = provider.chatModel("qwen-plus")

      expect(chat.provider).toBe("qwen.chat")
      expect(chat.modelId).toBe("qwen-plus")
    })

    it("should throw a NoSuchModelError when no image registry entry is supplied", () => {
      const provider = createQwen()

      expect(() => provider.imageModel("qwen-vl-max")).toThrow()
    })

    it("should allow overriding the image model through the registry", () => {
      const imageModel = { provider: "qwen", modelId: "qwen-vl-max" }
      const provider = createQwen({}, {
        image: () => imageModel as any,
      })

      expect(provider.imageModel("qwen-vl-max")).toBe(imageModel)
    })

    it("should build URLs from the base URL and path", () => {
      const provider = createQwen({ baseURL: "https://my.api.com/v1/" })
      const chat = provider.chatModel("qwen-plus") as any

      expect(chat.config?.url({ path: "/chat/completions", modelId: "qwen-plus" })).toBe(
        "https://my.api.com/v1/chat/completions",
      )
    })
  })
})
