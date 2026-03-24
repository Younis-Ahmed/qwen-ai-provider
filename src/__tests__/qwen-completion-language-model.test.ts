/* eslint-disable dot-notation */
import type { LanguageModelV2Prompt } from "@ai-sdk/provider"
import { convertReadableStreamToArray } from "@ai-sdk/provider-utils/test"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { QwenCompletionLanguageModel } from "../qwen-completion-language-model"
import { createQwen } from "../qwen-provider"

const TEST_PROMPT: LanguageModelV2Prompt = [
  { role: "user", content: [{ type: "text", text: "Hello" }] },
]

vi.stubEnv("DASHSCOPE_API_KEY", "test-api-key-123")

describe("config", () => {
  it("should extract base name from provider string", () => {
    const model = new QwenCompletionLanguageModel(
      "qwen-plus",
      {},
      {
        provider: "qwen.beta",
        url: () => "",
        headers: () => ({}),
      },
    )

    expect(model["providerOptionsName"]).toBe("qwen")
  })

  it("should handle provider without dot notation", () => {
    const model = new QwenCompletionLanguageModel(
      "qwen-plus",
      {},
      {
        provider: "qwen-plus",
        url: () => "",
        headers: () => ({}),
      },
    )

    expect(model["providerOptionsName"]).toBe("qwen-plus")
  })

  it("should return empty for empty provider", () => {
    const model = new QwenCompletionLanguageModel(
      "qwen-plus",
      {},
      {
        provider: "",
        url: () => "",
        headers: () => ({}),
      },
    )

    expect(model["providerOptionsName"]).toBe("")
  })
})

describe("doGenerate", () => {
  let requestBody: any
  let requestHeaders: Record<string, string>
  let responseBody: any
  let responseHeaders: Record<string, string>

  beforeEach(() => {
    requestBody = undefined
    requestHeaders = {}
    responseBody = {
      id: "cmpl-96cAM1v77r4jXa4qb2NSmRREV5oWB",
      object: "text_completion",
      created: 1711363706,
      model: "qwen-plus",
      choices: [
        {
          text: "Hello, World!",
          index: 0,
          finish_reason: "stop",
        },
      ],
      usage: {
        prompt_tokens: 4,
        total_tokens: 34,
        completion_tokens: 30,
      },
    }
    responseHeaders = {
      "content-type": "application/json",
    }
  })

  function createTestProvider(overrides?: any) {
    return createQwen({
      baseURL: "https://my.api.com/v1/",
      headers: {
        Authorization: `Bearer test-api-key`,
      },
      ...overrides,
      fetch: async (_url, init) => {
        // Capture request
        if (init?.body) {
          requestBody = JSON.parse(init.body as string)
        }
        if (init?.headers) {
          requestHeaders = init.headers as Record<string, string>
        }

        // Return response
        const bodyLength = JSON.stringify(responseBody).length
        return new Response(JSON.stringify(responseBody), {
          headers: {
            ...responseHeaders,
            "content-length": bodyLength.toString(),
          },
        })
      },
    })
  }

  it("should extract text response from content array", async () => {
    responseBody.choices[0].text = "Hello, World!"
    const provider = createTestProvider()
    const model = provider.completion("qwen-plus")

    const { content } = await model.doGenerate({
      prompt: TEST_PROMPT,
    })

    expect(content).toContainEqual({
      type: "text",
      text: "Hello, World!",
    })
  })

  it("should extract usage with V2 field names", async () => {
    responseBody.usage = {
      prompt_tokens: 20,
      total_tokens: 25,
      completion_tokens: 5,
    }
    const provider = createTestProvider()
    const model = provider.completion("qwen-plus")

    const { usage } = await model.doGenerate({
      prompt: TEST_PROMPT,
    })

    expect(usage).toMatchObject({
      inputTokens: 20,
      outputTokens: 5,
    })
  })

  it("should send request body", async () => {
    const provider = createTestProvider()
    const model = provider.completion("qwen-plus")

    await model.doGenerate({
      prompt: TEST_PROMPT,
    })

    expect(requestBody).toStrictEqual({
      model: "qwen-plus",
      prompt: "Hello",
    })
  })

  it("should send additional response information", async () => {
    responseBody = {
      ...responseBody,
      id: "test-id",
      created: 123,
      model: "test-model",
    }
    const provider = createTestProvider()
    const model = provider.completion("qwen-plus")

    const { response } = await model.doGenerate({
      prompt: TEST_PROMPT,
    })

    expect(response).toMatchObject({
      id: "test-id",
      timestamp: new Date(123 * 1000),
      modelId: "test-model",
    })
  })

  it("should extract finish reason", async () => {
    responseBody.choices[0].finish_reason = "stop"
    const provider = createTestProvider()
    const model = provider.completion("qwen-plus")

    const { finishReason } = await model.doGenerate({
      prompt: TEST_PROMPT,
    })

    expect(finishReason).toStrictEqual("stop")
  })

  it("should support unknown finish reason", async () => {
    responseBody.choices[0].finish_reason = "eos"
    const provider = createTestProvider()
    const model = provider.completion("qwen-plus")

    const { finishReason } = await model.doGenerate({
      prompt: TEST_PROMPT,
    })

    expect(finishReason).toStrictEqual("unknown")
  })

  it("should expose the raw response headers", async () => {
    responseHeaders = {
      ...responseHeaders,
      "test-header": "test-value",
    }
    const provider = createTestProvider()
    const model = provider.completion("qwen-plus")

    const { response } = await model.doGenerate({
      prompt: TEST_PROMPT,
    })

    expect(response?.headers).toMatchObject({
      "content-type": "application/json",
      "test-header": "test-value",
    })
  })

  it("should pass the model and the prompt", async () => {
    const provider = createTestProvider()
    const model = provider.completion("qwen-plus")

    await model.doGenerate({
      prompt: TEST_PROMPT,
    })

    expect(requestBody).toStrictEqual({
      model: "qwen-plus",
      prompt: "Hello",
    })
  })

  it("should pass headers", async () => {
    const provider = createTestProvider({
      headers: {
        "Authorization": `Bearer test-api-key`,
        "Custom-Provider-Header": "provider-header-value",
      },
    })

    await provider.completion("qwen-plus").doGenerate({
      prompt: TEST_PROMPT,
      headers: {
        "Custom-Request-Header": "request-header-value",
      },
    })

    expect(requestHeaders).toMatchObject({
      "authorization": "Bearer test-api-key",
      "content-type": "application/json",
      "custom-provider-header": "provider-header-value",
      "custom-request-header": "request-header-value",
    })
  })

  it("should include provider-specific options", async () => {
    const provider = createTestProvider()

    await provider.completion("qwen-plus").doGenerate({
      prompt: TEST_PROMPT,
      providerOptions: {
        qwen: {
          someCustomOption: "test-value",
        },
      },
    })

    expect(requestBody).toStrictEqual({
      model: "qwen-plus",
      prompt: "Hello",
      someCustomOption: "test-value",
    })
  })

  it("should not include provider-specific options for different provider", async () => {
    const provider = createTestProvider()

    await provider.completion("qwen-plus").doGenerate({
      prompt: TEST_PROMPT,
      providerOptions: {
        notQwen: {
          someCustomOption: "test-value",
        },
      },
    })

    expect(requestBody).toStrictEqual({
      model: "qwen-plus",
      prompt: "Hello",
    })
  })
})

describe("doStream", () => {
  let requestBody: any
  let requestHeaders: Record<string, string>
  let responseChunks: string[]
  let responseHeaders: Record<string, string>

  beforeEach(() => {
    requestBody = undefined
    requestHeaders = {}
    responseChunks = [
      `data: {"id":"cmpl-96c64EdfhOw8pjFFgVpLuT8k2MtdT","object":"text_completion","created":1711363440,"choices":[{"text":"Hello","index":0,"finish_reason":null}],"model":"qwen-plus"}\n\n`,
      `data: {"id":"cmpl-96c64EdfhOw8pjFFgVpLuT8k2MtdT","object":"text_completion","created":1711363440,"choices":[{"text":", ","index":0,"finish_reason":null}],"model":"qwen-plus"}\n\n`,
      `data: {"id":"cmpl-96c64EdfhOw8pjFFgVpLuT8k2MtdT","object":"text_completion","created":1711363440,"choices":[{"text":"World!","index":0,"finish_reason":null}],"model":"qwen-plus"}\n\n`,
      `data: {"id":"cmpl-96c3yLQE1TtZCd6n6OILVmzev8M8H","object":"text_completion","created":1711363310,"choices":[{"text":"","index":0,"finish_reason":"stop"}],"model":"qwen-plus"}\n\n`,
      `data: {"id":"cmpl-96c3yLQE1TtZCd6n6OILVmzev8M8H","object":"text_completion","created":1711363310,"model":"qwen-plus","usage":{"prompt_tokens":10,"total_tokens":372,"completion_tokens":362},"choices":[]}\n\n`,
      "data: [DONE]\n\n",
    ]
    responseHeaders = {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      "connection": "keep-alive",
    }
  })

  function createStreamingTestProvider(overrides?: any) {
    return createQwen({
      baseURL: "https://my.api.com/v1/",
      headers: {
        Authorization: `Bearer test-api-key`,
      },
      ...overrides,
      fetch: async (_url, init) => {
        // Capture request
        if (init?.body) {
          requestBody = JSON.parse(init.body as string)
        }
        if (init?.headers) {
          requestHeaders = init.headers as Record<string, string>
        }

        // Create streaming response
        const encoder = new TextEncoder()
        const stream = new ReadableStream({
          start(controller) {
            for (const chunk of responseChunks) {
              controller.enqueue(encoder.encode(chunk))
            }
            controller.close()
          },
        })

        return new Response(stream, {
          headers: responseHeaders,
        })
      },
    })
  }

  it("should stream text deltas", async () => {
    const provider = createStreamingTestProvider()
    const model = provider.completion("qwen-plus")

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
    })

    const parts = await convertReadableStreamToArray(stream)

    expect(parts).toContainEqual({
      id: "cmpl-96c64EdfhOw8pjFFgVpLuT8k2MtdT",
      modelId: "qwen-plus",
      timestamp: new Date("2024-03-25T10:44:00.000Z"),
      type: "response-metadata",
    })
    expect(
      parts.filter(p => p.type === "text-delta" && p.delta === "Hello"),
    ).toHaveLength(1)
    expect(
      parts.filter(p => p.type === "text-delta" && p.delta === ", "),
    ).toHaveLength(1)
    expect(
      parts.filter(p => p.type === "text-delta" && p.delta === "World!"),
    ).toHaveLength(1)
    expect(parts).toContainEqual({
      type: "finish",
      finishReason: "stop",
      usage: { inputTokens: 10, outputTokens: 362, totalTokens: 372 },
    })
  })

  it("should handle unparsable stream parts", async () => {
    responseChunks = [`data: {unparsable}\n\n`, "data: [DONE]\n\n"]
    const provider = createStreamingTestProvider()
    const model = provider.completion("qwen-plus")

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
    })

    const elements = await convertReadableStreamToArray(stream)

    expect(elements.length).toBeGreaterThanOrEqual(2)
    expect(elements.find(e => e.type === "error")).toBeDefined()
    expect(elements.find(e => e.type === "finish")).toMatchObject({
      finishReason: "unknown",
      type: "finish",
    })
  })

  it("should send request body with stream flag", async () => {
    const provider = createStreamingTestProvider()
    const model = provider.completion("qwen-plus")

    await model.doStream({
      prompt: TEST_PROMPT,
    })

    expect(requestBody).toStrictEqual({
      model: "qwen-plus",
      prompt: "Hello",
      stream: true,
    })
  })

  it("should expose the raw response headers", async () => {
    responseHeaders = {
      ...responseHeaders,
      "test-header": "test-value",
    }
    const provider = createStreamingTestProvider()
    const model = provider.completion("qwen-plus")

    const { response } = await model.doStream({
      prompt: TEST_PROMPT,
    })

    expect(response?.headers).toMatchObject({
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      "connection": "keep-alive",
      "test-header": "test-value",
    })
  })

  it("should pass the model and the prompt", async () => {
    const provider = createStreamingTestProvider()
    const model = provider.completion("qwen-plus")

    await model.doStream({
      prompt: TEST_PROMPT,
    })

    expect(requestBody).toStrictEqual({
      stream: true,
      model: "qwen-plus",
      prompt: "Hello",
    })
  })

  it("should pass headers", async () => {
    const provider = createStreamingTestProvider({
      headers: {
        "Authorization": `Bearer test-api-key`,
        "Custom-Provider-Header": "provider-header-value",
      },
    })

    await provider.completion("qwen-plus").doStream({
      prompt: TEST_PROMPT,
      headers: {
        "Custom-Request-Header": "request-header-value",
      },
    })

    expect(requestHeaders).toMatchObject({
      "authorization": "Bearer test-api-key",
      "content-type": "application/json",
      "custom-provider-header": "provider-header-value",
      "custom-request-header": "request-header-value",
    })
  })

  it("should include provider-specific options", async () => {
    const provider = createStreamingTestProvider()

    await provider.completion("qwen-plus").doStream({
      prompt: TEST_PROMPT,
      providerOptions: {
        qwen: {
          someCustomOption: "test-value",
        },
      },
    })

    expect(requestBody).toStrictEqual({
      stream: true,
      model: "qwen-plus",
      prompt: "Hello",
      someCustomOption: "test-value",
    })
  })

  it("should not include provider-specific options for different provider", async () => {
    const provider = createStreamingTestProvider()

    await provider.completion("qwen-plus").doStream({
      prompt: TEST_PROMPT,
      providerOptions: {
        notQwen: {
          someCustomOption: "test-value",
        },
      },
    })

    expect(requestBody).toStrictEqual({
      stream: true,
      model: "qwen-plus",
      prompt: "Hello",
    })
  })
})
