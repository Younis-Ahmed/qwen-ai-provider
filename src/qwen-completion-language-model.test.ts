/* eslint-disable dot-notation */
import type { LanguageModelV1Prompt } from "@ai-sdk/provider"
import {
  convertReadableStreamToArray,
  JsonTestServer,
  StreamingTestServer,
} from "@ai-sdk/provider-utils/test"
import { describe, expect, it } from "vitest"
import { QwenChatLanguageModel } from "./qwen-chat-language-model"
import { createQwen } from "./qwen-provider"

const TEST_PROMPT: LanguageModelV1Prompt = [
  { role: "user", content: [{ type: "text", text: "Hello" }] },
]

const provider = createQwen({
  baseURL: "https://my.api.com/v1/",
  headers: {
    Authorization: `Bearer test-api-key`,
  },
})

const model = provider.completion("qwen-plus")

describe("config", () => {
  it("should extract base name from provider string", () => {
    const model = new QwenChatLanguageModel(
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
    const model = new QwenChatLanguageModel(
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
    const model = new QwenChatLanguageModel(
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
  const server = new JsonTestServer("https://my.api.com/v1/completions")

  server.setupTestEnvironment()

  function prepareJsonResponse({
    content = "",
    usage = {
      prompt_tokens: 4,
      total_tokens: 34,
      completion_tokens: 30,
    },
    finish_reason = "stop",
    id = "cmpl-96cAM1v77r4jXa4qb2NSmRREV5oWB",
    created = 1711363706,
    model = "qwen-plus",
  }: {
    content?: string
    usage?: {
      prompt_tokens: number
      total_tokens: number
      completion_tokens: number
    }
    finish_reason?: string
    id?: string
    created?: number
    model?: string
  }) {
    server.responseBodyJson = {
      id,
      object: "text_completion",
      created,
      model,
      choices: [
        {
          text: content,
          index: 0,
          finish_reason,
        },
      ],
      usage,
    }
  }

  it("should extract text response", async () => {
    prepareJsonResponse({ content: "Hello, World!" })

    const { text } = await model.doGenerate({
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
    })

    expect(text).toStrictEqual("Hello, World!")
  })

  it("should extract usage", async () => {
    prepareJsonResponse({
      content: "",
      usage: { prompt_tokens: 20, total_tokens: 25, completion_tokens: 5 },
    })

    const { usage } = await model.doGenerate({
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
    })

    expect(usage).toStrictEqual({
      promptTokens: 20,
      completionTokens: 5,
    })
  })

  it("should send request body", async () => {
    prepareJsonResponse({})

    const { request } = await model.doGenerate({
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
    })

    expect(request).toStrictEqual({
      body: "{\"model\":\"qwen-plus\",\"prompt\":\"Hello\"}",
    })
  })

  it("should send additional response information", async () => {
    prepareJsonResponse({
      id: "test-id",
      created: 123,
      model: "test-model",
    })

    const { response } = await model.doGenerate({
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
    })

    expect(response).toStrictEqual({
      id: "test-id",
      timestamp: new Date(123 * 1000),
      modelId: "test-model",
    })
  })

  it("should extract finish reason", async () => {
    prepareJsonResponse({
      content: "",
      finish_reason: "stop",
    })

    const { finishReason } = await provider
      .completion("qwen-plus")
      .doGenerate({
        inputFormat: "prompt",
        mode: { type: "regular" },
        prompt: TEST_PROMPT,
      })

    expect(finishReason).toStrictEqual("stop")
  })

  it("should support unknown finish reason", async () => {
    prepareJsonResponse({
      content: "",
      finish_reason: "eos",
    })

    const { finishReason } = await provider
      .completion("qwen-plus")
      .doGenerate({
        inputFormat: "prompt",
        mode: { type: "regular" },
        prompt: TEST_PROMPT,
      })

    expect(finishReason).toStrictEqual("unknown")
  })

  it("should expose the raw response headers", async () => {
    prepareJsonResponse({ content: "" })

    server.responseHeaders = {
      "test-header": "test-value",
    }

    const { rawResponse } = await model.doGenerate({
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
    })

    expect(rawResponse?.headers).toStrictEqual({
      // default headers:
      "content-length": "237",
      "content-type": "application/json",

      // custom header
      "test-header": "test-value",
    })
  })

  it("should pass the model and the prompt", async () => {
    prepareJsonResponse({ content: "" })

    await model.doGenerate({
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
    })

    expect(await server.getRequestBodyJson()).toStrictEqual({
      model: "qwen-plus",
      prompt: "Hello",
    })
  })

  it("should pass headers", async () => {
    prepareJsonResponse({ content: "" })

    const provider = createQwen({
      baseURL: "https://my.api.com/v1/",
      headers: {
        "Authorization": `Bearer test-api-key`,
        "Custom-Provider-Header": "provider-header-value",
      },
    })

    await provider.completion("qwen-plus").doGenerate({
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
      headers: {
        "Custom-Request-Header": "request-header-value",
      },
    })

    const requestHeaders = await server.getRequestHeaders()

    expect(requestHeaders).toStrictEqual({
      "authorization": "Bearer test-api-key",
      "content-type": "application/json",
      "custom-provider-header": "provider-header-value",
      "custom-request-header": "request-header-value",
    })
  })

  it("should include provider-specific options", async () => {
    prepareJsonResponse({ content: "" })

    await provider.completion("qwen-plus").doGenerate({
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
      providerMetadata: {
        "test-provider": {
          someCustomOption: "test-value",
        },
      },
    })

    expect(await server.getRequestBodyJson()).toStrictEqual({
      model: "qwen-plus",
      prompt: "Hello",
    })
  })

  it("should not include provider-specific options for different provider", async () => {
    prepareJsonResponse({ content: "" })

    await provider.completion("qwen-plus").doGenerate({
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
      providerMetadata: {
        notThisProviderName: {
          someCustomOption: "test-value",
        },
      },
    })

    expect(await server.getRequestBodyJson()).toStrictEqual({
      model: "qwen-plus",
      prompt: "Hello",
    })
  })
})

describe("doStream", () => {
  const server = new StreamingTestServer("https://my.api.com/v1/completions")

  server.setupTestEnvironment()

  function prepareStreamResponse({
    content,
    finish_reason = "stop",
    usage = {
      prompt_tokens: 10,
      total_tokens: 372,
      completion_tokens: 362,
    },
  }: {
    content: string[]
    usage?: {
      prompt_tokens: number
      total_tokens: number
      completion_tokens: number
    }
    finish_reason?: string
  }) {
    server.responseChunks = [
      ...content.map((text) => {
        return (
          `data: {"id":"cmpl-96c64EdfhOw8pjFFgVpLuT8k2MtdT","object":"text_completion","created":1711363440,`
          + `"choices":[{"text":"${text}","index":0,"finish_reason":null}],"model":"qwen-plus"}\n\n`
        )
      }),
      `data: {"id":"cmpl-96c3yLQE1TtZCd6n6OILVmzev8M8H","object":"text_completion","created":1711363310,`
      + `"choices":[{"text":"","index":0,"finish_reason":"${finish_reason}"}],"model":"qwen-plus"}\n\n`,
      `data: {"id":"cmpl-96c3yLQE1TtZCd6n6OILVmzev8M8H","object":"text_completion","created":1711363310,`
      + `"model":"qwen-plus","usage":${JSON.stringify(
        usage,
      )},"choices":[]}\n\n`,
      "data: [DONE]\n\n",
    ]
  }

  it("should stream text deltas", async () => {
    prepareStreamResponse({
      content: ["Hello", ", ", "World!"],
      finish_reason: "stop",
      usage: {
        prompt_tokens: 10,
        total_tokens: 372,
        completion_tokens: 362,
      },
    })

    const { stream } = await model.doStream({
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
    })

    // note: space moved to last chunk bc of trimming
    expect(await convertReadableStreamToArray(stream)).toStrictEqual([
      {
        id: "cmpl-96c64EdfhOw8pjFFgVpLuT8k2MtdT",
        modelId: "qwen-plus",
        timestamp: new Date("2024-03-25T10:44:00.000Z"),
        type: "response-metadata",
      },
      { type: "text-delta", textDelta: "Hello" },
      { type: "text-delta", textDelta: ", " },
      { type: "text-delta", textDelta: "World!" },
      { type: "text-delta", textDelta: "" },
      {
        type: "finish",
        finishReason: "stop",
        usage: { promptTokens: 10, completionTokens: 362 },
      },
    ])
  })

  it("should handle unparsable stream parts", async () => {
    server.responseChunks = [`data: {unparsable}\n\n`, "data: [DONE]\n\n"]

    const { stream } = await model.doStream({
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
    })

    const elements = await convertReadableStreamToArray(stream)

    expect(elements.length).toBe(2)
    expect(elements[0].type).toBe("error")
    expect(elements[1]).toStrictEqual({
      finishReason: "error",
      type: "finish",
      usage: {
        completionTokens: Number.NaN,
        promptTokens: Number.NaN,
      },
    })
  })

  it("should send request body", async () => {
    prepareStreamResponse({ content: [] })

    const { request: _request } = await model.doStream({
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
    })

    expect(_request).toStrictEqual({
      // body: '{"model":"qwen-plus","prompt":"Hello","stream":true,"stream_options":{"include_usage":true}}',
      body: "{\"model\":\"qwen-plus\",\"prompt\":\"Hello\",\"stream\":true}",
    })
  })

  it("should expose the raw response headers", async () => {
    prepareStreamResponse({ content: [] })

    server.responseHeaders = {
      "test-header": "test-value",
    }

    const { rawResponse } = await model.doStream({
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
    })

    expect(rawResponse?.headers).toStrictEqual({
      // default headers:
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      "connection": "keep-alive",

      // custom header
      "test-header": "test-value",
    })
  })

  it("should pass the model and the prompt", async () => {
    prepareStreamResponse({ content: [] })

    await model.doStream({
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
    })

    expect(await server.getRequestBodyJson()).toStrictEqual({
      stream: true,
      // stream_options: { include_usage: true },
      model: "qwen-plus",
      prompt: "Hello",
    })
  })

  it("should pass headers", async () => {
    prepareStreamResponse({ content: [] })

    const provider = createQwen({
      baseURL: "https://my.api.com/v1/",
      headers: {
        "Authorization": `Bearer test-api-key`,
        "Custom-Provider-Header": "provider-header-value",
      },
    })

    await provider.completion("qwen-plus").doStream({
      inputFormat: "prompt",
      mode: { type: "regular" },
      prompt: TEST_PROMPT,
      headers: {
        "Custom-Request-Header": "request-header-value",
      },
    })

    const requestHeaders = await server.getRequestHeaders()

    expect(requestHeaders).toStrictEqual({
      "authorization": "Bearer test-api-key",
      "content-type": "application/json",
      "custom-provider-header": "provider-header-value",
      "custom-request-header": "request-header-value",
    })
  })

  it("should include provider-specific options", async () => {
    prepareStreamResponse({ content: [] })

    const { request: _request } = await model.doStream({
      inputFormat: "prompt",
      mode: { type: "regular" },
      providerMetadata: {
        "test-provider": {
          someCustomOption: "test-value",
        },
      },
      prompt: TEST_PROMPT,
    })

    expect(await server.getRequestBodyJson()).toStrictEqual({
      stream: true,
      model: "qwen-plus",
      prompt: "Hello",
    })
  })

  it("should not include provider-specific options for different provider", async () => {
    prepareStreamResponse({ content: [] })

    const { request: _request } = await model.doStream({
      inputFormat: "prompt",
      mode: { type: "regular" },
      providerMetadata: {
        notThisProviderName: {
          someCustomOption: "test-value",
        },
      },
      prompt: TEST_PROMPT,
    })

    expect(await server.getRequestBodyJson()).toStrictEqual({
      stream: true,
      model: "qwen-plus",
      prompt: "Hello",
    })
  })
})
