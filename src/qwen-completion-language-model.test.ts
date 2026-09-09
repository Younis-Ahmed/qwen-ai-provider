import type { LanguageModelV2Prompt } from "@ai-sdk/provider"
import { convertReadableStreamToArray } from "@ai-sdk/provider-utils/test"
import { createTestServer } from "@ai-sdk/test-server/with-vitest"
import { describe, expect, it, vi } from "vitest"
import { QwenCompletionLanguageModel } from "./qwen-completion-language-model"
import { createQwen } from "./qwen-provider"

const TEST_PROMPT: LanguageModelV2Prompt = [
  { role: "user", content: [{ type: "text", text: "Hello" }] },
]

vi.stubEnv("DASHSCOPE_API_KEY", "test-api-key-123")

const COMPLETIONS_URL = "https://my.api.com/v1/completions"

const COMPLETION_PROMPT = "user:\nHello\n\nassistant:\n"
const COMPLETION_STOP_SEQUENCES = ["\nuser:"]

const provider = createQwen({
  baseURL: "https://my.api.com/v1/",
  headers: {
    Authorization: `Bearer test-api-key`,
  },
})

const model = provider.completion("qwen-plus")

describe("config", () => {
  it("should extract base name from provider string", () => {
    const model = new QwenCompletionLanguageModel("qwen-plus", {}, {
      provider: "qwen.beta",
      url: () => "",
      headers: () => ({}),
    })

    expect(model.providerOptionsName).toBe("qwen")
  })

  it("should handle provider without dot notation", () => {
    const model = new QwenCompletionLanguageModel("qwen-plus", {}, {
      provider: "qwen-plus",
      url: () => "",
      headers: () => ({}),
    })

    expect(model.providerOptionsName).toBe("qwen-plus")
  })

  it("should return empty for empty provider", () => {
    const model = new QwenCompletionLanguageModel("qwen-plus", {}, {
      provider: "",
      url: () => "",
      headers: () => ({}),
    })

    expect(model.providerOptionsName).toBe("")
  })
})

describe("doGenerate", () => {
  const server = createTestServer({
    [COMPLETIONS_URL]: {
      response: {
        type: "json-value",
        body: {
          id: "cmpl-96cAM1v77r4jXa4qb2NSmRREV5oWB",
          object: "text_completion",
          created: 1711363706,
          model: "qwen-plus",
          choices: [
            {
              text: "",
              index: 0,
              finish_reason: "stop",
            },
          ],
          usage: {
            prompt_tokens: 4,
            total_tokens: 34,
            completion_tokens: 30,
          },
        },
      },
    },
  })

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
    server.urls[COMPLETIONS_URL].response = {
      type: "json-value",
      body: {
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
      },
    }
  }

  it("should extract text response", async () => {
    prepareJsonResponse({ content: "Hello, World!" })

    const { content } = await model.doGenerate({ prompt: TEST_PROMPT })

    expect(content).toStrictEqual([{ type: "text", text: "Hello, World!" }])
  })

  it("should extract usage", async () => {
    prepareJsonResponse({
      content: "",
      usage: { prompt_tokens: 20, total_tokens: 25, completion_tokens: 5 },
    })

    const { usage } = await model.doGenerate({ prompt: TEST_PROMPT })

    expect(usage).toStrictEqual({
      inputTokens: 20,
      outputTokens: 5,
      totalTokens: 25,
    })
  })

  it("should send request body", async () => {
    prepareJsonResponse({})

    const { request } = await model.doGenerate({ prompt: TEST_PROMPT })

    expect(request).toStrictEqual({
      body: JSON.stringify({
        model: "qwen-plus",
        prompt: COMPLETION_PROMPT,
        stop: COMPLETION_STOP_SEQUENCES,
      }),
    })
  })

  it("should send additional response information", async () => {
    prepareJsonResponse({
      id: "test-id",
      created: 123,
      model: "test-model",
    })

    const { response } = await model.doGenerate({ prompt: TEST_PROMPT })

    expect(response).toStrictEqual({
      id: "test-id",
      timestamp: new Date(123 * 1000),
      modelId: "test-model",
      headers: expect.any(Object),
    })
  })

  it("should extract finish reason", async () => {
    prepareJsonResponse({
      finish_reason: "stop",
    })

    const { finishReason } = await provider
      .completion("qwen-plus")
      .doGenerate({ prompt: TEST_PROMPT })

    expect(finishReason).toStrictEqual("stop")
  })

  it("should support unknown finish reason", async () => {
    prepareJsonResponse({
      finish_reason: "eos",
    })

    const { finishReason } = await provider
      .completion("qwen-plus")
      .doGenerate({ prompt: TEST_PROMPT })

    expect(finishReason).toStrictEqual("unknown")
  })

  it("should expose the raw response headers", async () => {
    server.urls[COMPLETIONS_URL].response = {
      type: "json-value",
      body: {
        id: "cmpl-96cAM1v77r4jXa4qb2NSmRREV5oWB",
        object: "text_completion",
        created: 1711363706,
        model: "qwen-plus",
        choices: [{ text: "", index: 0, finish_reason: "stop" }],
        usage: {
          prompt_tokens: 4,
          total_tokens: 34,
          completion_tokens: 30,
        },
      },
      headers: {
        "test-header": "test-value",
      },
    }

    const { response } = await model.doGenerate({ prompt: TEST_PROMPT })

    expect(response?.headers?.["test-header"]).toStrictEqual("test-value")
  })

  it("should pass the model and the prompt", async () => {
    prepareJsonResponse({ content: "" })

    await model.doGenerate({ prompt: TEST_PROMPT })

    expect(await server.calls[0].requestBodyJson).toStrictEqual({
      model: "qwen-plus",
      prompt: COMPLETION_PROMPT,
      stop: COMPLETION_STOP_SEQUENCES,
    })
  })

  it("should pass settings", async () => {
    prepareJsonResponse({ content: "" })

    await provider.completion("qwen-plus", {
      user: "test-user-id",
      echo: true,
      suffix: "test-suffix",
      logitBias: { 1234: 2 },
    }).doGenerate({
      prompt: TEST_PROMPT,
      maxOutputTokens: 100,
      temperature: 0.5,
      topP: 0.9,
      stopSequences: ["stop"],
      seed: 42,
    })

    expect(await server.calls[0].requestBodyJson).toMatchObject({
      user: "test-user-id",
      echo: true,
      suffix: "test-suffix",
      logit_bias: { 1234: 2 },
      max_tokens: 100,
      temperature: 0.5,
      top_p: 0.9,
      stop: [...COMPLETION_STOP_SEQUENCES, "stop"],
      seed: 42,
    })
  })

  it("should warn about unsupported settings", async () => {
    prepareJsonResponse({ content: "" })

    const { warnings } = await model.doGenerate({
      prompt: TEST_PROMPT,
      topK: 10,
      tools: [
        {
          type: "function",
          name: "test-tool",
          inputSchema: { type: "object", properties: {} },
        },
      ],
      toolChoice: { type: "auto" },
      responseFormat: { type: "json" },
    })

    expect(warnings).toEqual([
      { type: "unsupported-setting", setting: "topK" },
      { type: "unsupported-setting", setting: "tools" },
      { type: "unsupported-setting", setting: "toolChoice" },
      {
        type: "unsupported-setting",
        setting: "responseFormat",
        details: "JSON response format is not supported.",
      },
    ])
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
      prompt: TEST_PROMPT,
      headers: {
        "Custom-Request-Header": "request-header-value",
      },
    })

    const requestHeaders = server.calls[0].requestHeaders

    expect(requestHeaders).toMatchObject({
      "authorization": "Bearer test-api-key",
      "content-type": "application/json",
      "custom-provider-header": "provider-header-value",
      "custom-request-header": "request-header-value",
    })
  })

  it("should include provider-specific options", async () => {
    prepareJsonResponse({ content: "" })

    await provider.completion("qwen-plus").doGenerate({
      prompt: TEST_PROMPT,
      providerOptions: {
        qwen: {
          someCustomOption: "test-value",
        },
      },
    })

    expect(await server.calls[0].requestBodyJson).toMatchObject({
      someCustomOption: "test-value",
    })
  })

  it("should not include provider-specific options for different provider", async () => {
    prepareJsonResponse({ content: "" })

    await provider.completion("qwen-plus").doGenerate({
      prompt: TEST_PROMPT,
      providerOptions: {
        notThisProviderName: {
          someCustomOption: "test-value",
        },
      },
    })

    expect(await server.calls[0].requestBodyJson).toStrictEqual({
      model: "qwen-plus",
      prompt: COMPLETION_PROMPT,
      stop: COMPLETION_STOP_SEQUENCES,
    })
  })
})

describe("doStream", () => {
  const server = createTestServer({
    [COMPLETIONS_URL]: {
      response: {
        type: "stream-chunks",
        chunks: [],
      },
    },
  })

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
    server.urls[COMPLETIONS_URL].response = {
      type: "stream-chunks",
      chunks: [
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
      ],
    }
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

    const { stream } = await model.doStream({ prompt: TEST_PROMPT })

    const parts = await convertReadableStreamToArray(stream)

    expect(parts[0]).toStrictEqual({ type: "stream-start", warnings: [] })
    expect(parts[1]).toStrictEqual({
      type: "response-metadata",
      id: "cmpl-96c64EdfhOw8pjFFgVpLuT8k2MtdT",
      modelId: "qwen-plus",
      timestamp: new Date("2024-03-25T10:44:00.000Z"),
    })

    const textStart = parts.find(
      (part): part is Extract<typeof part, { type: "text-start" }> => part.type === "text-start",
    )
    expect(textStart).toBeDefined()

    const deltas = parts.filter(
      (part): part is Extract<typeof part, { type: "text-delta" }> => part.type === "text-delta",
    )
    expect(deltas.map(delta => delta.delta)).toStrictEqual([
      "Hello",
      ", ",
      "World!",
      "",
    ])

    const textEnd = parts.find(
      (part): part is Extract<typeof part, { type: "text-end" }> => part.type === "text-end",
    )
    expect(textEnd).toStrictEqual({
      type: "text-end",
      id: textStart?.id,
    })

    expect(parts[parts.length - 1]).toStrictEqual({
      type: "finish",
      finishReason: "stop",
      usage: {
        inputTokens: 10,
        outputTokens: 362,
        totalTokens: 372,
      },
    })
  })

  it("should handle unparsable stream parts", async () => {
    server.urls[COMPLETIONS_URL].response = {
      type: "stream-chunks",
      chunks: [`data: {unparsable}\n\n`, "data: [DONE]\n\n"],
    }

    const { stream } = await model.doStream({ prompt: TEST_PROMPT })

    const elements = await convertReadableStreamToArray(stream)

    expect(elements.length).toBe(2)
    expect(elements[0].type).toBe("error")
    expect(elements[1]).toStrictEqual({
      finishReason: "error",
      type: "finish",
      usage: {
        inputTokens: Number.NaN,
        outputTokens: Number.NaN,
        totalTokens: undefined,
      },
    })
  })

  it("should send request body", async () => {
    prepareStreamResponse({ content: [] })

    const { request } = await model.doStream({ prompt: TEST_PROMPT })

    expect(request?.body).toStrictEqual(
      JSON.stringify({
        model: "qwen-plus",
        prompt: COMPLETION_PROMPT,
        stop: COMPLETION_STOP_SEQUENCES,
        stream: true,
      }),
    )
  })

  it("should expose the raw response headers", async () => {
    prepareStreamResponse({ content: [] })

    const { response } = await model.doStream({ prompt: TEST_PROMPT })

    expect(response?.headers).toBeDefined()
  })

  it("should pass the model and the prompt", async () => {
    prepareStreamResponse({ content: [] })

    await model.doStream({ prompt: TEST_PROMPT })

    expect(await server.calls[0].requestBodyJson).toStrictEqual({
      stream: true,
      model: "qwen-plus",
      prompt: COMPLETION_PROMPT,
      stop: COMPLETION_STOP_SEQUENCES,
    })
  })

  it("should warn about unsupported settings on the stream start part", async () => {
    prepareStreamResponse({ content: ["Hello"] })

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
      topK: 10,
    })

    const parts = await convertReadableStreamToArray(stream)

    expect(parts[0]).toStrictEqual({
      type: "stream-start",
      warnings: [{ type: "unsupported-setting", setting: "topK" }],
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
      prompt: TEST_PROMPT,
      headers: {
        "Custom-Request-Header": "request-header-value",
      },
    })

    const requestHeaders = server.calls[0].requestHeaders

    expect(requestHeaders).toMatchObject({
      "authorization": "Bearer test-api-key",
      "content-type": "application/json",
      "custom-provider-header": "provider-header-value",
      "custom-request-header": "request-header-value",
    })
  })

  it("should include provider-specific options", async () => {
    prepareStreamResponse({ content: [] })

    await provider.completion("qwen-plus").doStream({
      prompt: TEST_PROMPT,
      providerOptions: {
        qwen: {
          someCustomOption: "test-value",
        },
      },
    })

    expect(await server.calls[0].requestBodyJson).toMatchObject({
      someCustomOption: "test-value",
    })
  })

  it("should not include provider-specific options for different provider", async () => {
    prepareStreamResponse({ content: [] })

    await provider.completion("qwen-plus").doStream({
      prompt: TEST_PROMPT,
      providerOptions: {
        notThisProviderName: {
          someCustomOption: "test-value",
        },
      },
    })

    expect(await server.calls[0].requestBodyJson).toStrictEqual({
      stream: true,
      model: "qwen-plus",
      prompt: COMPLETION_PROMPT,
      stop: COMPLETION_STOP_SEQUENCES,
    })
  })
})
