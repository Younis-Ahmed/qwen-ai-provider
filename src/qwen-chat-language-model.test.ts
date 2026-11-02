import type {
  LanguageModelV2FunctionTool,
  LanguageModelV2Prompt,
  LanguageModelV2StreamPart,
} from "@ai-sdk/provider"
import type { JSONSchema7 } from "json-schema"
import type { MetadataExtractor } from "./qwen-metadata-extractor"
import { InvalidResponseDataError } from "@ai-sdk/provider"
import { convertReadableStreamToArray } from "@ai-sdk/provider-utils/test"
import { createTestServer } from "@ai-sdk/test-server/with-vitest"
import { describe, expect, it, vi } from "vitest"
import { QwenChatLanguageModel } from "./qwen-chat-language-model"
import { createQwen } from "./qwen-provider"

const TEST_PROMPT: LanguageModelV2Prompt = [
  {
    role: "user",
    content: [
      { type: "text", text: "Hello" },
    ],
  },
]

vi.stubEnv("DASHSCOPE_API_KEY", "test-api-key-123")

const CHAT_COMPLETIONS_URL = "https://my.api.com/v1/chat/completions"

const provider = createQwen({
  baseURL: "https://my.api.com/v1/",
  headers: {
    Authorization: `Bearer test-api-key`,
  },
})

const model = provider("qwen-chat")

function streamText(parts: LanguageModelV2StreamPart[]): string {
  return parts
    .filter(
      (part): part is Extract<LanguageModelV2StreamPart, { type: "text-delta" }> =>
        part.type === "text-delta",
    )
    .map(part => part.delta)
    .join("")
}

function streamReasoning(parts: LanguageModelV2StreamPart[]): string {
  return parts
    .filter(
      (part):
        part is Extract<LanguageModelV2StreamPart, { type: "reasoning-delta" }> =>
        part.type === "reasoning-delta",
    )
    .map(part => part.delta)
    .join("")
}

function streamToolCalls(parts: LanguageModelV2StreamPart[]): string[] {
  return parts
    .filter(
      (part): part is Extract<LanguageModelV2StreamPart, { type: "tool-call" }> =>
        part.type === "tool-call",
    )
    .map(part => part.input as string)
}

const metadataExtractor: MetadataExtractor = {
  extractMetadata: ({ parsedBody }) => {
    const parsed = parsedBody as {
      model?: string
      usage?: { total_tokens?: number }
    }

    return {
      qwen: {
        modelId: parsed.model ?? "",
        usage: parsed.usage?.total_tokens ?? 0,
      },
    }
  },
  createStreamExtractor: () => {
    let processedChunkCount = 0

    return {
      processChunk: () => {
        processedChunkCount++
      },
      buildMetadata: () => {
        return processedChunkCount > 0
          ? {
              qwen: { processedChunkCount },
            }
          : undefined
      },
    }
  },
}

const testTool: LanguageModelV2FunctionTool = {
  type: "function",
  name: "test-tool",
  inputSchema: {
    type: "object",
    properties: {
      value: { type: "string" },
    },
    required: ["value"],
    additionalProperties: false,
    $schema: "http://json-schema.org/draft-07/schema#",
  },
}

describe("config", () => {
  it("should extract base name from provider string", () => {
    const model = new QwenChatLanguageModel("qwen-plus", {}, {
      provider: "qwen.beta",
      url: () => "",
      headers: () => ({}),
    })

    expect(model.providerOptionsName).toBe("qwen")
  })

  it("should handle provider without dot notation", () => {
    const model = new QwenChatLanguageModel("qwen-plus", {}, {
      provider: "qwen-plus",
      url: () => "",
      headers: () => ({}),
    })

    expect(model.providerOptionsName).toBe("qwen-plus")
  })

  it("should return empty for empty provider", () => {
    const model = new QwenChatLanguageModel("qwen-plus", {}, {
      provider: "",
      url: () => "",
      headers: () => ({}),
    })

    expect(model.providerOptionsName).toBe("")
  })

  it("should expose the supported urls", () => {
    expect(model.supportedUrls).toStrictEqual({
      "image/*": [/^https?:\/\/.*$/],
      "video/*": [/^https?:\/\/.*$/],
      "audio/*": [/^https?:\/\/.*$/],
      "application/pdf": [/^https?:\/\/.*$/],
    })
  })
})

describe("doGenerate", () => {
  const server = createTestServer({
    [CHAT_COMPLETIONS_URL]: {
      response: {
        type: "json-value",
        body: {
          id: "chatcmpl-95ZTZkhr0mHNKqerQfiwkuox3PHAd",
          object: "chat.completion",
          created: 1711115037,
          model: "qwen-chat",
          choices: [
            {
              index: 0,
              message: {
                role: "assistant",
                content: "Hello, World!",
              },
              finish_reason: "stop",
            },
          ],
          usage: {
            prompt_tokens: 4,
            total_tokens: 34,
            completion_tokens: 30,
          },
          system_fingerprint: "fp_3bc1b5746c",
        },
      },
    },
  })

  function prepareJsonResponse({
    content = "",
    reasoning_content = "",
    tool_calls,
    usage = {
      prompt_tokens: 4,
      total_tokens: 34,
      completion_tokens: 30,
    },
    finish_reason = "stop",
    id = "chatcmpl-95ZTZkhr0mHNKqerQfiwkuox3PHAd",
    created = 1711115037,
    model = "qwen-chat",
  }: {
    content?: string | null
    reasoning_content?: string
    tool_calls?: Array<{
      id: string
      type: "function"
      function: {
        name: string
        arguments: string
      }
    }>
    usage?: {
      prompt_tokens?: number
      total_tokens?: number
      completion_tokens?: number
    }
    finish_reason?: string
    created?: number
    id?: string
    model?: string
  } = {}) {
    server.urls[CHAT_COMPLETIONS_URL].response = {
      type: "json-value",
      body: {
        id,
        object: "chat.completion",
        created,
        model,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              ...(content != null
                ? { content }
                : {}),
              ...(reasoning_content !== ""
                ? { reasoning_content }
                : {}),
              tool_calls,
            },
            finish_reason,
          },
        ],
        usage,
        system_fingerprint: "fp_3bc1b5746c",
      },
    }
  }

  it("should pass user setting to requests", async () => {
    prepareJsonResponse({ content: "Hello, World!" })
    const modelWithUser = provider("qwen-chat", {
      user: "test-user-id",
    })
    await modelWithUser.doGenerate({ prompt: TEST_PROMPT })
    expect(await server.calls[0].requestBodyJson).toMatchObject({
      user: "test-user-id",
    })
  })

  it("should extract text response", async () => {
    prepareJsonResponse({ content: "Hello, World!" })

    const { content } = await model.doGenerate({ prompt: TEST_PROMPT })

    expect(content).toStrictEqual([
      { type: "text", text: "Hello, World!" },
    ])
  })

  it("should extract reasoning content before text", async () => {
    prepareJsonResponse({
      content: "Hello, World!",
      reasoning_content: "This is the reasoning behind the response",
    })

    const { content } = await model.doGenerate({ prompt: TEST_PROMPT })

    expect(content).toStrictEqual([
      {
        type: "reasoning",
        text: "This is the reasoning behind the response",
      },
      { type: "text", text: "Hello, World!" },
    ])
  })

  it("should extract tool calls", async () => {
    prepareJsonResponse({
      tool_calls: [
        {
          id: "call_O17Uplv4lJvD6DVdIvFFeRMw",
          type: "function",
          function: {
            name: "test-tool",
            arguments: "{\"value\":\"Spark\"}",
          },
        },
      ],
    })

    const { content } = await model.doGenerate({ prompt: TEST_PROMPT })

    expect(content).toStrictEqual([
      { type: "text", text: "" },
      {
        type: "tool-call",
        toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
        toolName: "test-tool",
        input: "{\"value\":\"Spark\"}",
      },
    ])
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

  it("should support partial usage", async () => {
    prepareJsonResponse({
      content: "",
      usage: { prompt_tokens: 20, total_tokens: 20 },
    })

    const { usage } = await model.doGenerate({ prompt: TEST_PROMPT })

    expect(usage).toStrictEqual({
      inputTokens: 20,
      outputTokens: Number.NaN,
      totalTokens: undefined,
    })
  })

  it("should extract finish reason", async () => {
    prepareJsonResponse({
      finish_reason: "stop",
    })

    const { finishReason } = await model.doGenerate({ prompt: TEST_PROMPT })

    expect(finishReason).toStrictEqual("stop")
  })

  it("should support unknown finish reason", async () => {
    prepareJsonResponse({
      finish_reason: "eos",
    })

    const { finishReason } = await model.doGenerate({ prompt: TEST_PROMPT })

    expect(finishReason).toStrictEqual("unknown")
  })

  it("should expose the raw response headers", async () => {
    prepareJsonResponse({ content: "" })

    server.urls[CHAT_COMPLETIONS_URL].response = {
      type: "json-value",
      body: {
        id: "chatcmpl-95ZTZkhr0mHNKqerQfiwkuox3PHAd",
        object: "chat.completion",
        created: 1711115037,
        model: "qwen-chat",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "" },
            finish_reason: "stop",
          },
        ],
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

  it("should pass the model and the messages", async () => {
    prepareJsonResponse({ content: "" })

    await model.doGenerate({ prompt: TEST_PROMPT })

    expect(await server.calls[0].requestBodyJson).toStrictEqual({
      model: "qwen-chat",
      messages: [
        { role: "user", content: "Hello" },
      ],
    })
  })

  it("should pass standardized options", async () => {
    prepareJsonResponse({ content: "" })

    await provider("qwen-chat", {
      user: "test-user-id",
    }).doGenerate({
      prompt: TEST_PROMPT,
      maxOutputTokens: 100,
      temperature: 0.5,
      topP: 0.9,
      topK: 10,
      stopSequences: ["stop"],
      seed: 42,
      presencePenalty: 0.3,
      frequencyPenalty: 0.7,
    })

    expect(await server.calls[0].requestBodyJson).toMatchObject({
      user: "test-user-id",
      max_tokens: 100,
      temperature: 0.5,
      top_p: 0.9,
      stop: ["stop"],
      seed: 42,
      presence_penalty: 0.3,
      frequency_penalty: 0.7,
    })
  })

  it("should warn about unsupported topK setting", async () => {
    prepareJsonResponse({ content: "" })

    const { warnings } = await model.doGenerate({
      prompt: TEST_PROMPT,
      topK: 10,
    })

    expect(warnings).toEqual([
      { type: "unsupported-setting", setting: "topK" },
    ])
  })

  it("should include provider-specific options", async () => {
    prepareJsonResponse({ content: "" })

    await provider("qwen-chat").doGenerate({
      prompt: TEST_PROMPT,
      providerOptions: {
        qwen: {
          cacheControl: { type: "ephemeral" },
        },
      },
    })

    expect(await server.calls[0].requestBodyJson).toMatchObject({
      cacheControl: { type: "ephemeral" },
    })
  })

  it("should not include provider-specific options for different provider", async () => {
    prepareJsonResponse({ content: "" })

    await provider("qwen-chat").doGenerate({
      prompt: TEST_PROMPT,
      providerOptions: {
        notThisProviderName: {
          someCustomOption: "test-value",
        },
      },
    })

    expect(await server.calls[0].requestBodyJson).toStrictEqual({
      model: "qwen-chat",
      messages: [
        { role: "user", content: "Hello" },
      ],
    })
  })

  it("should pass tools and toolChoice", async () => {
    prepareJsonResponse({ content: "" })

    await model.doGenerate({
      prompt: TEST_PROMPT,
      tools: [testTool],
      toolChoice: {
        type: "tool",
        toolName: "test-tool",
      },
    })

    expect(await server.calls[0].requestBodyJson).toStrictEqual({
      model: "qwen-chat",
      messages: [
        { role: "user", content: "Hello" },
      ],
      tools: [
        {
          type: "function",
          function: {
            name: "test-tool",
            parameters: testTool.inputSchema,
          },
        },
      ],
      tool_choice: {
        type: "function",
        function: { name: "test-tool" },
      },
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

    await provider("qwen-chat").doGenerate({
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

  describe("response format", () => {
    it("should not send a response_format when response format is text", async () => {
      prepareJsonResponse({ content: "{\"value\":\"Spark\"}" })

      await model.doGenerate({
        prompt: TEST_PROMPT,
        responseFormat: { type: "text" },
      })

      expect(await server.calls[0].requestBodyJson).toStrictEqual({
        model: "qwen-chat",
        messages: [
          { role: "user", content: "Hello" },
        ],
      })
    })

    it("should forward json response format as \"json_object\" without schema", async () => {
      prepareJsonResponse({ content: "{\"value\":\"Spark\"}" })

      await model.doGenerate({
        prompt: TEST_PROMPT,
        responseFormat: { type: "json" },
      })

      expect(await server.calls[0].requestBodyJson).toStrictEqual({
        model: "qwen-chat",
        messages: [
          { role: "user", content: "Hello" },
        ],
        response_format: { type: "json_object" },
      })
    })

    it("should forward json response format as \"json_object\" and omit schema when structuredOutputs are disabled", async () => {
      prepareJsonResponse({ content: "{\"value\":\"Spark\"}" })

      const model = new QwenChatLanguageModel("qwen-plus", {}, {
        provider: "test-provider",
        url: () => CHAT_COMPLETIONS_URL,
        headers: () => ({}),
        supportsStructuredOutputs: false,
      })

      await model.doGenerate({
        prompt: TEST_PROMPT,
        responseFormat: {
          type: "json",
          schema: {
            type: "object",
            properties: {
              value: { type: "string" },
            },
            required: ["value"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        },
      })

      expect(await server.calls[0].requestBodyJson).toStrictEqual({
        model: "qwen-plus",
        messages: [
          { role: "user", content: "Hello" },
        ],
        response_format: { type: "json_object" },
      })
    })

    it("should forward json response format as \"json_schema\" with schema when structuredOutputs are enabled", async () => {
      prepareJsonResponse({ content: "{\"value\":\"Spark\"}" })

      const model = new QwenChatLanguageModel("qwen-plus", {}, {
        provider: "test-provider",
        url: () => CHAT_COMPLETIONS_URL,
        headers: () => ({}),
        supportsStructuredOutputs: true,
      })

      const schema: JSONSchema7 = {
        type: "object",
        properties: {
          value: { type: "string" },
        },
        required: ["value"],
        additionalProperties: false,
        $schema: "http://json-schema.org/draft-07/schema#",
      }

      const { warnings } = await model.doGenerate({
        prompt: TEST_PROMPT,
        responseFormat: {
          type: "json",
          schema,
        },
      })

      expect(await server.calls[0].requestBodyJson).toStrictEqual({
        model: "qwen-plus",
        messages: [
          { role: "user", content: "Hello" },
        ],
        response_format: {
          type: "json_schema",
          json_schema: {
            name: "response",
            schema,
          },
        },
      })

      expect(warnings).toEqual([])
    })

    it("should set name & description in json response format when structuredOutputs are enabled", async () => {
      prepareJsonResponse({ content: "{\"value\":\"Spark\"}" })

      const model = new QwenChatLanguageModel("qwen-plus", {}, {
        provider: "test-provider",
        url: () => CHAT_COMPLETIONS_URL,
        headers: () => ({}),
        supportsStructuredOutputs: true,
      })

      const schema: JSONSchema7 = {
        type: "object",
        properties: {
          value: { type: "string" },
        },
        required: ["value"],
        additionalProperties: false,
      }

      await model.doGenerate({
        prompt: TEST_PROMPT,
        responseFormat: {
          type: "json",
          name: "test-name",
          description: "test description",
          schema,
        },
      })

      expect(await server.calls[0].requestBodyJson).toStrictEqual({
        model: "qwen-plus",
        messages: [
          { role: "user", content: "Hello" },
        ],
        response_format: {
          type: "json_schema",
          json_schema: {
            name: "test-name",
            description: "test description",
            schema,
          },
        },
      })
    })

    it("should fall back to \"json_object\" when schema is undefined even with structuredOutputs enabled", async () => {
      prepareJsonResponse({ content: "{\"value\":\"Spark\"}" })

      const model = new QwenChatLanguageModel("qwen-plus", {}, {
        provider: "test-provider",
        url: () => CHAT_COMPLETIONS_URL,
        headers: () => ({}),
        supportsStructuredOutputs: true,
      })

      await model.doGenerate({
        prompt: TEST_PROMPT,
        responseFormat: {
          type: "json",
          name: "test-name",
        },
      })

      expect(await server.calls[0].requestBodyJson).toStrictEqual({
        model: "qwen-plus",
        messages: [
          { role: "user", content: "Hello" },
        ],
        response_format: {
          type: "json_object",
        },
      })
    })
  })

  it("should send request body", async () => {
    prepareJsonResponse({ content: "" })

    const { request } = await model.doGenerate({ prompt: TEST_PROMPT })

    expect(request).toStrictEqual({
      body: "{\"model\":\"qwen-chat\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}",
    })
  })

  it("should return provider metadata when a metadata extractor is configured", async () => {
    const modelWithMetadata = new QwenChatLanguageModel(
      "qwen-plus",
      {},
      {
        provider: "test-provider",
        url: () => CHAT_COMPLETIONS_URL,
        headers: () => ({}),
        metadataExtractor,
      },
    )

    const { providerMetadata } = await modelWithMetadata.doGenerate({
      prompt: TEST_PROMPT,
    })

    expect(providerMetadata).toMatchObject({
      qwen: {
        modelId: "qwen-chat",
        usage: 34,
      },
    })
  })

  it("should throw when the model does not return content", async () => {
    prepareJsonResponse({ content: null })

    await expect(model.doGenerate({ prompt: TEST_PROMPT })).rejects.toBeInstanceOf(
      InvalidResponseDataError,
    )
  })
})

describe("doStream", () => {
  const server = createTestServer({
    [CHAT_COMPLETIONS_URL]: {
      response: {
        type: "stream-chunks",
        chunks: [],
      },
    },
  })

  function prepareStreamResponse({
    content,
    finish_reason = "stop",
  }: {
    content: string[]
    finish_reason?: string
  }) {
    server.urls[CHAT_COMPLETIONS_URL].response = {
      type: "stream-chunks",
      chunks: [
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1702657020,"model":"qwen-chat",`
        + `"system_fingerprint":null,"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n`,
        ...content.map((text) => {
          return (
            `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1702657020,"model":"qwen-chat",`
            + `"system_fingerprint":null,"choices":[{"index":1,"delta":{"content":"${text}"},"finish_reason":null}]}\n\n`
          )
        }),
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1702657020,"model":"qwen-chat",`
        + `"system_fingerprint":null,"choices":[{"index":0,"delta":{},"finish_reason":"${finish_reason}"}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1729171479,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_10c08bf97d","choices":[{"index":0,"delta":{},"finish_reason":"${finish_reason}"}],`
        + `"usage":{"queue_time":0.061348671,"prompt_tokens":18,"prompt_time":0.000211569,`
        + `"completion_tokens":439,"completion_time":0.798181818,"total_tokens":457,"total_time":0.798393387}}\n\n`,
        "data: [DONE]\n\n",
      ],
    }
  }

  it("should stream text deltas", async () => {
    prepareStreamResponse({
      content: ["Hello", ", ", "World!"],
      finish_reason: "stop",
    })

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
    })

    const parts = await convertReadableStreamToArray(stream)

    expect(parts.slice(0, 2)).toStrictEqual([
      {
        type: "stream-start",
        warnings: [],
      },
      {
        type: "response-metadata",
        id: "chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798",
        modelId: "qwen-chat",
        timestamp: new Date("2023-12-15T16:17:00.000Z"),
      },
    ])

    const textStart = parts.find((part): part is Extract<LanguageModelV2StreamPart, { type: "text-start" }> => part.type === "text-start")
    expect(textStart).toBeDefined()

    const deltas = parts.filter((part): part is Extract<LanguageModelV2StreamPart, { type: "text-delta" }> => part.type === "text-delta")
    expect(deltas.map(delta => delta.delta)).toStrictEqual(["", "Hello", ", ", "World!"])
    expect(deltas.every(delta => delta.id === textStart?.id)).toBe(true)

    expect(parts.find((part): part is Extract<LanguageModelV2StreamPart, { type: "text-end" }> => part.type === "text-end")?.id).toBe(textStart?.id)

    expect(parts[parts.length - 1]).toStrictEqual({
      type: "finish",
      finishReason: "stop",
      usage: {
        inputTokens: 18,
        outputTokens: 439,
        totalTokens: 457,
      },
    })
  })

  it("should emit reasoning before text deltas", async () => {
    server.urls[CHAT_COMPLETIONS_URL].response = {
      type: "stream-chunks",
      chunks: [
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1702657020,"model":"qwen-chat",`
        + `"system_fingerprint":null,"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello","reasoning_content":"thinking"},"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1729171479,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_10c08bf97d","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],`
        + `"usage":{"queue_time":0.061348671,"prompt_tokens":18,"prompt_time":0.000211569,`
        + `"completion_tokens":439,"completion_time":0.798181818,"total_tokens":457,"total_time":0.798393387}}\n\n`,
        "data: [DONE]\n\n",
      ],
    }

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
    })

    const parts = await convertReadableStreamToArray(stream)

    expect(parts.map(part => part.type)).toStrictEqual([
      "stream-start",
      "response-metadata",
      "reasoning-start",
      "reasoning-delta",
      "text-start",
      "text-delta",
      "text-end",
      "reasoning-end",
      "finish",
    ])

    expect(streamReasoning(parts)).toStrictEqual("thinking")
    expect(streamText(parts)).toStrictEqual("Hello")

    // exit early check for tool calls
    expect(streamToolCalls(parts)).toStrictEqual([])
  })

  it("should stream tool deltas", async () => {
    server.urls[CHAT_COMPLETIONS_URL].response = {
      type: "stream-chunks",
      chunks: [
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"role":"assistant","content":null,`
        + `"tool_calls":[{"index":0,"id":"call_O17Uplv4lJvD6DVdIvFFeRMw","type":"function","function":{"name":"test-tool","arguments":""}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\""}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"value"}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\":\\""}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Spark"}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"le"}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":" Day"}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"}"}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1729171479,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_10c08bf97d","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],`
        + `"usage":{"queue_time":0.061348671,"prompt_tokens":18,"prompt_time":0.000211569,`
        + `"completion_tokens":439,"completion_time":0.798181818,"total_tokens":457,"total_time":0.798393387}}\n\n`,
        "data: [DONE]\n\n",
      ],
    }

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
      tools: [testTool],
    })

    const parts = await convertReadableStreamToArray(stream)

    const toolInputStart = parts.find((part): part is Extract<LanguageModelV2StreamPart, { type: "tool-input-start" }> => part.type === "tool-input-start")
    expect(toolInputStart).toStrictEqual({
      type: "tool-input-start",
      toolName: "test-tool",
      id: toolInputStart?.id,
    })

    const deltas = parts.filter((part): part is Extract<LanguageModelV2StreamPart, { type: "tool-input-delta" }> => part.type === "tool-input-delta")
    expect(deltas.map(delta => delta.delta)).toStrictEqual([
      "{\"",
      "value",
      "\":\"",
      "Spark",
      "le",
      " Day",
      "\"}",
    ])
    expect(deltas.every(delta => delta.id === toolInputStart?.id)).toBe(true)

    expect(parts.find((part): part is Extract<LanguageModelV2StreamPart, { type: "tool-input-end" }> => part.type === "tool-input-end")?.id).toBe(toolInputStart?.id)

    expect(streamToolCalls(parts)).toStrictEqual(["{\"value\":\"Sparkle Day\"}"])

    expect(parts[parts.length - 1]).toStrictEqual({
      type: "finish",
      finishReason: "tool-calls",
      usage: {
        inputTokens: 18,
        outputTokens: 439,
        totalTokens: 457,
      },
    })
  })

  it("should stream tool call deltas when tool call arguments are passed in the first chunk", async () => {
    server.urls[CHAT_COMPLETIONS_URL].response = {
      type: "stream-chunks",
      chunks: [
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"role":"assistant","content":null,`
        + `"tool_calls":[{"index":0,"id":"call_O17Uplv4lJvD6DVdIvFFeRMw","type":"function","function":{"name":"test-tool","arguments":"{\\"value\\":\\"Sparkle Day\\"}"}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1729171479,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_10c08bf97d","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],`
        + `"usage":{"queue_time":0.061348671,"prompt_tokens":18,"prompt_time":0.000211569,`
        + `"completion_tokens":439,"completion_time":0.798181818,"total_tokens":457,"total_time":0.798393387}}\n\n`,
        "data: [DONE]\n\n",
      ],
    }

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
      tools: [testTool],
    })

    const parts = await convertReadableStreamToArray(stream)

    const types = parts.map(part => part.type)
    expect(types).toMatchObject([
      "stream-start",
      "response-metadata",
      "tool-input-start",
      "tool-input-delta",
      "tool-input-end",
      "tool-call",
      "finish",
    ])

    expect(streamToolCalls(parts)).toStrictEqual(["{\"value\":\"Sparkle Day\"}"])
  })

  it("should not duplicate tool calls when there is an additional empty chunk after the tool call has been completed", async () => {
    server.urls[CHAT_COMPLETIONS_URL].response = {
      type: "stream-chunks",
      chunks: [
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"role":"assistant","content":null,`
        + `"tool_calls":[{"index":0,"id":"call_O17Uplv4lJvD6DVdIvFFeRMw","type":"function","function":{"name":"test-tool","arguments":"{\\"value\\":\\"Sparkle Day\\"}"}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":""}}]},`
        + `"finish_reason":null}]}\n\n`,
        `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1729171479,"model":"qwen-chat",`
        + `"system_fingerprint":"fp_10c08bf97d","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],`
        + `"usage":{"queue_time":0.061348671,"prompt_tokens":18,"prompt_time":0.000211569,`
        + `"completion_tokens":439,"completion_time":0.798181818,"total_tokens":457,"total_time":0.798393387}}\n\n`,
        "data: [DONE]\n\n",
      ],
    }

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
      tools: [testTool],
    })

    const parts = await convertReadableStreamToArray(stream)

    expect(streamToolCalls(parts)).toStrictEqual(["{\"value\":\"Sparkle Day\"}"])
  })

  it("should emit an error part for unparsable stream chunks", async () => {
    server.urls[CHAT_COMPLETIONS_URL].response = {
      type: "stream-chunks",
      chunks: [
        `data: {"error":{"message":"boom"}}\n\n`,
        "data: [DONE]\n\n",
      ],
    }

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
    })

    const parts = await convertReadableStreamToArray(stream)

    expect(parts.length).toBe(2)
    expect(parts[0]).toMatchObject({
      type: "error",
    })
    expect(parts[1]).toMatchObject({
      type: "finish",
      finishReason: "error",
    })
  })

  it("should expose the raw response headers", async () => {
    prepareStreamResponse({
      content: ["Hello"],
    })

    const { response } = await model.doStream({
      prompt: TEST_PROMPT,
    })

    expect(response?.headers).toBeDefined()
  })

  it("should pass the messages and the model", async () => {
    prepareStreamResponse({
      content: ["Hello"],
    })

    await model.doStream({
      prompt: TEST_PROMPT,
    })

    expect(await server.calls[0].requestBodyJson).toMatchObject({
      model: "qwen-chat",
      messages: [
        { role: "user", content: "Hello" },
      ],
      stream: true,
      stream_options: {
        include_usage: true,
      },
    })
  })

  it("should warn about unsupported topK setting", async () => {
    prepareStreamResponse({
      content: ["Hello"],
    })

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
      topK: 10,
    })

    const parts = await convertReadableStreamToArray(stream)

    expect(parts[0]).toStrictEqual({
      type: "stream-start",
      warnings: [
        { type: "unsupported-setting", setting: "topK" },
      ],
    })
  })

  it("should pass headers", async () => {
    prepareStreamResponse({
      content: ["Hello"],
    })

    const provider = createQwen({
      baseURL: "https://my.api.com/v1/",
      headers: {
        "Authorization": `Bearer test-api-key`,
        "Custom-Provider-Header": "provider-header-value",
      },
    })

    await provider("qwen-chat").doStream({
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
    prepareStreamResponse({
      content: ["Hello"],
    })

    await model.doStream({
      prompt: TEST_PROMPT,
      providerOptions: {
        qwen: {
          cacheControl: { type: "ephemeral" },
        },
      },
    })

    expect(await server.calls[0].requestBodyJson).toMatchObject({
      cacheControl: { type: "ephemeral" },
    })
  })

  it("should not include provider-specific options for different provider", async () => {
    prepareStreamResponse({
      content: ["Hello"],
    })

    await provider("qwen-chat").doStream({
      prompt: TEST_PROMPT,
      providerOptions: {
        notThisProviderName: {
          someCustomOption: "test-value",
        },
      },
    })

    expect(await server.calls[0].requestBodyJson).toStrictEqual({
      model: "qwen-chat",
      messages: [
        { role: "user", content: "Hello" },
      ],
      stream: true,
      stream_options: {
        include_usage: true,
      },
    })
  })

  it("should send request body", async () => {
    prepareStreamResponse({
      content: ["Hello"],
    })

    const { request } = await model.doStream({
      prompt: TEST_PROMPT,
    })

    expect(request?.body).toStrictEqual(
      `{"model":"qwen-chat","messages":[{"role":"user","content":"Hello"}],"stream":true,"stream_options":{"include_usage":true}}`,
    )
  })

  it("should return provider metadata on the finish part when a metadata extractor is configured", async () => {
    const modelWithMetadata = new QwenChatLanguageModel(
      "qwen-plus",
      {},
      {
        provider: "test-provider",
        url: () => CHAT_COMPLETIONS_URL,
        headers: () => ({}),
        metadataExtractor,
      },
    )

    prepareStreamResponse({
      content: ["Hello"],
    })

    const { stream } = await modelWithMetadata.doStream({
      prompt: TEST_PROMPT,
    })

    const parts = await convertReadableStreamToArray(stream)

    const finish = parts.find((part): part is Extract<LanguageModelV2StreamPart, { type: "finish" }> => part.type === "finish")
    expect(finish?.providerMetadata).toMatchObject({
      qwen: {
        processedChunkCount: 4,
      },
    })
  })
})

describe("doStream simulated streaming", () => {
  const server = createTestServer({
    [CHAT_COMPLETIONS_URL]: {
      response: {
        type: "json-value",
        body: {
          id: "chatcmpl-95ZTZkhr0mHNKqerQfiwkuox3PHAd",
          object: "chat.completion",
          created: 1711115037,
          model: "qwen-chat",
          choices: [
            {
              index: 0,
              message: {
                role: "assistant",
                content: "",
              },
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
    reasoning_content = "",
    tool_calls,
    finish_reason = "stop",
  }: {
    content?: string
    reasoning_content?: string
    tool_calls?: Array<{
      id: string
      type: "function"
      function: {
        name: string
        arguments: string
      }
    }>
    finish_reason?: string
  } = {}) {
    server.urls[CHAT_COMPLETIONS_URL].response = {
      type: "json-value",
      body: {
        id: "chatcmpl-95ZTZkhr0mHNKqerQfiwkuox3PHAd",
        object: "chat.completion",
        created: 1711115037,
        model: "qwen-chat",
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              ...(content !== ""
                ? { content }
                : {}),
              ...(reasoning_content !== ""
                ? { reasoning_content }
                : {}),
              tool_calls,
            },
            finish_reason,
          },
        ],
        usage: {
          prompt_tokens: 4,
          total_tokens: 34,
          completion_tokens: 30,
        },
      },
    }
  }

  it("should stream text delta", async () => {
    prepareJsonResponse({
      content: "Hello, World!",
    })

    const { stream } = await provider("qwen-chat", {
      simulateStreaming: true,
    }).doStream({
      prompt: TEST_PROMPT,
    })

    const parts = await convertReadableStreamToArray(stream)

    expect(parts[0]).toStrictEqual({ type: "stream-start", warnings: [] })
    expect(streamText(parts)).toStrictEqual("Hello, World!")

    const finish = parts.find((part): part is Extract<LanguageModelV2StreamPart, { type: "finish" }> => part.type === "finish")
    expect(finish).toStrictEqual({
      type: "finish",
      finishReason: "stop",
      usage: {
        inputTokens: 4,
        outputTokens: 30,
        totalTokens: 34,
      },
    })
  })

  it("should stream reasoning content before text delta in simulated streaming", async () => {
    prepareJsonResponse({
      content: "Hello, World!",
      reasoning_content: "thinking",
    })

    const { stream } = await provider("qwen-chat", {
      simulateStreaming: true,
    }).doStream({
      prompt: TEST_PROMPT,
    })

    const parts = await convertReadableStreamToArray(stream)

    expect(parts.map(part => part.type)).toStrictEqual([
      "stream-start",
      "response-metadata",
      "reasoning-start",
      "reasoning-delta",
      "text-start",
      "text-delta",
      "text-end",
      "reasoning-end",
      "finish",
    ])

    expect(streamReasoning(parts)).toStrictEqual("thinking")
    expect(streamText(parts)).toStrictEqual("Hello, World!")
  })

  it("should stream tool calls", async () => {
    prepareJsonResponse({
      tool_calls: [
        {
          id: "call_O17Uplv4lJvD6DVdIvFFeRMw",
          type: "function",
          function: {
            name: "test-tool",
            arguments: "{\"value\":\"Sparkle Day\"}",
          },
        },
      ],
    })

    const { stream } = await provider("qwen-chat", {
      simulateStreaming: true,
    }).doStream({
      prompt: TEST_PROMPT,
    })

    const parts = await convertReadableStreamToArray(stream)

    expect(parts.map(part => part.type)).toStrictEqual([
      "stream-start",
      "response-metadata",
      "tool-input-start",
      "tool-input-delta",
      "tool-input-end",
      "tool-call",
      "finish",
    ])

    expect(streamToolCalls(parts)).toStrictEqual(["{\"value\":\"Sparkle Day\"}"])
  })

  it("should include the stream option in the request body", async () => {
    prepareJsonResponse({
      content: "Hello, World!",
    })

    const { request } = await provider("qwen-chat", {
      simulateStreaming: true,
    }).doStream({
      prompt: TEST_PROMPT,
    })

    const requestBody = JSON.parse(request?.body as string)
    expect(requestBody.stream).toBe(true)
  })
})
