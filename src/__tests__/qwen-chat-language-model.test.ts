/* eslint-disable dot-notation */
import type { LanguageModelV2Prompt } from "@ai-sdk/provider"
import { convertReadableStreamToArray } from "@ai-sdk/provider-utils/test"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { QwenChatLanguageModel } from "../qwen-chat-language-model"
import { createQwen } from "../qwen-provider"

const TEST_PROMPT: LanguageModelV2Prompt = [
  { role: "user", content: [{ type: "text", text: "Hello" }] },
]

vi.stubEnv("DASHSCOPE_API_KEY", "test-api-key-123")

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
  let requestBody: any
  let requestHeaders: Record<string, string>
  let responseBody: any
  let responseHeaders: Record<string, string>

  beforeEach(() => {
    requestBody = undefined
    requestHeaders = {}
    responseBody = {
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

  it("should pass user setting to requests", async () => {
    const provider = createTestProvider()
    const modelWithUser = provider("qwen-chat", {
      user: "test-user-id",
    })
    await modelWithUser.doGenerate({
      prompt: TEST_PROMPT,
    })
    expect(requestBody).toMatchObject({
      user: "test-user-id",
    })
  })

  it("should extract text response from content array", async () => {
    responseBody.choices[0].message.content = "Hello, World!"
    const provider = createTestProvider()
    const model = provider("qwen-chat")

    const { content } = await model.doGenerate({
      prompt: TEST_PROMPT,
    })

    expect(content).toContainEqual({
      type: "text",
      text: "Hello, World!",
    })
  })

  it("should extract reasoning content", async () => {
    responseBody.choices[0].message = {
      role: "assistant",
      content: "Hello, World!",
      reasoning_content: "This is the reasoning behind the response",
    }
    const provider = createTestProvider()
    const model = provider("qwen-chat")

    const { content } = await model.doGenerate({
      prompt: TEST_PROMPT,
    })

    expect(content).toContainEqual({
      type: "reasoning",
      text: "This is the reasoning behind the response",
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
    const model = provider("qwen-chat")

    const { usage } = await model.doGenerate({
      prompt: TEST_PROMPT,
    })

    expect(usage).toMatchObject({
      inputTokens: 20,
      outputTokens: 5,
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
    const model = provider("qwen-chat")

    const { response } = await model.doGenerate({
      prompt: TEST_PROMPT,
    })

    expect(response).toMatchObject({
      id: "test-id",
      timestamp: new Date(123 * 1000),
      modelId: "test-model",
    })
  })

  it("should support partial usage", async () => {
    responseBody.usage = { prompt_tokens: 20, total_tokens: 20 }
    const provider = createTestProvider()
    const model = provider("qwen-chat")

    const { usage } = await model.doGenerate({
      prompt: TEST_PROMPT,
    })

    expect(usage).toMatchObject({
      inputTokens: 20,
      outputTokens: undefined,
    })
  })

  it("should extract finish reason", async () => {
    responseBody.choices[0].finish_reason = "stop"
    const provider = createTestProvider()
    const model = provider("qwen-chat")

    const { finishReason } = await model.doGenerate({
      prompt: TEST_PROMPT,
    })

    expect(finishReason).toStrictEqual("stop")
  })

  it("should support unknown finish reason", async () => {
    responseBody.choices[0].finish_reason = "eos"
    const provider = createTestProvider()
    const model = provider("qwen-chat")

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
    const model = provider("qwen-chat")

    const { response } = await model.doGenerate({
      prompt: TEST_PROMPT,
    })

    expect(response?.headers).toMatchObject({
      "content-type": "application/json",
      "test-header": "test-value",
    })
  })

  it("should pass the model and the messages", async () => {
    const provider = createTestProvider()
    const model = provider("qwen-chat")

    await model.doGenerate({
      prompt: TEST_PROMPT,
    })

    expect(requestBody).toStrictEqual({
      model: "qwen-chat",
      messages: [{ role: "user", content: "Hello" }],
    })
  })

  it("should pass settings", async () => {
    const provider = createTestProvider()

    await provider("qwen-chat", {
      user: "test-user-id",
    }).doGenerate({
      prompt: TEST_PROMPT,
    })

    expect(requestBody).toStrictEqual({
      model: "qwen-chat",
      messages: [{ role: "user", content: "Hello" }],
      user: "test-user-id",
    })
  })

  it("should include provider-specific options", async () => {
    const provider = createTestProvider()

    await provider("qwen-chat").doGenerate({
      prompt: TEST_PROMPT,
      providerOptions: {
        qwen: {
          someCustomOption: "test-value",
        },
      },
    })

    expect(requestBody).toStrictEqual({
      model: "qwen-chat",
      messages: [{ role: "user", content: "Hello" }],
      someCustomOption: "test-value",
    })
  })

  it("should not include provider-specific options for different provider", async () => {
    const provider = createTestProvider()

    await provider("qwen-chat").doGenerate({
      prompt: TEST_PROMPT,
      providerOptions: {
        notQwen: {
          someCustomOption: "test-value",
        },
      },
    })

    expect(requestBody).toStrictEqual({
      model: "qwen-chat",
      messages: [{ role: "user", content: "Hello" }],
    })
  })

  it("should pass tools and toolChoice", async () => {
    const provider = createTestProvider()
    const model = provider("qwen-chat")

    await model.doGenerate({
      prompt: TEST_PROMPT,
      tools: [
        {
          type: "function",
          name: "test-tool",
          inputSchema: {
            type: "object",
            properties: { value: { type: "string" } },
            required: ["value"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        },
      ],
      toolChoice: {
        type: "tool",
        toolName: "test-tool",
      },
    })

    expect(requestBody).toStrictEqual({
      model: "qwen-chat",
      messages: [{ role: "user", content: "Hello" }],
      tools: [
        {
          type: "function",
          function: {
            name: "test-tool",
            parameters: {
              type: "object",
              properties: { value: { type: "string" } },
              required: ["value"],
              additionalProperties: false,
              $schema: "http://json-schema.org/draft-07/schema#",
            },
          },
        },
      ],
      tool_choice: {
        type: "function",
        function: { name: "test-tool" },
      },
    })
  })

  it("should include only function tools and warn for provider-defined tools", async () => {
    const provider = createTestProvider()
    const model = provider("qwen-chat")

    const { warnings } = await model.doGenerate({
      prompt: TEST_PROMPT,
      tools: [
        {
          type: "function",
          name: "func-tool",
          inputSchema: {
            type: "object",
            properties: { x: { type: "number" } },
            required: ["x"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        },
        {
          type: "provider-defined",
          id: "qwen.unsupported",
          name: "unsupported",
          args: {},
        },
      ],
    })

    // Request should contain only the function tool
    expect(requestBody).toMatchObject({
      tools: [
        {
          type: "function",
          function: {
            name: "func-tool",
            parameters: expect.any(Object),
          },
        },
      ],
    })

    // Warning for dropped provider-defined tool
    expect(warnings).toEqual([
      {
        type: "unsupported-tool",
        tool: {
          type: "provider-defined",
          id: "qwen.unsupported",
          name: "unsupported",
          args: {},
        },
      },
    ])
  })

  it("should pass headers", async () => {
    const provider = createTestProvider({
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

    expect(requestHeaders).toMatchObject({
      "authorization": "Bearer test-api-key",
      "content-type": "application/json",
      "custom-provider-header": "provider-header-value",
      "custom-request-header": "request-header-value",
    })
  })

  it("should parse tool results in content array", async () => {
    responseBody.choices[0].message = {
      role: "assistant",
      content: "",
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
    }
    const provider = createTestProvider()
    const model = provider("qwen-chat")

    const { content } = await model.doGenerate({
      prompt: TEST_PROMPT,
      tools: [
        {
          type: "function",
          name: "test-tool",
          inputSchema: {
            type: "object",
            properties: { value: { type: "string" } },
            required: ["value"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        },
      ],
      toolChoice: {
        type: "tool",
        toolName: "test-tool",
      },
    })

    expect(content).toContainEqual({
      type: "tool-call",
      toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
      toolName: "test-tool",
      input: "{\"value\":\"Spark\"}",
    })
  })

  describe("response format", () => {
    it("should not send a response_format when response format is text", async () => {
      responseBody.choices[0].message.content = "{\"value\":\"Spark\"}"
      createTestProvider()
      const model = new QwenChatLanguageModel(
        "qwen-plus",
        {},
        {
          provider: "qwen",
          url: () => "https://my.api.com/v1/chat/completions",
          headers: () => ({}),
          supportsStructuredOutputs: false,
          fetch: async (_url, init) => {
            if (init?.body) {
              requestBody = JSON.parse(init.body as string)
            }
            return new Response(JSON.stringify(responseBody), {
              headers: responseHeaders,
            })
          },
        },
      )

      await model.doGenerate({
        prompt: TEST_PROMPT,
        responseFormat: { type: "text" },
      })

      expect(requestBody).toStrictEqual({
        model: "qwen-plus",
        messages: [{ role: "user", content: "Hello" }],
      })
    })

    it("should forward json response format as \"json_object\" without schema", async () => {
      responseBody.choices[0].message.content = "{\"value\":\"Spark\"}"
      const provider = createTestProvider()
      const model = provider("qwen-plus")

      await model.doGenerate({
        prompt: TEST_PROMPT,
        responseFormat: { type: "json" },
      })

      expect(requestBody).toStrictEqual({
        model: "qwen-plus",
        messages: [{ role: "user", content: "Hello" }],
        response_format: { type: "json_object" },
      })
    })

    it("should forward json response format as \"json_object\" and omit schema when structuredOutputs are disabled", async () => {
      responseBody.choices[0].message.content = "{\"value\":\"Spark\"}"
      createTestProvider()
      const model = new QwenChatLanguageModel(
        "qwen-plus",
        {},
        {
          provider: "qwen",
          url: () => "https://my.api.com/v1/chat/completions",
          headers: () => ({}),
          supportsStructuredOutputs: false,
          fetch: async (_url, init) => {
            if (init?.body) {
              requestBody = JSON.parse(init.body as string)
            }
            return new Response(JSON.stringify(responseBody), {
              headers: responseHeaders,
            })
          },
        },
      )

      const { warnings } = await model.doGenerate({
        prompt: TEST_PROMPT,
        responseFormat: {
          type: "json",
          schema: {
            type: "object",
            properties: { value: { type: "string" } },
            required: ["value"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        },
      })

      expect(requestBody).toStrictEqual({
        model: "qwen-plus",
        messages: [{ role: "user", content: "Hello" }],
        response_format: { type: "json_object" },
      })

      expect(warnings).toEqual([
        {
          details:
            "JSON response format schema is only supported with structuredOutputs",
          setting: "responseFormat",
          type: "unsupported-setting",
        },
      ])
    })

    it("should forward json response format as \"json_schema\" and include schema when structuredOutputs are enabled", async () => {
      responseBody.choices[0].message.content = "{\"value\":\"Spark\"}"
      createTestProvider()
      const model = new QwenChatLanguageModel(
        "qwen-plus",
        {},
        {
          provider: "qwen",
          url: () => "https://my.api.com/v1/chat/completions",
          headers: () => ({}),
          supportsStructuredOutputs: true,
          fetch: async (_url, init) => {
            if (init?.body) {
              requestBody = JSON.parse(init.body as string)
            }
            return new Response(JSON.stringify(responseBody), {
              headers: responseHeaders,
            })
          },
        },
      )

      const { warnings } = await model.doGenerate({
        prompt: TEST_PROMPT,
        responseFormat: {
          type: "json",
          schema: {
            type: "object",
            properties: { value: { type: "string" } },
            required: ["value"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        },
      })

      expect(requestBody).toStrictEqual({
        model: "qwen-plus",
        messages: [{ role: "user", content: "Hello" }],
        response_format: {
          type: "json_schema",
          json_schema: {
            name: "response",
            schema: {
              type: "object",
              properties: { value: { type: "string" } },
              required: ["value"],
              additionalProperties: false,
              $schema: "http://json-schema.org/draft-07/schema#",
            },
          },
        },
      })

      expect(warnings).toEqual([])
    })

    it("should set name & description with json schema when structuredOutputs are enabled", async () => {
      responseBody.choices[0].message.content = "{\"value\":\"Spark\"}"
      createTestProvider()
      const model = new QwenChatLanguageModel(
        "qwen-plus",
        {},
        {
          provider: "qwen",
          url: () => "https://my.api.com/v1/chat/completions",
          headers: () => ({}),
          supportsStructuredOutputs: true,
          fetch: async (_url, init) => {
            if (init?.body) {
              requestBody = JSON.parse(init.body as string)
            }
            return new Response(JSON.stringify(responseBody), {
              headers: responseHeaders,
            })
          },
        },
      )

      await model.doGenerate({
        prompt: TEST_PROMPT,
        responseFormat: {
          type: "json",
          name: "test-name",
          description: "test description",
          schema: {
            type: "object",
            properties: { value: { type: "string" } },
            required: ["value"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        },
      })

      expect(requestBody).toStrictEqual({
        model: "qwen-plus",
        messages: [{ role: "user", content: "Hello" }],
        response_format: {
          type: "json_schema",
          json_schema: {
            name: "test-name",
            description: "test description",
            schema: {
              type: "object",
              properties: { value: { type: "string" } },
              required: ["value"],
              additionalProperties: false,
              $schema: "http://json-schema.org/draft-07/schema#",
            },
          },
        },
      })
    })

    it("should allow for json without schema when structuredOutputs are enabled", async () => {
      responseBody.choices[0].message.content = "{\"value\":\"Spark\"}"
      createTestProvider()
      const model = new QwenChatLanguageModel(
        "qwen-plus",
        {},
        {
          provider: "qwen",
          url: () => "https://my.api.com/v1/chat/completions",
          headers: () => ({}),
          supportsStructuredOutputs: true,
          fetch: async (_url, init) => {
            if (init?.body) {
              requestBody = JSON.parse(init.body as string)
            }
            return new Response(JSON.stringify(responseBody), {
              headers: responseHeaders,
            })
          },
        },
      )

      await model.doGenerate({
        prompt: TEST_PROMPT,
        responseFormat: {
          type: "json",
        },
      })

      expect(requestBody).toStrictEqual({
        model: "qwen-plus",
        messages: [{ role: "user", content: "Hello" }],
        response_format: {
          type: "json_object",
        },
      })
    })
  })

  it("should send request body", async () => {
    const provider = createTestProvider()
    const model = provider("qwen-chat")

    const { request } = await model.doGenerate({
      prompt: TEST_PROMPT,
    })

    expect(request).toMatchObject({
      body: "{\"model\":\"qwen-chat\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}",
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
      `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1702657020,"model":"qwen-chat",`
      + `"system_fingerprint":null,"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n`,
      `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1702657020,"model":"qwen-chat",`
      + `"system_fingerprint":null,"choices":[{"index":1,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n`,
      `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1702657020,"model":"qwen-chat",`
      + `"system_fingerprint":null,"choices":[{"index":1,"delta":{"content":", "},"finish_reason":null}]}\n\n`,
      `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1702657020,"model":"qwen-chat",`
      + `"system_fingerprint":null,"choices":[{"index":1,"delta":{"content":"World!"},"finish_reason":null}]}\n\n`,
      `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1702657020,"model":"qwen-chat",`
      + `"system_fingerprint":null,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n`,
      `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1729171479,"model":"qwen-chat",`
      + `"system_fingerprint":"fp_10c08bf97d","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],`
      + `"usage":{"queue_time":0.061348671,"prompt_tokens":18,"prompt_time":0.000211569,`
      + `"completion_tokens":439,"completion_time":0.798181818,"total_tokens":457,"total_time":0.798393387}}\n\n`,
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

  it("should stream text deltas with V2 format", async () => {
    const provider = createStreamingTestProvider()
    const model = provider("qwen-chat")

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
    })

    const parts = await convertReadableStreamToArray(stream)

    expect(parts).toContainEqual({
      type: "stream-start",
      warnings: [],
    })
    expect(parts).toContainEqual({
      type: "response-metadata",
      id: "chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798",
      modelId: "qwen-chat",
      timestamp: new Date("2023-12-15T16:17:00.000Z"),
    })
    expect(parts.filter(p => p.type === "text-start")).toHaveLength(1)
    expect(
      parts.filter(p => p.type === "text-delta" && p.delta === ""),
    ).toHaveLength(1)
    expect(
      parts.filter(p => p.type === "text-delta" && p.delta === "Hello"),
    ).toHaveLength(1)
    expect(
      parts.filter(p => p.type === "text-delta" && p.delta === ", "),
    ).toHaveLength(1)
    expect(
      parts.filter(p => p.type === "text-delta" && p.delta === "World!"),
    ).toHaveLength(1)
    expect(parts.filter(p => p.type === "text-end")).toHaveLength(1)
    expect(parts).toContainEqual({
      type: "finish",
      finishReason: "stop",
      usage: { inputTokens: 18, outputTokens: 439, totalTokens: 457 },
    })
  })

  it("should emit error parts when server sends error objects", async () => {
    responseChunks = [
      `data: {"object":"error","message":"boom","type":"bad_request","param":null,"code":null}\n\n`,
      "data: [DONE]\n\n",
    ]

    const provider = createStreamingTestProvider()
    const model = provider("qwen-chat")
    const { stream } = await model.doStream({ prompt: TEST_PROMPT })
    const parts = await convertReadableStreamToArray(stream)

    expect(parts).toContainEqual({ type: "error", error: "boom" })
  })

  it("should handle chunks with null delta by emitting only metadata/finish", async () => {
    responseChunks = [
      `data: {"id":"abc","object":"chat.completion.chunk","created":1702657020,"model":"qwen-chat","choices":[{"index":0,"delta":null,"finish_reason":null}]}\n\n`,
      `data: {"id":"abc","object":"chat.completion.chunk","created":1702657021,"model":"qwen-chat","choices":[{"index":0,"delta":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2}}\n\n`,
      "data: [DONE]\n\n",
    ]

    const provider = createStreamingTestProvider()
    const model = provider("qwen-chat")
    const { stream } = await model.doStream({ prompt: TEST_PROMPT })
    const parts = await convertReadableStreamToArray(stream)

    // Should have stream-start, response-metadata, finish; no text-delta/tool events
    expect(parts.find(p => p.type === "response-metadata")).toBeTruthy()
    expect(parts.find(p => p.type === "finish")).toBeTruthy()
    expect(parts.find(p => p.type === "text-delta")).toBeFalsy()
    expect(parts.find(p => p.type === "tool-call")).toBeFalsy()
  })

  it("should throw InvalidResponseDataError when tool_calls delta has invalid type", async () => {
    responseChunks = [
      `data: {"id":"abc","object":"chat.completion.chunk","created":1702657020,"model":"qwen-chat","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"id1","type":null,"function":{"name":"t","arguments":"{}"}}]},"finish_reason":null}]}\n\n`,
      "data: [DONE]\n\n",
    ]

    const provider = createStreamingTestProvider()
    const model = provider("qwen-chat")
    const { stream } = await model.doStream({ prompt: TEST_PROMPT })
    await expect(async () => await convertReadableStreamToArray(stream)).rejects.toThrow()
  })

  it("should throw InvalidResponseDataError when tool_calls delta missing id", async () => {
    responseChunks = [
      `data: {"id":"abc","object":"chat.completion.chunk","created":1702657020,"model":"qwen-chat","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"type":"function","function":{"name":"t","arguments":"{}"}}]},"finish_reason":null}]}\n\n`,
      "data: [DONE]\n\n",
    ]

    const provider = createStreamingTestProvider()
    const model = provider("qwen-chat")
    const { stream } = await model.doStream({ prompt: TEST_PROMPT })
    await expect(async () => await convertReadableStreamToArray(stream)).rejects.toThrow()
  })

  it("should throw InvalidResponseDataError when tool_calls delta missing function.name", async () => {
    responseChunks = [
      `data: {"id":"abc","object":"chat.completion.chunk","created":1702657020,"model":"qwen-chat","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"id1","type":"function","function":{}}]},"finish_reason":null}]}\n\n`,
      "data: [DONE]\n\n",
    ]

    const provider = createStreamingTestProvider()
    const model = provider("qwen-chat")
    const { stream } = await model.doStream({ prompt: TEST_PROMPT })
    await expect(async () => await convertReadableStreamToArray(stream)).rejects.toThrow()
  })

  it("should stream reasoning content before text deltas", async () => {
    responseChunks = [
      `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
      + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Let me think"},"finish_reason":null}]}\n\n`,
      `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
      + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"reasoning_content":" about this"},"finish_reason":null}]}\n\n`,
      `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
      + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"content":"Here's"},"finish_reason":null}]}\n\n`,
      `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
      + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"content":" my response"},"finish_reason":null}]}\n\n`,
      `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1729171479,"model":"qwen-chat",`
      + `"system_fingerprint":"fp_10c08bf97d","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],`
      + `"usage":{"prompt_tokens":18,"completion_tokens":439}}\n\n`,
      "data: [DONE]\n\n",
    ]

    const provider = createStreamingTestProvider()
    const model = provider("qwen-chat")

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
    })

    const parts = await convertReadableStreamToArray(stream)

    expect(parts).toContainEqual({
      type: "response-metadata",
      id: "chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798",
      modelId: "qwen-chat",
      timestamp: new Date("2024-03-25T09:06:38.000Z"),
    })
    const reasoningStartIndex = parts.findIndex(
      p => p.type === "reasoning-start",
    )
    const reasoningDelta1 = parts.find(
      p => p.type === "reasoning-delta" && p.delta === "Let me think",
    )
    const reasoningDelta2 = parts.find(
      p => p.type === "reasoning-delta" && p.delta === " about this",
    )
    const textDelta1 = parts.find(
      p => p.type === "text-delta" && p.delta === "Here's",
    )
    const textDelta2 = parts.find(
      p => p.type === "text-delta" && p.delta === " my response",
    )

    expect(reasoningStartIndex).toBeGreaterThan(-1)
    expect(reasoningDelta1).toBeDefined()
    expect(reasoningDelta2).toBeDefined()
    expect(textDelta1).toBeDefined()
    expect(textDelta2).toBeDefined()

    expect(parts).toContainEqual({
      type: "finish",
      finishReason: "stop",
      usage: { inputTokens: 18, outputTokens: 439, totalTokens: 457 },
    })
  })

  it("should stream tool deltas with V2 format", async () => {
    responseChunks = [
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
    ]

    const provider = createStreamingTestProvider()
    const model = provider("qwen-chat")

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
      tools: [
        {
          type: "function",
          name: "test-tool",
          inputSchema: {
            type: "object",
            properties: { value: { type: "string" } },
            required: ["value"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        },
      ],
    })

    const parts = await convertReadableStreamToArray(stream)

    expect(parts).toContainEqual({
      type: "response-metadata",
      id: "chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798",
      modelId: "qwen-chat",
      timestamp: new Date("2024-03-25T09:06:38.000Z"),
    })
    expect(parts.filter(p => p.type === "tool-input-start")).toHaveLength(1)
    expect(parts.filter(p => p.type === "tool-input-delta")).toHaveLength(7)
    expect(parts.filter(p => p.type === "tool-input-end")).toHaveLength(1)
    expect(parts).toContainEqual({
      type: "tool-call",
      toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
      toolName: "test-tool",
      input: "{\"value\":\"Sparkle Day\"}",
    })
    expect(parts).toContainEqual({
      type: "finish",
      finishReason: "tool-calls",
      usage: { inputTokens: 18, outputTokens: 439, totalTokens: 457 },
    })
  })

  it("should stream tool call deltas when tool call arguments are passed in the first chunk", async () => {
    responseChunks = [
      `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
      + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"role":"assistant","content":null,`
      + `"tool_calls":[{"index":0,"id":"call_O17Uplv4lJvD6DVdIvFFeRMw","type":"function","function":{"name":"test-tool","arguments":"{\\""}}]},`
      + `"finish_reason":null}]}\n\n`,
      `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
      + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"va"}}]},`
      + `"finish_reason":null}]}\n\n`,
      `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
      + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"lue"}}]},`
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
    ]

    const provider = createStreamingTestProvider()
    const model = provider("qwen-chat")

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
      tools: [
        {
          type: "function",
          name: "test-tool",
          inputSchema: {
            type: "object",
            properties: { value: { type: "string" } },
            required: ["value"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        },
      ],
    })

    const parts = await convertReadableStreamToArray(stream)

    expect(parts.filter(p => p.type === "tool-input-start")).toHaveLength(1)
    expect(parts.filter(p => p.type === "tool-input-delta")).toHaveLength(8)
    expect(parts.filter(p => p.type === "tool-input-end")).toHaveLength(1)
    expect(parts).toContainEqual({
      type: "tool-call",
      toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
      toolName: "test-tool",
      input: "{\"value\":\"Sparkle Day\"}",
    })
    expect(parts).toContainEqual({
      type: "finish",
      finishReason: "tool-calls",
      usage: { inputTokens: 18, outputTokens: 439, totalTokens: 457 },
    })
  })

  it("should not duplicate tool calls when there is an additional empty chunk after the tool call has been completed", async () => {
    responseChunks = [
      `data: {"id":"chat-2267f7e2910a4254bac0650ba74cfc1c","object":"chat.completion.chunk","created":1733162241,`
      + `"model":"meta/llama-3.1-8b-instruct:fp8","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}],`
      + `"usage":{"prompt_tokens":226,"total_tokens":226,"completion_tokens":0}}\n\n`,
      `data: {"id":"chat-2267f7e2910a4254bac0650ba74cfc1c","object":"chat.completion.chunk","created":1733162241,`
      + `"model":"meta/llama-3.1-8b-instruct:fp8","choices":[{"index":0,"delta":{"tool_calls":[{"id":"chatcmpl-tool-b3b307239370432d9910d4b79b4dbbaa",`
      + `"type":"function","index":0,"function":{"name":"searchGoogle"}}]},"logprobs":null,"finish_reason":null}],`
      + `"usage":{"prompt_tokens":226,"total_tokens":233,"completion_tokens":7}}\n\n`,
      `data: {"id":"chat-2267f7e2910a4254bac0650ba74cfc1c","object":"chat.completion.chunk","created":1733162241,`
      + `"model":"meta/llama-3.1-8b-instruct:fp8","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,`
      + `"function":{"arguments":"{\\"query\\": \\""}}]},"logprobs":null,"finish_reason":null}],`
      + `"usage":{"prompt_tokens":226,"total_tokens":241,"completion_tokens":15}}\n\n`,
      `data: {"id":"chat-2267f7e2910a4254bac0650ba74cfc1c","object":"chat.completion.chunk","created":1733162241,`
      + `"model":"meta/llama-3.1-8b-instruct:fp8","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,`
      + `"function":{"arguments":"latest"}}]},"logprobs":null,"finish_reason":null}],`
      + `"usage":{"prompt_tokens":226,"total_tokens":242,"completion_tokens":16}}\n\n`,
      `data: {"id":"chat-2267f7e2910a4254bac0650ba74cfc1c","object":"chat.completion.chunk","created":1733162241,`
      + `"model":"meta/llama-3.1-8b-instruct:fp8","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,`
      + `"function":{"arguments":" news"}}]},"logprobs":null,"finish_reason":null}],`
      + `"usage":{"prompt_tokens":226,"total_tokens":243,"completion_tokens":17}}\n\n`,
      `data: {"id":"chat-2267f7e2910a4254bac0650ba74cfc1c","object":"chat.completion.chunk","created":1733162241,`
      + `"model":"meta/llama-3.1-8b-instruct:fp8","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,`
      + `"function":{"arguments":" on"}}]},"logprobs":null,"finish_reason":null}],`
      + `"usage":{"prompt_tokens":226,"total_tokens":244,"completion_tokens":18}}\n\n`,
      `data: {"id":"chat-2267f7e2910a4254bac0650ba74cfc1c","object":"chat.completion.chunk","created":1733162241,`
      + `"model":"meta/llama-3.1-8b-instruct:fp8","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,`
      + `"function":{"arguments":" ai\\"}"}}]},"logprobs":null,"finish_reason":null}],`
      + `"usage":{"prompt_tokens":226,"total_tokens":245,"completion_tokens":19}}\n\n`,
      // empty arguments chunk after the tool call has already been finished:
      `data: {"id":"chat-2267f7e2910a4254bac0650ba74cfc1c","object":"chat.completion.chunk","created":1733162241,`
      + `"model":"meta/llama-3.1-8b-instruct:fp8","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,`
      + `"function":{"arguments":""}}]},"logprobs":null,"finish_reason":"tool_calls","stop_reason":128008}],`
      + `"usage":{"prompt_tokens":226,"total_tokens":246,"completion_tokens":20}}\n\n`,
      `data: {"id":"chat-2267f7e2910a4254bac0650ba74cfc1c","object":"chat.completion.chunk","created":1733162241,`
      + `"model":"meta/llama-3.1-8b-instruct:fp8","choices":[],`
      + `"usage":{"prompt_tokens":226,"total_tokens":246,"completion_tokens":20}}\n\n`,
      `data: [DONE]\n\n`,
    ]

    const provider = createStreamingTestProvider()
    const model = provider("qwen-chat")

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
      tools: [
        {
          type: "function",
          name: "searchGoogle",
          inputSchema: {
            type: "object",
            properties: { query: { type: "string" } },
            required: ["query"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        },
      ],
    })

    const parts = await convertReadableStreamToArray(stream)

    // Should only have one tool call
    const toolCalls = parts.filter(p => p.type === "tool-call")
    expect(toolCalls).toHaveLength(1)
    expect(toolCalls[0]).toMatchObject({
      type: "tool-call",
      toolCallId: "chatcmpl-tool-b3b307239370432d9910d4b79b4dbbaa",
      toolName: "searchGoogle",
      input: "{\"query\": \"latest news on ai\"}",
    })
  })

  it("should stream tool call that is sent in one chunk", async () => {
    responseChunks = [
      `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1711357598,"model":"qwen-chat",`
      + `"system_fingerprint":"fp_3bc1b5746c","choices":[{"index":0,"delta":{"role":"assistant","content":null,`
      + `"tool_calls":[{"index":0,"id":"call_O17Uplv4lJvD6DVdIvFFeRMw","type":"function","function":{"name":"test-tool","arguments":"{\\"value\\":\\"Sparkle Day\\"}"}}]},`
      + `"finish_reason":null}]}\n\n`,
      `data: {"id":"chatcmpl-e7f8e220-656c-4455-a132-dacfc1370798","object":"chat.completion.chunk","created":1729171479,"model":"qwen-chat",`
      + `"system_fingerprint":"fp_10c08bf97d","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],`
      + `"usage":{"queue_time":0.061348671,"prompt_tokens":18,"prompt_time":0.000211569,`
      + `"completion_tokens":439,"completion_time":0.798181818,"total_tokens":457,"total_time":0.798393387}}\n\n`,
      "data: [DONE]\n\n",
    ]

    const provider = createStreamingTestProvider()
    const model = provider("qwen-chat")

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
      tools: [
        {
          type: "function",
          name: "test-tool",
          inputSchema: {
            type: "object",
            properties: { value: { type: "string" } },
            required: ["value"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        },
      ],
    })

    const parts = await convertReadableStreamToArray(stream)

    expect(parts.filter(p => p.type === "tool-input-start")).toHaveLength(1)
    expect(parts.filter(p => p.type === "tool-input-delta")).toHaveLength(1)
    expect(parts.filter(p => p.type === "tool-input-end")).toHaveLength(1)
    expect(parts).toContainEqual({
      type: "tool-call",
      toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
      toolName: "test-tool",
      input: "{\"value\":\"Sparkle Day\"}",
    })
    expect(parts).toContainEqual({
      type: "finish",
      finishReason: "tool-calls",
      usage: { inputTokens: 18, outputTokens: 439, totalTokens: 457 },
    })
  })

  it("should handle unparsable stream parts", async () => {
    responseChunks = [`data: {unparsable}\n\n`, "data: [DONE]\n\n"]

    const provider = createStreamingTestProvider()
    const model = provider("qwen-chat")

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
    })

    const elements = await convertReadableStreamToArray(stream)

    expect(elements.find(e => e.type === "error")).toBeDefined()
    expect(elements.find(e => e.type === "finish")).toMatchObject({
      finishReason: "unknown",
      type: "finish",
    })
  })

  it("should expose the raw response headers", async () => {
    responseHeaders = {
      ...responseHeaders,
      "test-header": "test-value",
    }

    const provider = createStreamingTestProvider()
    const model = provider("qwen-chat")

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

  it("should pass the messages and the model", async () => {
    const provider = createStreamingTestProvider()
    const model = provider("qwen-chat")

    await model.doStream({
      prompt: TEST_PROMPT,
    })

    expect(requestBody).toStrictEqual({
      stream: true,
      stream_options: {
        include_usage: true,
      },
      model: "qwen-chat",
      messages: [{ role: "user", content: "Hello" }],
    })
  })

  it("should pass headers", async () => {
    const provider = createStreamingTestProvider({
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

    expect(requestHeaders).toMatchObject({
      "authorization": "Bearer test-api-key",
      "content-type": "application/json",
      "custom-provider-header": "provider-header-value",
      "custom-request-header": "request-header-value",
    })
  })

  it("should include provider-specific options", async () => {
    const provider = createStreamingTestProvider()

    await provider("qwen-chat").doStream({
      prompt: TEST_PROMPT,
      providerOptions: {
        qwen: {
          someCustomOption: "test-value",
        },
      },
    })

    expect(requestBody).toStrictEqual({
      stream: true,
      stream_options: {
        include_usage: true,
      },
      model: "qwen-chat",
      messages: [{ role: "user", content: "Hello" }],
      someCustomOption: "test-value",
    })
  })

  it("should not include provider-specific options for different provider", async () => {
    const provider = createStreamingTestProvider()

    await provider("qwen-chat").doStream({
      prompt: TEST_PROMPT,
      providerOptions: {
        notQwen: {
          someCustomOption: "test-value",
        },
      },
    })

    expect(requestBody).toStrictEqual({
      stream: true,
      stream_options: {
        include_usage: true,
      },
      model: "qwen-chat",
      messages: [{ role: "user", content: "Hello" }],
    })
  })

  it("should send request body", async () => {
    const provider = createStreamingTestProvider()
    const model = provider("qwen-chat")

    const { request } = await model.doStream({
      prompt: TEST_PROMPT,
    })

    expect(request).toMatchObject({
      body: "{\"model\":\"qwen-chat\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"stream\":true,\"stream_options\":{\"include_usage\":true}}",
    })
  })
})

describe("doStream simulated streaming", () => {
  let responseBody: any
  let responseHeaders: Record<string, string>

  beforeEach(() => {
    responseBody = {
      id: "chatcmpl-95ZTZkhr0mHNKqerQfiwkuox3PHAd",
      object: "chat.completion",
      created: 1711115037,
      model: "o1-preview",
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
      fetch: async (_url, _init) => {
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

  it("should stream text delta", async () => {
    responseBody.choices[0].message.content = "Hello, World!"
    const provider = createTestProvider()
    const model = provider.chatModel("o1", {
      simulateStreaming: true,
    })

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
    })

    const parts = await convertReadableStreamToArray(stream)

    expect(parts).toContainEqual({
      type: "stream-start",
      warnings: [],
    })
    expect(parts).toContainEqual({
      type: "response-metadata",
      id: "chatcmpl-95ZTZkhr0mHNKqerQfiwkuox3PHAd",
      modelId: "o1-preview",
      timestamp: expect.any(Date),
    })
    expect(parts.filter(p => p.type === "text-start")).toHaveLength(1)
    expect(
      parts.filter(
        p => p.type === "text-delta" && p.delta === "Hello, World!",
      ),
    ).toHaveLength(1)
    expect(parts.filter(p => p.type === "text-end")).toHaveLength(1)
    expect(parts).toContainEqual({
      type: "finish",
      finishReason: "stop",
      usage: { inputTokens: 4, outputTokens: 30, totalTokens: 34 },
      providerMetadata: undefined,
    })
  })

  it("should stream reasoning content before text delta in simulated streaming", async () => {
    responseBody.choices[0].message = {
      role: "assistant",
      content: "Hello, World!",
      reasoning_content: "This is the reasoning",
    }
    const provider = createTestProvider()
    const model = provider.chatModel("o1", {
      simulateStreaming: true,
    })

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
    })

    const parts = await convertReadableStreamToArray(stream)

    expect(parts.filter(p => p.type === "reasoning-start")).toHaveLength(1)
    expect(
      parts.filter(
        p =>
          p.type === "reasoning-delta" && p.delta === "This is the reasoning",
      ),
    ).toHaveLength(1)
    expect(parts.filter(p => p.type === "reasoning-end")).toHaveLength(1)
    expect(parts.filter(p => p.type === "text-start")).toHaveLength(1)
    expect(
      parts.filter(
        p => p.type === "text-delta" && p.delta === "Hello, World!",
      ),
    ).toHaveLength(1)
    expect(parts.filter(p => p.type === "text-end")).toHaveLength(1)
    expect(parts).toContainEqual({
      type: "finish",
      finishReason: "stop",
      usage: { inputTokens: 4, outputTokens: 30, totalTokens: 34 },
      providerMetadata: undefined,
    })
  })

  it("should stream tool calls", async () => {
    responseBody.choices[0].message = {
      role: "assistant",
      content: "",
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
    }
    const provider = createTestProvider()
    const model = provider.chatModel("o1", {
      simulateStreaming: true,
    })

    const { stream } = await model.doStream({
      prompt: TEST_PROMPT,
      tools: [
        {
          type: "function",
          name: "test-tool",
          inputSchema: {
            type: "object",
            properties: { value: { type: "string" } },
            required: ["value"],
            additionalProperties: false,
            $schema: "http://json-schema.org/draft-07/schema#",
          },
        },
      ],
    })

    const parts = await convertReadableStreamToArray(stream)

    expect(parts).toContainEqual({
      type: "response-metadata",
      id: "chatcmpl-95ZTZkhr0mHNKqerQfiwkuox3PHAd",
      modelId: "o1-preview",
      timestamp: expect.any(Date),
    })
    expect(parts).toContainEqual({
      type: "tool-call",
      toolCallId: "call_O17Uplv4lJvD6DVdIvFFeRMw",
      toolName: "test-tool",
      input: "{\"value\":\"Sparkle Day\"}",
    })
    expect(parts).toContainEqual({
      type: "finish",
      finishReason: "stop",
      usage: { inputTokens: 4, outputTokens: 30, totalTokens: 34 },
      providerMetadata: undefined,
    })
  })
})

describe("metadata extraction", () => {
  const testMetadataExtractor = {
    extractMetadata: ({ parsedBody }: { parsedBody: unknown }) => {
      if (
        typeof parsedBody !== "object"
        || !parsedBody
        || !("test_field" in parsedBody)
      ) {
        return undefined
      }
      return {
        test: {
          value: parsedBody.test_field as string,
        },
      }
    },
    createStreamExtractor: () => {
      let accumulatedValue: string | undefined

      return {
        processChunk: (chunk: unknown) => {
          if (
            typeof chunk === "object"
            && chunk
            && "choices" in chunk
            && Array.isArray(chunk.choices)
            && chunk.choices[0]?.finish_reason === "stop"
            && "test_field" in chunk
          ) {
            accumulatedValue = chunk.test_field as string
          }
        },
        buildMetadata: () =>
          accumulatedValue
            ? {
                test: {
                  value: accumulatedValue,
                },
              }
            : undefined,
      }
    },
  }

  describe("non-streaming", () => {
    let requestBody: any
    let responseBody: any
    let responseHeaders: Record<string, string>

    beforeEach(() => {
      requestBody = undefined
      responseBody = {
        id: "chatcmpl-123",
        object: "chat.completion",
        created: 1711115037,
        model: "qwen-plus",
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: "Hello",
            },
            finish_reason: "stop",
          },
        ],
        test_field: "test_value",
      }
      responseHeaders = {
        "content-type": "application/json",
      }
    })

    it("should process metadata from complete response", async () => {
      const model = new QwenChatLanguageModel(
        "qwen-plus",
        {},
        {
          provider: "qwen",
          url: () => "https://my.api.com/v1/chat/completions",
          headers: () => ({}),
          metadataExtractor: testMetadataExtractor,
          fetch: async (_url, init) => {
            if (init?.body) {
              requestBody = JSON.parse(init.body as string)
            }
            return new Response(JSON.stringify(responseBody), {
              headers: responseHeaders,
            })
          },
        },
      )

      const result = await model.doGenerate({
        prompt: TEST_PROMPT,
      })

      expect(result.providerMetadata).toEqual({
        test: {
          value: "test_value",
        },
      })

      expect(requestBody).toStrictEqual({
        model: "qwen-plus",
        messages: [{ role: "user", content: "Hello" }],
      })
    })
  })

  describe("streaming", () => {
    let requestBody: any
    let responseChunks: string[]
    let responseHeaders: Record<string, string>

    beforeEach(() => {
      requestBody = undefined
      responseChunks = [
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
        "data: {\"choices\":[{\"finish_reason\":\"stop\"}],\"test_field\":\"test_value\"}\n\n",
        "data: [DONE]\n\n",
      ]
      responseHeaders = {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
        "connection": "keep-alive",
      }
    })

    it("should process metadata from streaming response", async () => {
      const model = new QwenChatLanguageModel(
        "qwen-plus",
        {},
        {
          provider: "qwen",
          url: () => "https://my.api.com/v1/chat/completions",
          headers: () => ({}),
          metadataExtractor: testMetadataExtractor,
          fetch: async (_url, init) => {
            if (init?.body) {
              requestBody = JSON.parse(init.body as string)
            }

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
        },
      )

      const result = await model.doStream({
        prompt: TEST_PROMPT,
      })

      const parts = await convertReadableStreamToArray(result.stream)
      const finishPart = parts.find(part => part.type === "finish")

      expect(finishPart?.providerMetadata).toEqual({
        test: {
          value: "test_value",
        },
      })

      expect(requestBody).toStrictEqual({
        model: "qwen-plus",
        messages: [{ role: "user", content: "Hello" }],
        stream: true,
        stream_options: {
          include_usage: true,
        },
      })
    })
  })
})
