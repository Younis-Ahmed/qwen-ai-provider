import { TooManyEmbeddingValuesForCallError } from "@ai-sdk/provider"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { createQwen } from "../qwen-provider"

const dummyEmbeddings = [
  [0.1, 0.2, 0.3, 0.4, 0.5],
  [0.6, 0.7, 0.8, 0.9, 1.0],
]

vi.stubEnv("DASHSCOPE_API_KEY", "test-api-key-123")

const testValues = ["sunny day at the beach", "rainy day in the city"]

describe("doEmbed", () => {
  let requestBody: any
  let requestHeaders: Record<string, string>
  let responseBody: any
  let responseHeaders: Record<string, string>

  beforeEach(() => {
    requestBody = undefined
    requestHeaders = {}
    responseBody = {
      object: "list",
      data: dummyEmbeddings.map((embedding, i) => ({
        object: "embedding",
        index: i,
        embedding,
      })),
      model: "text-embedding-3-large",
      usage: { prompt_tokens: 8, total_tokens: 8 },
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

  it("should extract embedding", async () => {
    const provider = createTestProvider()
    const model = provider.textEmbeddingModel("text-embedding-3-large")

    const { embeddings } = await model.doEmbed({ values: testValues })

    expect(embeddings).toStrictEqual(dummyEmbeddings)
  })

  it("should expose the raw response headers", async () => {
    responseHeaders = {
      ...responseHeaders,
      "test-header": "test-value",
    }

    const provider = createTestProvider()
    const model = provider.textEmbeddingModel("text-embedding-3-large")

    const { response } = await model.doEmbed({ values: testValues })

    expect(response?.headers).toMatchObject({
      "content-type": "application/json",
      "test-header": "test-value",
    })
  })

  it("should extract usage", async () => {
    responseBody = {
      object: "list",
      data: dummyEmbeddings.map((embedding, i) => ({
        object: "embedding",
        index: i,
        embedding,
      })),
      model: "text-embedding-3-large",
      usage: { prompt_tokens: 20, total_tokens: 20 },
    }

    const provider = createTestProvider()
    const model = provider.textEmbeddingModel("text-embedding-3-large")

    const { usage } = await model.doEmbed({ values: testValues })

    expect(usage).toStrictEqual({ tokens: 20 })
  })

  it("should pass the model and the values", async () => {
    const provider = createTestProvider()
    const model = provider.textEmbeddingModel("text-embedding-3-large")

    await model.doEmbed({ values: testValues })

    expect(requestBody).toStrictEqual({
      model: "text-embedding-3-large",
      input: testValues,
      encoding_format: "float",
    })
  })

  it("should pass the dimensions setting", async () => {
    const provider = createTestProvider()

    await provider
      .textEmbeddingModel("text-embedding-3-large", { dimensions: 64 })
      .doEmbed({ values: testValues })

    expect(requestBody).toStrictEqual({
      model: "text-embedding-3-large",
      input: testValues,
      encoding_format: "float",
      dimensions: 64,
    })
  })

  it("should pass headers", async () => {
    const provider = createTestProvider({
      headers: {
        "Authorization": `Bearer test-api-key`,
        "Custom-Provider-Header": "provider-header-value",
      },
    })

    await provider.textEmbeddingModel("text-embedding-3-large").doEmbed({
      values: testValues,
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

  it("should throw TooManyEmbeddingValuesForCallError when values exceed model limit", async () => {
    const provider = createTestProvider()
    const model = provider.textEmbeddingModel("text-embedding-3-large")

    const many = Array.from({ length: 2049 }, (_, i) => `v${i}`)
    await expect(
      async () => await model.doEmbed({ values: many }),
    ).rejects.toBeInstanceOf(TooManyEmbeddingValuesForCallError)
  })
})
