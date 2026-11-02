import { createTestServer } from "@ai-sdk/test-server/with-vitest"
import { describe, expect, it } from "vitest"
import { createQwen } from "./qwen-provider"

const dummyEmbeddings = [
  [0.1, 0.2, 0.3, 0.4, 0.5],
  [0.6, 0.7, 0.8, 0.9, 1.0],
]

vi.stubEnv("DASHSCOPE_API_KEY", "test-api-key-123")

const testValues = ["sunny day at the beach", "rainy day in the city"]

const provider = createQwen({
  baseURL: "https://my.api.com/v1/",
  headers: {
    Authorization: `Bearer test-api-key`,
  },
})
const model = provider.textEmbeddingModel("text-embedding-v3")

describe("doEmbed", () => {
  const server = createTestServer({
    "https://my.api.com/v1/embeddings": {
      response: {
        type: "json-value",
        body: {
          object: "list",
          data: dummyEmbeddings.map((embedding, i) => ({
            object: "embedding",
            index: i,
            embedding,
          })),
          model: "text-embedding-v3",
          usage: { prompt_tokens: 8 },
        },
      },
    },
  })

  it("should extract embeddings", async () => {
    const { embeddings } = await model.doEmbed({ values: testValues })

    expect(embeddings).toStrictEqual(dummyEmbeddings)
  })

  it("should expose the response headers", async () => {
    const { response } = await model.doEmbed({ values: testValues })

    expect(response?.headers).toBeDefined()
  })

  it("should extract usage", async () => {
    const { usage } = await model.doEmbed({ values: testValues })

    expect(usage).toStrictEqual({ tokens: 8 })
  })

  it("should pass the model and the values", async () => {
    await model.doEmbed({ values: testValues })

    expect(await server.calls[0].requestBodyJson).toStrictEqual({
      model: "text-embedding-v3",
      input: testValues,
      encoding_format: "float",
    })
  })

  it("should pass the dimensions setting", async () => {
    await provider
      .textEmbeddingModel("text-embedding-v3", { dimensions: 64 })
      .doEmbed({ values: testValues })

    expect(await server.calls[0].requestBodyJson).toStrictEqual({
      model: "text-embedding-v3",
      input: testValues,
      encoding_format: "float",
      dimensions: 64,
    })
  })

  it("should pass per-call headers", async () => {
    const providerWithHeaders = createQwen({
      baseURL: "https://my.api.com/v1/",
      headers: {
        "Authorization": `Bearer test-api-key`,
        "Custom-Provider-Header": "provider-header-value",
      },
    })

    await providerWithHeaders.textEmbeddingModel("text-embedding-v3").doEmbed({
      values: testValues,
      headers: {
        "Custom-Request-Header": "request-header-value",
      },
    })

    const requestHeaders = server.calls[0].requestHeaders

    expect(requestHeaders.authorization).toBe("Bearer test-api-key")
    expect(requestHeaders["content-type"]).toBe("application/json")
    expect(requestHeaders["custom-provider-header"]).toBe("provider-header-value")
    expect(requestHeaders["custom-request-header"]).toBe("request-header-value")
  })

  it("should throw when the number of values exceeds the max embeddings per call", async () => {
    const values = Array.from({ length: 2049 }, (_, i) => `value-${i}`)

    await expect(model.doEmbed({ values })).rejects.toThrow()
  })
})
