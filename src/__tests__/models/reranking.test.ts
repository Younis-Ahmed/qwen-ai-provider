import { beforeEach, describe, expect, it, vi } from "vitest"
import { createQwen } from "../../provider"

const dummyRankingResults = [
  { index: 1, relevance_score: 0.95 },
  { index: 0, relevance_score: 0.72 },
  { index: 2, relevance_score: 0.31 },
]

vi.stubEnv("DASHSCOPE_API_KEY", "test-api-key-123")

const testDocuments = [
  "sunny day at the beach",
  "rainy afternoon in the city",
  "snowy night in the mountains",
]

describe("doRerank", () => {
  let requestUrl: string
  let requestBody: any
  let requestHeaders: Record<string, string>
  let responseBody: any
  let responseHeaders: Record<string, string>

  beforeEach(() => {
    requestUrl = ""
    requestBody = undefined
    requestHeaders = {}
    // DashScope native API response format
    responseBody = {
      output: {
        results: dummyRankingResults,
      },
      usage: { total_tokens: 100 },
      request_id: "test-request-id-123",
    }
    responseHeaders = {
      "content-type": "application/json",
    }
  })

  function createTestProvider(overrides?: any) {
    return createQwen({
      baseURL: "https://my.api.com/compatible-mode/v1",
      headers: {
        Authorization: `Bearer test-api-key`,
      },
      ...overrides,
      fetch: async (url, init) => {
        // Capture request
        requestUrl = url as string
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

  it("should use DashScope native API endpoint (not compatible mode)", async () => {
    const provider = createTestProvider()
    const model = provider.rerankingModel("gte-rerank-v2")

    await model.doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
    })

    // Should use native API endpoint, not compatible mode
    expect(requestUrl).toBe(
      "https://my.api.com/api/v1/services/rerank/text-rerank/text-rerank",
    )
  })

  it("should extract ranking results", async () => {
    const provider = createTestProvider()
    const model = provider.rerankingModel("gte-rerank-v2")

    const { ranking } = await model.doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
    })

    expect(ranking).toStrictEqual([
      { index: 1, relevanceScore: 0.95 },
      { index: 0, relevanceScore: 0.72 },
      { index: 2, relevanceScore: 0.31 },
    ])
  })

  it("should return request_id in response", async () => {
    const provider = createTestProvider()
    const model = provider.rerankingModel("gte-rerank-v2")

    const { response } = await model.doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
    })

    expect(response?.id).toBe("test-request-id-123")
  })

  it("should expose the raw response headers", async () => {
    responseHeaders = {
      ...responseHeaders,
      "test-header": "test-value",
    }

    const provider = createTestProvider()
    const model = provider.rerankingModel("gte-rerank-v2")

    const { response } = await model.doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
    })

    expect(response?.headers).toMatchObject({
      "content-type": "application/json",
      "test-header": "test-value",
    })
  })

  it("should pass the model, query, and documents in DashScope native format", async () => {
    const provider = createTestProvider()
    const model = provider.rerankingModel("gte-rerank-v2")

    await model.doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
    })

    expect(requestBody).toStrictEqual({
      model: "gte-rerank-v2",
      input: {
        query: "talk about rain",
        documents: testDocuments,
      },
      parameters: {},
    })
  })

  it("should pass topN parameter in parameters object", async () => {
    const provider = createTestProvider()
    const model = provider.rerankingModel("gte-rerank-v2")

    await model.doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
      topN: 2,
    })

    expect(requestBody).toStrictEqual({
      model: "gte-rerank-v2",
      input: {
        query: "talk about rain",
        documents: testDocuments,
      },
      parameters: {
        top_n: 2,
      },
    })
  })

  it("should pass returnDocuments setting in parameters object", async () => {
    const provider = createTestProvider()
    const model = provider.rerankingModel("gte-rerank-v2", {
      returnDocuments: true,
    })

    await model.doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
    })

    expect(requestBody).toStrictEqual({
      model: "gte-rerank-v2",
      input: {
        query: "talk about rain",
        documents: testDocuments,
      },
      parameters: {
        return_documents: true,
      },
    })
  })

  it("should handle object documents by stringifying them", async () => {
    const provider = createTestProvider()
    const model = provider.rerankingModel("gte-rerank-v2")

    const objectDocs = [
      { title: "Beach Day", content: "sunny day at the beach" },
      { title: "City Rain", content: "rainy afternoon in the city" },
    ]

    await model.doRerank({
      documents: { type: "object", values: objectDocs },
      query: "talk about rain",
    })

    expect(requestBody.input.documents).toStrictEqual([
      JSON.stringify(objectDocs[0]),
      JSON.stringify(objectDocs[1]),
    ])
  })

  it("should pass headers", async () => {
    const provider = createTestProvider({
      headers: {
        "Authorization": `Bearer test-api-key`,
        "Custom-Provider-Header": "provider-header-value",
      },
    })

    await provider.rerankingModel("gte-rerank-v2").doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
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

  it("should work with other models using DashScope native format", async () => {
    const provider = createTestProvider()
    const model = provider.rerankingModel("gte-rerank")

    await model.doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
    })

    expect(requestBody.model).toBe("gte-rerank")
    // Should use native API format
    expect(requestBody.input).toBeDefined()
  })

  it("should return empty warnings array", async () => {
    const provider = createTestProvider()
    const model = provider.rerankingModel("gte-rerank-v2")

    const { warnings } = await model.doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
    })

    expect(warnings).toStrictEqual([])
  })

  it("should throw on retryable errors from rerank request", async () => {
    let fetchCalls = 0

    const provider = createQwen({
      baseURL: "https://my.api.com/compatible-mode/v1",
      headers: {
        Authorization: `Bearer test-api-key`,
      },
      fetch: async (url, init) => {
        fetchCalls++
        requestUrl = url as string
        if (init?.body) {
          requestBody = JSON.parse(init.body as string)
        }

        if (fetchCalls === 1) {
          return new Response(JSON.stringify({
            object: "error",
            message: "Bad Gateway",
            type: "BadGateway",
            param: null,
            code: null,
          }), {
            status: 502,
            headers: { "content-type": "application/json" },
          })
        }

        const bodyLength = JSON.stringify(responseBody).length
        return new Response(JSON.stringify(responseBody), {
          headers: {
            ...responseHeaders,
            "content-length": bodyLength.toString(),
          },
        })
      },
    })

    const model = provider.rerankingModel("gte-rerank-v2")
    await expect(model.doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
    })).rejects.toThrow("Bad Gateway")
    expect(fetchCalls).toBe(1)
    expect(requestUrl).toBe("https://my.api.com/api/v1/services/rerank/text-rerank/text-rerank")
  })
})

describe("doRerank with qwen3-rerank (OpenAI-compatible format)", () => {
  let requestUrl: string
  let requestBody: any
  let responseBody: any

  const dummyRankingResults = [
    { index: 1, relevance_score: 0.95 },
    { index: 0, relevance_score: 0.72 },
  ]

  beforeEach(() => {
    requestUrl = ""
    requestBody = undefined
    // OpenAI-compatible API response format
    responseBody = {
      results: dummyRankingResults,
      usage: { total_tokens: 100 },
      id: "test-request-id-456",
    }
  })

  function createTestProvider(overrides?: any) {
    return createQwen({
      baseURL: "https://my.api.com/compatible-mode/v1",
      headers: {
        Authorization: `Bearer test-api-key`,
      },
      ...overrides,
      fetch: async (url, init) => {
        requestUrl = url as string
        if (init?.body) {
          requestBody = JSON.parse(init.body as string)
        }
        const bodyLength = JSON.stringify(responseBody).length
        return new Response(JSON.stringify(responseBody), {
          headers: {
            "content-type": "application/json",
            "content-length": bodyLength.toString(),
          },
        })
      },
    })
  }

  it("should use OpenAI-compatible API endpoint for qwen3-rerank", async () => {
    const provider = createTestProvider()
    const model = provider.rerankingModel("qwen3-rerank")

    await model.doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
    })

    // Should use OpenAI-compatible endpoint (/compatible-api/v1/reranks)
    expect(requestUrl).toBe("https://my.api.com/compatible-api/v1/reranks")
  })

  it("should use flat request format for qwen3-rerank", async () => {
    const provider = createTestProvider()
    const model = provider.rerankingModel("qwen3-rerank")

    await model.doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
    })

    // Should use flat format (not nested input/parameters)
    expect(requestBody).toStrictEqual({
      model: "qwen3-rerank",
      documents: testDocuments,
      query: "talk about rain",
    })
  })

  it("should pass topN parameter directly for qwen3-rerank", async () => {
    const provider = createTestProvider()
    const model = provider.rerankingModel("qwen3-rerank")

    await model.doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
      topN: 2,
    })

    expect(requestBody).toStrictEqual({
      model: "qwen3-rerank",
      documents: testDocuments,
      query: "talk about rain",
      top_n: 2,
    })
  })

  it("should pass instruct setting for qwen3-rerank", async () => {
    const provider = createTestProvider()
    const model = provider.rerankingModel("qwen3-rerank", {
      instruct: "Retrieve semantically similar text.",
    })

    await model.doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
    })

    expect(requestBody.instruct).toBe("Retrieve semantically similar text.")
  })

  it("should extract ranking results from OpenAI-compatible response", async () => {
    const provider = createTestProvider()
    const model = provider.rerankingModel("qwen3-rerank")

    const { ranking } = await model.doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
    })

    expect(ranking).toStrictEqual([
      { index: 1, relevanceScore: 0.95 },
      { index: 0, relevanceScore: 0.72 },
    ])
  })

  it("should return id from OpenAI-compatible response", async () => {
    const provider = createTestProvider()
    const model = provider.rerankingModel("qwen3-rerank")

    const { response } = await model.doRerank({
      documents: { type: "text", values: testDocuments },
      query: "talk about rain",
    })

    expect(response?.id).toBe("test-request-id-456")
  })
})
