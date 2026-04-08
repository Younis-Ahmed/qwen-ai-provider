/**
 * Integration tests for reranking models with real API.
 *
 * Run manually with: pnpm test:integration
 * Requires DASHSCOPE_API_KEY environment variable.
 */
import { rerank } from "ai"
import { describe, expect, it } from "vitest"
import { createQwen } from "../../provider"
import "dotenv/config"

// Skip if no API key
const SKIP_INTEGRATION = !process.env.DASHSCOPE_API_KEY

const qwen = createQwen({
  baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1",
})

const testDocuments = [
  "The quick brown fox jumps over the lazy dog.",
  "A lazy cat naps in the sun.",
  "The dog barks loudly at the mailman.",
  "The fox is a cunning animal.",
]

describe.skipIf(SKIP_INTEGRATION)("reranking Integration Tests", () => {
  describe("gte-rerank-v2 (DashScope native API)", () => {
    it("should rerank documents successfully", async () => {
      const result = await rerank({
        model: qwen.rerankingModel("gte-rerank-v2"),
        documents: testDocuments,
        query: "animals that are not lazy",
        topN: 2,
      })

      expect(result.ranking).toHaveLength(2)
      expect(result.ranking[0]).toHaveProperty("originalIndex")
      expect(result.ranking[0]).toHaveProperty("score")
      expect(result.rerankedDocuments).toHaveLength(2)
    })
  })

  describe("qwen3-rerank (OpenAI-compatible API)", () => {
    it("should rerank documents successfully", async () => {
      const result = await rerank({
        model: qwen.rerankingModel("qwen3-rerank"),
        documents: testDocuments,
        query: "animals that are not lazy",
        topN: 2,
      })

      expect(result.ranking).toHaveLength(2)
      expect(result.ranking[0]).toHaveProperty("originalIndex")
      expect(result.ranking[0]).toHaveProperty("score")
      expect(result.rerankedDocuments).toHaveLength(2)
    })

    it("should support custom instruction", async () => {
      const result = await rerank({
        model: qwen.rerankingModel("qwen3-rerank", {
          instruct: "Retrieve semantically similar text.",
        }),
        documents: testDocuments,
        query: "animals that are not lazy",
        topN: 2,
      })

      expect(result.ranking).toHaveLength(2)
      expect(result.rerankedDocuments).toHaveLength(2)
    })
  })
})
