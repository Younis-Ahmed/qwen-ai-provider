/**
 * Supported reranking model IDs.
 * - qwen3-rerank: Latest Qwen3 reranking model (uses OpenAI-compatible API)
 * - gte-rerank: GTE reranking models (uses DashScope native API)
 */
export type QwenRerankingModelId
  = | "qwen3-rerank"
    | "gte-rerank"
    | "gte-rerank-v2"
    | "gte-rerank-hybrid-v1"
    | (string & {})

/**
 * Settings configuration for Qwen reranking models.
 */
export interface QwenRerankingSettings {
  /**
   * Whether to return the documents in the response.
   * Default is false.
   * Note: Only supported by gte-rerank-v2 model.
   */
  returnDocuments?: boolean

  /**
   * Custom instruction for ranking task (qwen3-rerank only).
   * Examples:
   * - QA retrieval (default): "Given a web search query, retrieve relevant passages that answer the query."
   * - Semantic similarity: "Retrieve semantically similar text."
   */
  instruct?: string
}
