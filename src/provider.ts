import type {
  EmbeddingModelV3,
  LanguageModelV3,
  RerankingModelV3,
} from "@ai-sdk/provider"
import type { FetchFunction } from "@ai-sdk/provider-utils"
import type { QwenChatModelId, QwenChatSettings } from "./config/chat"
import type {
  QwenCompletionModelId,
  QwenCompletionSettings,
} from "./config/completion"
import type {
  QwenEmbeddingModelId,
  QwenEmbeddingSettings,
} from "./config/embedding"
import type {
  QwenRerankingModelId,
  QwenRerankingSettings,
} from "./config/reranking"
import { loadApiKey, withoutTrailingSlash } from "@ai-sdk/provider-utils"
import { QwenChatLanguageModel } from "./models/chat"
import { QwenCompletionLanguageModel } from "./models/completion"
import { QwenEmbeddingModel } from "./models/embedding"
import { QwenRerankingModel } from "./models/reranking"

/**
 * QwenProvider function type and its properties.
 * Creates various language or embedding models based on the provided settings.
 */
export interface QwenProvider {
  (modelId: QwenChatModelId, settings?: QwenChatSettings): LanguageModelV3

  /**
   * Create a new chat model for text generation.
   * @param modelId The model ID.
   * @param settings The settings for the model.
   * @returns The chat model.
   */
  chatModel: (
    modelId: QwenChatModelId,
    settings?: QwenChatSettings,
  ) => LanguageModelV3

  /**
  Creates an embedding model for text generation.
  @param modelId The model ID.
  @param settings The settings for the model.
  @returns The embedding model.
   */
  embeddingModel: (
    modelId: QwenEmbeddingModelId,
    settings?: QwenEmbeddingSettings,
  ) => EmbeddingModelV3

  /**
   * @deprecated Use `embeddingModel` instead.
   */
  textEmbeddingModel: (
    modelId: QwenEmbeddingModelId,
    settings?: QwenEmbeddingSettings,
  ) => EmbeddingModelV3

  /**
   * Creates a reranking model for document relevance ranking.
   * @param modelId The model ID (e.g., 'gte-rerank-v2', 'qwen3-reranker-0.6b').
   * @param settings The settings for the model.
   * @returns The reranking model.
   */
  rerankingModel: (
    modelId: QwenRerankingModelId,
    settings?: QwenRerankingSettings,
  ) => RerankingModelV3

  languageModel: (
    modelId: QwenChatModelId,
    settings?: QwenChatSettings,
  ) => LanguageModelV3

  completion: (
    modelId: QwenCompletionModelId,
    settings?: QwenCompletionSettings,
  ) => LanguageModelV3
}

/**
 * QwenProviderSettings interface holds configuration options for Qwen.
 */
export interface QwenProviderSettings {
  /**
  Use a different URL prefix for API calls, e.g. to use proxy servers.
  The default prefix is `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`.
   */
  baseURL?: string

  /**
  API key that is being send using the `Authorization` header.
  It defaults to the `DASHSCOPE_API_KEY` environment variable.
   */
  apiKey?: string

  /**
  Custom headers to include in the requests.
   */
  headers?: Record<string, string>

  /**
  Optional custom url query parameters to include in request urls.
   */
  queryParams?: Record<string, string>
  /**
  /**
  Custom fetch implementation. You can use it as a middleware to intercept requests,
  or to provide a custom fetch implementation for e.g. testing.
   */
  fetch?: FetchFunction

  // generateId?: () => string
}

/**
 * Creates a Qwen provider instance with the specified options.
 * @param options Provider configuration options.
 * @returns A QwenProvider instance.
 */
export function createQwen(options: QwenProviderSettings = {}): QwenProvider {
  // Remove trailing slash from the base URL.
  const baseURL = withoutTrailingSlash(
    options.baseURL ?? "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
  )

  // Build headers including the API key.
  const getHeaders = () => ({
    Authorization: `Bearer ${loadApiKey({
      apiKey: options.apiKey,
      environmentVariableName: "DASHSCOPE_API_KEY",
      description: "Qwen API key",
    })}`,
    ...options.headers,
  })

  interface CommonModelConfig {
    provider: string
    url: ({ path }: { path: string }) => string
    headers: () => Record<string, string>
    fetch?: FetchFunction
  }

  /**
   * Helper to get common model configuration.
   * @param modelType The type of model (chat, completion, embedding).
   */
  const getCommonModelConfig = (modelType: string): CommonModelConfig => ({
    provider: `qwen.${modelType}`,
    url: ({ path }) => {
      const url = new URL(`${baseURL}${path}`)
      if (options.queryParams) {
        // Append custom query parameters if provided.
        url.search = new URLSearchParams(options.queryParams).toString()
      }
      return url.toString()
    },
    headers: getHeaders,
    fetch: options.fetch,
  })

  // Create a chat language model instance.
  const createChatModel = (
    modelId: QwenChatModelId,
    settings: QwenChatSettings = {},
  ) =>
    new QwenChatLanguageModel(modelId, settings, getCommonModelConfig("chat"))

  // Create a completion model instance.
  const createCompletionModel = (
    modelId: QwenCompletionModelId,
    settings: QwenCompletionSettings = {},
  ) =>
    new QwenCompletionLanguageModel(
      modelId,
      settings,
      getCommonModelConfig("completion"),
    )

  // Create a text embedding model instance.
  const createTextEmbeddingModel = (
    modelId: QwenEmbeddingModelId,
    settings: QwenEmbeddingSettings = {},
  ) =>
    new QwenEmbeddingModel(
      modelId,
      settings,
      getCommonModelConfig("embedding"),
    )

  // Create a reranking model instance.
  // Note: qwen3-rerank uses OpenAI-compatible API (/compatible-mode/v1/rerank)
  //       gte-rerank models use DashScope native API (/api/v1/services/rerank/...)
  // The baseURL is extracted without /compatible-mode/v1 suffix for flexibility.
  const createRerankingModel = (
    modelId: QwenRerankingModelId,
    settings: QwenRerankingSettings = {},
  ) => {
    // Extract the base domain from the baseURL (remove /compatible-mode/v1 if present)
    const rerankBaseURL = (baseURL ?? "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
      .replace(/\/compatible-mode\/v1$/, "")
    return new QwenRerankingModel(modelId, settings, {
      provider: "qwen.reranking",
      baseURL: rerankBaseURL,
      headers: getHeaders,
      fetch: options.fetch,
    })
  }

  // Default provider returns a chat model.
  const provider = (modelId: QwenChatModelId, settings?: QwenChatSettings) =>
    createChatModel(modelId, settings)

  provider.chatModel = createChatModel
  provider.completion = createCompletionModel
  provider.embeddingModel = createTextEmbeddingModel
  provider.textEmbeddingModel = createTextEmbeddingModel // deprecated alias
  provider.rerankingModel = createRerankingModel
  provider.languageModel = createChatModel
  return provider as QwenProvider
}

export const qwen = createQwen()
