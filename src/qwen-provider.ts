import type {
  EmbeddingModelV1,
  LanguageModelV1,
  ProviderV1,
} from "@ai-sdk/provider"
import type { FetchFunction } from "@ai-sdk/provider-utils"
import type { QwenChatModelId, QwenChatSettings } from "./qwen-chat-settings"
import type {
  QwenCompletionModelId,
  QwenCompletionSettings,
} from "./qwen-completion-settings"
import type {
  QwenEmbeddingModelId,
  QwenEmbeddingSettings,
} from "./qwen-embedding-settings"
import { loadApiKey, withoutTrailingSlash } from "@ai-sdk/provider-utils"
import { QwenChatLanguageModel } from "./qwen-chat-language-model"
import { QwenCompletionLanguageModel } from "./qwen-completion-language-model"
import { QwenEmbeddingModel } from "./qwen-embedding-model"

/**
 * QwenProvider function type and its properties.
 * Creates various language or embedding models based on the provided settings.
 */
export interface QwenProvider extends ProviderV1 {
  (modelId: QwenChatModelId, settings?: QwenChatSettings): LanguageModelV1

  /**
   * Create a new chat model for text generation.
   * @param modelId The model ID.
   * @param settings The settings for the model.
   * @returns The chat model.
   */
  chatModel: (
    modelId: QwenChatModelId,
    settings?: QwenChatSettings,
  ) => LanguageModelV1

  /**
  Creates a text embedding model for text generation.
  @param modelId The model ID.
  @param settings The settings for the model.
  @returns The text embedding model.
   */
  textEmbeddingModel: (
    modelId: QwenEmbeddingModelId,
    settings?: QwenEmbeddingSettings,
  ) => EmbeddingModelV1<string>

  languageModel: (
    modelId: QwenChatModelId,
    settings?: QwenChatSettings,
  ) => LanguageModelV1

  completion: (
    modelId: QwenCompletionModelId,
    settings?: QwenCompletionSettings,
  ) => LanguageModelV1
}

/**
 * QwenProviderSettings interface holds configuration options for Qwen.
 */
export interface QwenProviderSettings {
  /**
  Use a different URL prefix for API calls, e.g. to use proxy servers.
  The default prefix is `https://dashscope.aliyuncs.com/compatible-mode/v1`.
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
    options.baseURL ?? "https://dashscope.aliyuncs.com/compatible-mode/v1",
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
    new QwenChatLanguageModel(modelId, settings, {
      ...getCommonModelConfig("chat"),
      defaultObjectGenerationMode: "tool",
    })

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

  // Default provider returns a chat model.
  const provider = (modelId: QwenChatModelId, settings?: QwenChatSettings) =>
    createChatModel(modelId, settings)

  provider.chatModel = createChatModel
  provider.completion = createCompletionModel
  provider.textEmbeddingModel = createTextEmbeddingModel
  provider.languageModel = createChatModel
  return provider as QwenProvider
}

export const qwen = createQwen()
