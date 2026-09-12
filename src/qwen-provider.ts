import type {
  EmbeddingModelV2,
  ImageModelV2,
  LanguageModelV2,
  ProviderV2,
} from "@ai-sdk/provider"
import type {
  FetchFunction,
} from "@ai-sdk/provider-utils"
import type { QwenChatConfig } from "./qwen-chat-language-model"
import type { QwenChatModelId, QwenChatSettings } from "./qwen-chat-settings"
import type { QwenCompletionConfig } from "./qwen-completion-language-model"
import type {
  QwenCompletionModelId,
  QwenCompletionSettings,
} from "./qwen-completion-settings"
import type { QwenEmbeddingConfig } from "./qwen-embedding-model"
import type {
  QwenEmbeddingModelId,
  QwenEmbeddingSettings,
} from "./qwen-embedding-settings"
import { NoSuchModelError } from "@ai-sdk/provider"
import { loadApiKey, withoutTrailingSlash } from "@ai-sdk/provider-utils"
import { QwenChatLanguageModel } from "./qwen-chat-language-model"
import { QwenCompletionLanguageModel } from "./qwen-completion-language-model"
import { QwenEmbeddingModel } from "./qwen-embedding-model"

/**
 * QwenProvider function type and its properties.
 * Creates various language or embedding models based on the provided settings.
 */
export interface QwenProvider extends ProviderV2 {
  (modelId: QwenChatModelId, settings?: QwenChatSettings): LanguageModelV2

  /**
   * Create a new chat model for text generation.
   * @param modelId The model ID.
   * @param settings The settings for the model.
   * @returns The chat model.
   */
  chatModel: (
    modelId: QwenChatModelId,
    settings?: QwenChatSettings,
  ) => LanguageModelV2

  /**
  Creates a text embedding model for text generation.
  @param modelId The model ID.
  @param settings The settings for the model.
  @returns The text embedding model.
   */
  textEmbeddingModel: (
    modelId: QwenEmbeddingModelId,
    settings?: QwenEmbeddingSettings,
  ) => EmbeddingModelV2<string>

  languageModel: (
    modelId: QwenChatModelId,
    settings?: QwenChatSettings,
  ) => LanguageModelV2

  completion: (
    modelId: QwenCompletionModelId,
    settings?: QwenCompletionSettings,
  ) => LanguageModelV2
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
  Custom fetch implementation. You can use it as a middleware to intercept requests,
  or to provide a custom fetch implementation for e.g. testing.
   */
  fetch?: FetchFunction
}

/**
 * Registry of model factories used to build the adapters returned by
 * `createQwen`.
 *
 * This is the injection seam for the provider. Each entry defaults to the
 * concrete Qwen model class, but callers (and tests) can supply their own
 * factory to control which adapter is returned for a given model kind without
 * touching the provider's orchestration.
 */
export interface QwenModelRegistry {
  chat?: (
    modelId: QwenChatModelId,
    settings: QwenChatSettings,
    config: QwenChatConfig,
  ) => LanguageModelV2
  completion?: (
    modelId: QwenCompletionModelId,
    settings: QwenCompletionSettings,
    config: QwenCompletionConfig,
  ) => LanguageModelV2
  embedding?: (
    modelId: QwenEmbeddingModelId,
    settings: QwenEmbeddingSettings,
    config: QwenEmbeddingConfig,
  ) => EmbeddingModelV2<string>
  image?: () => ImageModelV2
}

/**
 * Creates a Qwen provider instance with the specified options.
 * @param options Provider configuration options.
 * @param registry Optional registry of model factories. Defaults to the
 * concrete Qwen model classes. This is the injection seam used to supply
 * alternative or test adapters.
 * @returns A QwenProvider instance.
 */
export function createQwen(
  options: QwenProviderSettings = {},
  registry: QwenModelRegistry = {},
): QwenProvider {
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

  const getCommonModelConfig = (modelType: string) => ({
    provider: `qwen.${modelType}`,
    url: ({ path }: { path: string }) => {
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
    (registry.chat ?? (
      (modelId: QwenChatModelId, settings2: QwenChatSettings, config: QwenChatConfig) =>
        new QwenChatLanguageModel(modelId, settings2, config)
    ))(modelId, settings, {
      ...getCommonModelConfig("chat"),
      supportsStructuredOutputs: settings.supportsStructuredOutputs ?? true,
    })

  // Create a completion model instance.
  const createCompletionModel = (
    modelId: QwenCompletionModelId,
    settings: QwenCompletionSettings = {},
  ) =>
    (registry.completion ?? (
      (modelId: QwenCompletionModelId, settings2: QwenCompletionSettings, config: QwenCompletionConfig) =>
        new QwenCompletionLanguageModel(modelId, settings2, config)
    ))(modelId, settings, getCommonModelConfig("completion"))

  // Create a text embedding model instance.
  const createTextEmbeddingModel = (
    modelId: QwenEmbeddingModelId,
    settings: QwenEmbeddingSettings = {},
  ) =>
    (registry.embedding ?? (
      (modelId: QwenEmbeddingModelId, settings2: QwenEmbeddingSettings, config: QwenEmbeddingConfig) =>
        new QwenEmbeddingModel(modelId, settings2, config)
    ))(modelId, settings, getCommonModelConfig("embedding"))

  // Default provider returns a chat model.
  const provider = (modelId: QwenChatModelId, settings?: QwenChatSettings) =>
    createChatModel(modelId, settings)

  provider.chatModel = createChatModel
  provider.completion = createCompletionModel
  provider.textEmbeddingModel = createTextEmbeddingModel
  provider.languageModel = createChatModel
  provider.imageModel
    = registry.image
      ?? (() => {
        throw new NoSuchModelError({
          modelId: "qwen-vl",
          modelType: "imageModel",
        })
      })
  return provider as QwenProvider
}

export const qwen = createQwen()
