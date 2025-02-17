import type { EmbeddingModelV1, LanguageModelV1, ProviderV1 } from '@ai-sdk/provider'
import type {
  FetchFunction,
} from '@ai-sdk/provider-utils'
import type { QwenChatModelId, QwenChatSettings } from './qwen-chat-settings'
import type { QwenCompletionModelId, QwenCompletionSettings } from './qwen-completion-settings'
import type { QwenEmbeddingModelId, QwenEmbeddingSettings } from './qwen-embedding-settings'
import {
  OpenAICompatibleChatLanguageModel,
  OpenAICompatibleCompletionLanguageModel,
  OpenAICompatibleEmbeddingModel,
} from '@ai-sdk/openai-compatible'
import {
  loadApiKey,
  withoutTrailingSlash,
} from '@ai-sdk/provider-utils'

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

export function createQwen(
  options: QwenProviderSettings = {},
): QwenProvider {
  const baseURL
        = withoutTrailingSlash(options.baseURL ?? 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1')

  const getHeaders = () => ({
    Authorization: `Bearer ${loadApiKey({
      apiKey: options.apiKey,
      environmentVariableName: 'DASHSCOPE_API_KEY',
      description: 'Qwen API key',
    })}`,
    ...options.headers,
  })

  interface CommonModelConfig {
    provider: string
    url: ({ path }: { path: string }) => string
    headers: () => Record<string, string>
    fetch?: FetchFunction
  }

  const getCommonModelConfig = (modelType: string): CommonModelConfig => ({
    provider: `qwen.${modelType}`,
    url: ({ path }) => {
      const url = new URL(`${baseURL}${path}`)
      if (options.queryParams) {
        url.search = new URLSearchParams(options.queryParams).toString()
      }
      return url.toString()
    },
    headers: getHeaders,
    fetch: options.fetch,
  })

  const createChatModel = (
    modelId: QwenChatModelId,
    settings: QwenChatSettings = {},
  ) => new OpenAICompatibleChatLanguageModel(modelId, settings, {
    ...getCommonModelConfig('chat'),
    defaultObjectGenerationMode: 'tool',
  })

  const createCompletionModel = (
    modelId: QwenCompletionModelId,
    settings: QwenCompletionSettings = {},
  ) =>
    new OpenAICompatibleCompletionLanguageModel(
      modelId,
      settings,
      getCommonModelConfig('completion'),
    )

  const createTextEmbeddingModel = (
    modelId: QwenEmbeddingModelId,
    settings: QwenEmbeddingSettings = {},
  ) =>
    new OpenAICompatibleEmbeddingModel(
      modelId,
      settings,
      getCommonModelConfig('embedding'),
    )

  const provider = (modelId: QwenChatModelId, settings?: QwenChatSettings) => createChatModel(modelId, settings)

  provider.chatModel = createChatModel
  provider.completion = createCompletionModel
  provider.textEmbeddingModel = createTextEmbeddingModel
  provider.languageModel = createChatModel
  return provider as QwenProvider
}

export const qwen = createQwen()
