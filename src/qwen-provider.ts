import type { LanguageModelV1, ProviderV1 } from '@ai-sdk/provider'
import type {
  FetchFunction,
} from '@ai-sdk/provider-utils'
import type { QwenChatModelId, QwenChatSettings } from './qwen-chat-settings'
import {
  generateId,
  loadApiKey,
  withoutTrailingSlash,
} from '@ai-sdk/provider-utils'
import { QwenChatLanguageModel } from './qwen-chat-language-model'

// model factory function with additional methods and properties
export interface QwenProvider extends ProviderV1 {
  (modelId: QwenChatModelId, settings?: QwenChatSettings): LanguageModelV1

  chat: (
    modelId: QwenChatModelId,
    settings?: QwenChatSettings,
  ) => LanguageModelV1

  languageModel: (
    modelId: QwenChatModelId,
    settings?: QwenChatSettings,
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
  Custom fetch implementation. You can use it as a middleware to intercept requests,
  or to provide a custom fetch implementation for e.g. testing.
   */
  fetch?: FetchFunction

  generateId?: () => string
}

export function createQwen(
  options: QwenProviderSettings = {},
): QwenProvider {
  const baseURL
        = withoutTrailingSlash(options.baseURL) ?? 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'

  const getHeaders = () => ({
    Authorization: `Bearer ${loadApiKey({
      apiKey: options.apiKey,
      environmentVariableName: 'DASHSCOPE_API_KEY',
      description: 'Qwen API key',
    })}`,
    ...options.headers,
  })

  const createChatModel = (
    modelId: QwenChatModelId,
    settings?: QwenChatSettings,
  ) => new QwenChatLanguageModel(modelId, settings, {
    provider: 'qwen.chat',
    baseURL,
    headers: getHeaders,
    fetch: options.fetch,
  })

  const provider = function (modelId: QwenChatModelId, settings?: QwenChatSettings) {
    return createChatModel(modelId, settings)
  }

  provider.chat = createChatModel

  provider.languageModel = createChatModel

  return provider as QwenProvider
}

export const qwen = createQwen()
