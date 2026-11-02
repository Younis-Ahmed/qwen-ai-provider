import type {
  EmbeddingModelV2,
} from "@ai-sdk/provider"
import type {
  QwenEmbeddingModelId,
  QwenEmbeddingSettings,
} from "./qwen-embedding-settings"
import type { QwenModelConfig } from "./qwen-model-config"
import {
  TooManyEmbeddingValuesForCallError,
} from "@ai-sdk/provider"
import { z } from "zod"
import { QwenHttpClient } from "./qwen-http-client"

/**
 * Configuration for the Qwen Embedding Model.
 */
export interface QwenEmbeddingConfig extends QwenModelConfig {
  /**
  Override the maximum number of embeddings per call.
   */
  maxEmbeddingsPerCall?: number

  /**
  Override the parallelism of embedding calls.
   */
  supportsParallelCalls?: boolean
}

// limited version of the schema, focussed on what is needed for the implementation
// this approach limits breakages when the API changes and increases efficiency
const qwenTextEmbeddingResponseSchema = z.object({
  data: z.array(z.object({ embedding: z.array(z.number()) })),
  usage: z.object({ prompt_tokens: z.number() }).nullish(),
})

/**
 * A model for generating text embeddings using the Qwen API.
 */
export class QwenEmbeddingModel implements EmbeddingModelV2<string> {
  readonly specificationVersion = "v2" as const

  readonly modelId: QwenEmbeddingModelId
  readonly settings: QwenEmbeddingSettings

  private readonly config: QwenEmbeddingConfig
  readonly http: QwenHttpClient

  constructor(
    modelId: QwenEmbeddingModelId,
    settings: QwenEmbeddingSettings,
    config: QwenEmbeddingConfig,
  ) {
    this.modelId = modelId
    this.settings = settings
    this.config = config
    this.http = new QwenHttpClient(config)
  }

  get provider(): string {
    return this.config.provider
  }

  get maxEmbeddingsPerCall(): number {
    return this.config.maxEmbeddingsPerCall ?? 2048
  }

  get supportsParallelCalls(): boolean {
    return this.config.supportsParallelCalls ?? true
  }

  /**
   * Sends a request to the Qwen API to generate embeddings for the provided
   * text inputs.
   *
   * @param param0 - The parameters object.
   * @param param0.values - An array of strings to be embedded.
   * @param param0.headers - Optional HTTP headers for the API request.
   * @param param0.abortSignal - Optional signal to abort the API request.
   * @returns A promise that resolves with an object containing:
   *   - embeddings: An array of embedding arrays.
   *   - usage: Optional usage information, including token counts.
   *   - response: The response headers from the API call.
   *
   * @throws TooManyEmbeddingValuesForCallError if the number of input values exceeds the maximum allowed.
   */
  async doEmbed({
    values,
    headers,
    abortSignal,
  }: Parameters<EmbeddingModelV2<string>["doEmbed"]>[0]) {
    const maxEmbeddingsPerCall = await this.maxEmbeddingsPerCall
    if (maxEmbeddingsPerCall != null && values.length > maxEmbeddingsPerCall) {
      throw new TooManyEmbeddingValuesForCallError({
        provider: this.provider,
        modelId: this.modelId,
        maxEmbeddingsPerCall,
        values,
      })
    }

    const { responseHeaders, value: response } = await this.http.generate({
      modelId: this.modelId,
      path: "/embeddings",
      body: {
        model: this.modelId,
        input: values,
        encoding_format: "float",
        dimensions: this.settings.dimensions,
        user: this.settings.user,
      },
      schema: qwenTextEmbeddingResponseSchema,
      headers,
      abortSignal,
    })

    return {
      embeddings: response.data.map(item => item.embedding),
      usage: response.usage
        ? { tokens: response.usage.prompt_tokens }
        : undefined,
      response: { headers: responseHeaders },
    }
  }
}
