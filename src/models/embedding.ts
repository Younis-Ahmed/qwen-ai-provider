import type { EmbeddingModelV3 } from "@ai-sdk/provider"
import type { FetchFunction } from "@ai-sdk/provider-utils"
import type {
  QwenEmbeddingModelId,
  QwenEmbeddingSettings,
} from "../config/embedding"
import type { QwenErrorStructure } from "../error"
import { TooManyEmbeddingValuesForCallError } from "@ai-sdk/provider"
import {
  combineHeaders,
  createJsonErrorResponseHandler,
  createJsonResponseHandler,
  postJsonToApi,
} from "@ai-sdk/provider-utils"
import { z } from "zod"
import { defaultQwenErrorStructure } from "../error"

interface QwenEmbeddingConfig {
  /**
  Override the maximum number of embeddings per call.
   */
  maxEmbeddingsPerCall?: number

  /**
  Override the parallelism of embedding calls.
   */
  supportsParallelCalls?: boolean

  provider: string
  url: (options: { modelId: string, path: string }) => string
  headers: () => Record<string, string | undefined>
  fetch?: FetchFunction
  errorStructure?: QwenErrorStructure<any>
}

// minimal version of the schema, focussed on what is needed for the implementation
// this approach limits breakages when the API changes and increases efficiency
const qwenTextEmbeddingResponseSchema = z.object({
  data: z.array(z.object({ embedding: z.array(z.number()) })),
  usage: z.object({ prompt_tokens: z.number() }).nullish(),
})

export class QwenEmbeddingModel implements EmbeddingModelV3 {
  readonly specificationVersion = "v3"
  readonly modelId: QwenEmbeddingModelId

  private readonly config: QwenEmbeddingConfig
  private readonly settings: QwenEmbeddingSettings

  get provider(): string {
    return this.config.provider
  }

  get maxEmbeddingsPerCall(): number | undefined {
    return this.config.maxEmbeddingsPerCall ?? 2048
  }

  get supportsParallelCalls(): boolean {
    return this.config.supportsParallelCalls ?? true
  }

  constructor(
    modelId: QwenEmbeddingModelId,
    settings: QwenEmbeddingSettings,
    config: QwenEmbeddingConfig,
  ) {
    this.modelId = modelId
    this.settings = settings
    this.config = config
  }

  /**
   * Sends a request to the Qwen API to generate embeddings for the provided text inputs (V3).
   *
   * @param param0 - The parameters object.
   * @param param0.values - An array of strings to be embedded.
   * @param param0.headers - Optional HTTP headers for the API request.
   * @param param0.abortSignal - Optional signal to abort the API request.
   * @param param0.providerOptions - Additional provider-specific options.
   * @returns A promise that resolves with an object containing embeddings, usage, and response metadata.
   *
   * @throws TooManyEmbeddingValuesForCallError if the number of input values exceeds the maximum allowed.
   */
  async doEmbed({
    values,
    headers,
    abortSignal,
    providerOptions,
  }: Parameters<EmbeddingModelV3["doEmbed"]>[0]): Promise<
    Awaited<ReturnType<EmbeddingModelV3["doEmbed"]>>
  > {
    // Validate that number of embeddings does not exceed maximum allowed.
    const maxEmbeddings = this.maxEmbeddingsPerCall ?? Number.POSITIVE_INFINITY
    if (values.length > maxEmbeddings) {
      throw new TooManyEmbeddingValuesForCallError({
        provider: this.provider,
        modelId: this.modelId,
        maxEmbeddingsPerCall: maxEmbeddings,
        values,
      })
    }

    // Get provider-specific options if available
    const providerOptionsName = this.config.provider.split(".")[0].trim()
    const specificProviderOptions = providerOptions?.[providerOptionsName]

    // Post the JSON payload to the API endpoint.
    const { responseHeaders, value: response } = await postJsonToApi({
      url: this.config.url({
        path: "/embeddings",
        modelId: this.modelId,
      }),
      headers: combineHeaders(this.config.headers(), headers),
      body: {
        model: this.modelId,
        input: values,
        encoding_format: "float",
        dimensions: this.settings.dimensions,
        user: this.settings.user,
        ...specificProviderOptions,
      },
      failedResponseHandler: createJsonErrorResponseHandler(
        this.config.errorStructure ?? defaultQwenErrorStructure,
      ),
      successfulResponseHandler: createJsonResponseHandler(
        qwenTextEmbeddingResponseSchema,
      ),
      abortSignal,
      fetch: this.config.fetch,
    })

    // Map response data to V3 output format.
    return {
      embeddings: response.data.map(item => item.embedding),
      usage: response.usage
        ? { tokens: response.usage.prompt_tokens }
        : undefined,
      response: { headers: responseHeaders },
      warnings: [],
    }
  }
}
