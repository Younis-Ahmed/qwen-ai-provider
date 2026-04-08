import type { RerankingModelV3 } from "@ai-sdk/provider"
import type { FetchFunction } from "@ai-sdk/provider-utils"
import type {
  QwenRerankingModelId,
  QwenRerankingSettings,
} from "../config/reranking"
import type { QwenErrorStructure } from "../error"
import {
  combineHeaders,
  createJsonErrorResponseHandler,
  createJsonResponseHandler,
  postJsonToApi,
} from "@ai-sdk/provider-utils"
import { z } from "zod"
import { defaultQwenErrorStructure } from "../error"

export interface QwenRerankingConfig {
  provider: string
  baseURL: string
  headers: () => Record<string, string | undefined>
  fetch?: FetchFunction
  errorStructure?: QwenErrorStructure<any>
}

/**
 * Response schema for DashScope native reranking API (gte-rerank models).
 * The API returns results wrapped in output.results, sorted by relevance score.
 */
const dashscopeRerankingResponseSchema = z.object({
  output: z.object({
    results: z.array(
      z.object({
        index: z.number(),
        relevance_score: z.number(),
        document: z
          .object({
            text: z.string(),
          })
          .nullish(),
      }),
    ),
  }),
  usage: z
    .object({
      total_tokens: z.number().optional(),
    })
    .optional(),
  request_id: z.string().optional(),
})

/**
 * Response schema for OpenAI-compatible reranking API (qwen3-rerank model).
 */
const openaiCompatibleRerankingResponseSchema = z.object({
  results: z.array(
    z.object({
      index: z.number(),
      relevance_score: z.number(),
      document: z
        .object({
          text: z.string(),
        })
        .nullish(),
    }),
  ),
  usage: z
    .object({
      total_tokens: z.number().optional(),
    })
    .optional(),
  id: z.string().optional(),
})

/**
 * Check if a model uses the OpenAI-compatible API format.
 * qwen3-rerank uses the OpenAI-compatible endpoint.
 */
function usesOpenAICompatibleAPI(modelId: string): boolean {
  return modelId.startsWith("qwen3-rerank")
}

export class QwenRerankingModel implements RerankingModelV3 {
  readonly specificationVersion = "v3"
  readonly modelId: QwenRerankingModelId

  private readonly config: QwenRerankingConfig
  private readonly settings: QwenRerankingSettings

  get provider(): string {
    return this.config.provider
  }

  constructor(
    modelId: QwenRerankingModelId,
    settings: QwenRerankingSettings,
    config: QwenRerankingConfig,
  ) {
    this.modelId = modelId
    this.settings = settings
    this.config = config
  }

  /**
   * Reranks documents based on their relevance to the query.
   *
   * @param options - The reranking options.
   * @param options.documents - Documents to rerank (text or objects).
   * @param options.query - The query to rank documents against.
   * @param options.topN - Optional limit on number of results.
   * @param options.headers - Optional HTTP headers.
   * @param options.abortSignal - Optional abort signal.
   * @param options.providerOptions - Additional provider-specific options.
   * @returns Promise with ranking results sorted by relevance.
   */
  async doRerank(
    options: Parameters<RerankingModelV3["doRerank"]>[0],
  ): Promise<Awaited<ReturnType<RerankingModelV3["doRerank"]>>> {
    const { documents, query, topN, headers, abortSignal, providerOptions }
      = options

    // Convert documents to the format expected by DashScope API
    const documentTexts
      = documents.type === "text"
        ? documents.values
        : documents.values.map(doc => JSON.stringify(doc))

    // Get provider-specific options if available
    const providerOptionsName = this.config.provider.split(".")[0].trim()
    const specificProviderOptions = providerOptions?.[providerOptionsName]

    const isOpenAICompatible = usesOpenAICompatibleAPI(this.modelId)

    if (isOpenAICompatible) {
      // qwen3-rerank uses OpenAI-compatible format
      return this.doRerankOpenAICompatible({
        documentTexts,
        query,
        topN,
        headers,
        abortSignal,
        specificProviderOptions,
      })
    }
    else {
      // gte-rerank models use DashScope native format
      return this.doRerankDashScope({
        documentTexts,
        query,
        topN,
        headers,
        abortSignal,
        specificProviderOptions,
      })
    }
  }

  /**
   * Rerank using OpenAI-compatible API format (for qwen3-rerank).
   * Endpoint: /compatible-mode/v1/rerank
   */
  private async doRerankOpenAICompatible(options: {
    documentTexts: string[]
    query: string
    topN?: number
    headers?: Record<string, string | undefined>
    abortSignal?: AbortSignal
    specificProviderOptions?: Record<string, unknown>
  }): Promise<Awaited<ReturnType<RerankingModelV3["doRerank"]>>> {
    const {
      documentTexts,
      query,
      topN,
      headers,
      abortSignal,
      specificProviderOptions,
    } = options

    // Build request body in OpenAI-compatible format
    const body: Record<string, unknown> = {
      model: this.modelId,
      documents: documentTexts,
      query,
      ...specificProviderOptions,
    }

    if (topN != null) {
      body.top_n = topN
    }
    if (this.settings.instruct != null) {
      body.instruct = this.settings.instruct
    }

    // qwen3-rerank uses OpenAI-compatible endpoint
    // Note: endpoint is /compatible-api/v1/reranks (not /compatible-mode/v1/rerank)
    const url = `${this.config.baseURL}/compatible-api/v1/reranks`

    const { responseHeaders, value: response } = await postJsonToApi({
      url,
      headers: combineHeaders(this.config.headers(), headers),
      body,
      failedResponseHandler: createJsonErrorResponseHandler(
        this.config.errorStructure ?? defaultQwenErrorStructure,
      ),
      successfulResponseHandler: createJsonResponseHandler(
        openaiCompatibleRerankingResponseSchema,
      ),
      abortSignal,
      fetch: this.config.fetch,
    })

    return {
      ranking: response.results.map(result => ({
        index: result.index,
        relevanceScore: result.relevance_score,
      })),
      response: {
        id: response.id,
        headers: responseHeaders,
      },
      warnings: [],
    }
  }

  /**
   * Rerank using DashScope native API format (for gte-rerank models).
   * Endpoint: /api/v1/services/rerank/text-rerank/text-rerank
   */
  private async doRerankDashScope(options: {
    documentTexts: string[]
    query: string
    topN?: number
    headers?: Record<string, string | undefined>
    abortSignal?: AbortSignal
    specificProviderOptions?: Record<string, unknown>
  }): Promise<Awaited<ReturnType<RerankingModelV3["doRerank"]>>> {
    const {
      documentTexts,
      query,
      topN,
      headers,
      abortSignal,
      specificProviderOptions,
    } = options

    // Build parameters object (only include defined values)
    const parameters: Record<string, unknown> = {}
    if (topN != null) {
      parameters.top_n = topN
    }
    if (this.settings.returnDocuments != null) {
      parameters.return_documents = this.settings.returnDocuments
    }

    // Build request body in DashScope native format
    const body = {
      model: this.modelId,
      input: {
        query,
        documents: documentTexts,
      },
      parameters,
      ...specificProviderOptions,
    }

    // DashScope reranking uses native API endpoint
    const url = `${this.config.baseURL}/api/v1/services/rerank/text-rerank/text-rerank`

    const { responseHeaders, value: response } = await postJsonToApi({
      url,
      headers: combineHeaders(this.config.headers(), headers),
      body,
      failedResponseHandler: createJsonErrorResponseHandler(
        this.config.errorStructure ?? defaultQwenErrorStructure,
      ),
      successfulResponseHandler: createJsonResponseHandler(
        dashscopeRerankingResponseSchema,
      ),
      abortSignal,
      fetch: this.config.fetch,
    })

    return {
      ranking: response.output.results.map(result => ({
        index: result.index,
        relevanceScore: result.relevance_score,
      })),
      response: {
        id: response.request_id,
        headers: responseHeaders,
      },
      warnings: [],
    }
  }
}
