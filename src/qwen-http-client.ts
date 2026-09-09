import type {
  APICallError,
} from "@ai-sdk/provider"
import type {
  FetchFunction,
  ParseResult,
  ResponseHandler,
} from "@ai-sdk/provider-utils"
import type { z } from "zod"
import type {
  QwenErrorStructure,
} from "./qwen-error"
import type { QwenModelConfig } from "./qwen-model-config"
import {
  combineHeaders,
  createEventSourceResponseHandler,
  createJsonErrorResponseHandler,
  createJsonResponseHandler,
  postJsonToApi,
} from "@ai-sdk/provider-utils"
import {
  defaultQwenErrorStructure,
} from "./qwen-error"

/**
 * A deep module that centralises the Qwen HTTP transport pipeline.
 *
 * It hides the header-combining, URL resolution, request posting, response
 * parsing and error-handling concerns that every model adapter would otherwise
 * repeat. Model adapters delegate their request/response wiring to this client.
 */
export class QwenHttpClient {
  private readonly config: QwenModelConfig
  private readonly failedResponseHandler: ResponseHandler<APICallError>

  get provider(): string {
    return this.config.provider
  }

  /**
   * The error structure in use. Exposed so model adapters can build their
   * chunk schemas from the same error schema the client uses to decode
   * failures.
   */
  get baseErrorStructure(): QwenErrorStructure<any> {
    return this.config.errorStructure ?? defaultQwenErrorStructure
  }

  constructor(config: QwenModelConfig) {
    this.config = config
    this.failedResponseHandler = createJsonErrorResponseHandler(
      this.baseErrorStructure,
    )
  }

  /**
   * Performs a non-streaming POST request and returns the parsed response.
   */
  async generate<T>(params: {
    modelId: string
    path: string
    body: unknown
    schema: z.ZodType<T>
    abortSignal?: AbortSignal
    headers?: Record<string, string | undefined>
    fetch?: FetchFunction
  }): Promise<{
    responseHeaders?: Record<string, string>
    value: T
    rawValue?: unknown
  }> {
    const url = this.config.url({ modelId: params.modelId, path: params.path })
    const headers = combineHeaders(this.config.headers(), params.headers)

    return postJsonToApi({
      url,
      headers,
      body: params.body,
      failedResponseHandler: this.failedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(params.schema),
      abortSignal: params.abortSignal,
      fetch: params.fetch ?? this.config.fetch,
    })
  }

  /**
   * Performs a streaming POST request and returns the event-source stream.
   */
  async stream<T>(params: {
    modelId: string
    path: string
    body: unknown
    chunkSchema: z.ZodType<T>
    abortSignal?: AbortSignal
    headers?: Record<string, string | undefined>
    fetch?: FetchFunction
  }): Promise<{
    responseHeaders?: Record<string, string>
    value: ReadableStream<ParseResult<T>>
  }> {
    const url = this.config.url({ modelId: params.modelId, path: params.path })
    const headers = combineHeaders(this.config.headers(), params.headers)

    return postJsonToApi({
      url,
      headers,
      body: params.body,
      failedResponseHandler: this.failedResponseHandler,
      successfulResponseHandler: createEventSourceResponseHandler(
        params.chunkSchema,
      ),
      abortSignal: params.abortSignal,
      fetch: params.fetch ?? this.config.fetch,
    })
  }
}
