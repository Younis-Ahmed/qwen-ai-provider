import type {
  APICallError,
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3Content,
  LanguageModelV3FinishReason,
  LanguageModelV3StreamPart,
  LanguageModelV3Usage,
  SharedV3Warning,
} from "@ai-sdk/provider"
import type {
  FetchFunction,
  ParseResult,
  ResponseHandler,
} from "@ai-sdk/provider-utils"
import type {
  QwenCompletionModelId,
  QwenCompletionSettings,
} from "../config/completion"
import type { QwenErrorStructure } from "../error"
import { UnsupportedFunctionalityError } from "@ai-sdk/provider"
import {
  combineHeaders,
  createEventSourceResponseHandler,
  createJsonErrorResponseHandler,
  createJsonResponseHandler,
  generateId,
  postJsonToApi,
} from "@ai-sdk/provider-utils"
import { z } from "zod"
import { defaultQwenErrorStructure } from "../error"
import { buildUsage } from "../utils/build-usage"
import { convertToQwenCompletionPrompt } from "../utils/convert-to-completion-prompt"
import { getResponseMetadata } from "../utils/get-response-metadata"
import { mapQwenFinishReason } from "../utils/map-finish-reason"

interface QwenCompletionConfig {
  provider: string
  headers: () => Record<string, string | undefined>
  url: (options: { modelId: string, path: string }) => string
  fetch?: FetchFunction
  errorStructure?: QwenErrorStructure<any>
}
// limited version of the schema, focussed on what is needed for the implementation
// this approach limits breakages when the API changes and increases efficiency
const QwenCompletionResponseSchema = z.object({
  id: z.string().nullish(),
  created: z.number().nullish(),
  model: z.string().nullish(),
  choices: z.array(
    z.object({
      text: z.string(),
      finish_reason: z.string(),
    }),
  ),
  usage: z
    .object({
      prompt_tokens: z.number(),
      completion_tokens: z.number(),
    })
    .nullish(),
})

/**
 * A language model implementation for Qwen completions.
 *
 * @remarks
 * Implements the LanguageModelV3 interface and handles regular, streaming completions.
 */
export class QwenCompletionLanguageModel implements LanguageModelV3 {
  readonly specificationVersion = "v3"
  readonly supportedUrls: Record<string, RegExp[]> = {}

  readonly modelId: QwenCompletionModelId
  readonly settings: QwenCompletionSettings

  private readonly config: QwenCompletionConfig
  private readonly failedResponseHandler: ResponseHandler<APICallError>
  private readonly chunkSchema // type inferred via constructor

  /**
   * Creates an instance of QwenCompletionLanguageModel.
   *
   * @param modelId - The model identifier.
   * @param settings - The settings specific for Qwen completions.
   * @param config - The configuration object which includes provider options and error handling.
   */
  constructor(
    modelId: QwenCompletionModelId,
    settings: QwenCompletionSettings,
    config: QwenCompletionConfig,
  ) {
    this.modelId = modelId
    this.settings = settings
    this.config = config

    // Initialize error handling schema and response handler.
    const errorStructure = config.errorStructure ?? defaultQwenErrorStructure
    this.chunkSchema = createQwenCompletionChunkSchema(
      errorStructure.errorSchema,
    )
    this.failedResponseHandler = createJsonErrorResponseHandler(errorStructure)
  }

  get provider(): string {
    return this.config.provider
  }

  private get providerOptionsName(): string {
    return this.config.provider.split(".")[0].trim()
  }

  /**
   * Generates the arguments for invoking the LanguageModelV3 doGenerate method.
   */
  private getArgs({
    prompt,
    maxOutputTokens,
    temperature,
    topP,
    topK,
    frequencyPenalty,
    presencePenalty,
    stopSequences: userStopSequences,
    responseFormat,
    seed,
    providerOptions,
    tools,
    toolChoice,
  }: LanguageModelV3CallOptions) {
    const warnings: SharedV3Warning[] = []

    // Warn if unsupported settings are used.
    if (topK != null) {
      warnings.push({
        type: "unsupported",
        feature: "topK",
      })
    }

    if (responseFormat != null && responseFormat.type !== "text") {
      warnings.push({
        type: "unsupported",
        feature: "responseFormat",
        details: "JSON response format is not supported.",
      })
    }

    // Tools are not supported in completion model
    if (tools && tools.length > 0) {
      throw new UnsupportedFunctionalityError({
        functionality: "tools",
      })
    }

    if (toolChoice) {
      throw new UnsupportedFunctionalityError({
        functionality: "toolChoice",
      })
    }

    // Convert prompt to Qwen-specific prompt info.
    const { prompt: completionPrompt, stopSequences }
      = convertToQwenCompletionPrompt({ prompt, inputFormat: "prompt" })

    const stop = [...(stopSequences ?? []), ...(userStopSequences ?? [])]

    const args = {
      // Model id and settings:
      model: this.modelId,
      echo: this.settings.echo,
      logit_bias: this.settings.logitBias,
      suffix: this.settings.suffix,
      user: this.settings.user,
      // Standardized settings:
      max_tokens: maxOutputTokens,
      temperature,
      top_p: topP,
      frequency_penalty: frequencyPenalty,
      presence_penalty: presencePenalty,
      seed,
      ...providerOptions?.[this.providerOptionsName],
      // Prompt and stop sequences:
      prompt: completionPrompt,
      stop: stop.length > 0 ? stop : undefined,
    }

    return { args, warnings }
  }

  /**
   * Generates a completion response (V3).
   *
   * @param options - Generation options including prompt and parameters.
   * @returns A promise resolving the generated content, usage, finish status, and metadata.
   */
  async doGenerate(
    options: LanguageModelV3CallOptions,
  ): Promise<Awaited<ReturnType<LanguageModelV3["doGenerate"]>>> {
    const { args, warnings } = this.getArgs(options)

    const {
      responseHeaders,
      value: response,
      rawValue: parsedBody,
    } = await postJsonToApi({
      url: this.config.url({
        path: "/completions",
        modelId: this.modelId,
      }),
      headers: combineHeaders(this.config.headers(), options.headers),
      body: args,
      failedResponseHandler: this.failedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(
        QwenCompletionResponseSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    })

    const choice = response.choices[0]

    // Build V3 content array
    const content: LanguageModelV3Content[] = []

    if (choice.text) {
      content.push({
        type: "text",
        text: choice.text,
      })
    }

    return {
      content,
      finishReason: mapQwenFinishReason(choice.finish_reason),
      usage: buildUsage(
        response.usage?.prompt_tokens,
        response.usage?.completion_tokens,
      ),
      response: {
        ...getResponseMetadata(response),
        headers: responseHeaders,
        body: parsedBody,
      },
      warnings,
      request: { body: JSON.stringify(args) },
    }
  }

  /**
   * Streams a completion response (V3).
   *
   * @param options - Generation options including prompt and parameters.
   * @returns A promise resolving a stream of response parts and metadata.
   */
  async doStream(
    options: LanguageModelV3CallOptions,
  ): Promise<Awaited<ReturnType<LanguageModelV3["doStream"]>>> {
    const { args, warnings } = this.getArgs(options)

    const body = {
      ...args,
      stream: true,
    }

    const { responseHeaders, value: response } = await postJsonToApi({
      url: this.config.url({
        path: "/completions",
        modelId: this.modelId,
      }),
      headers: combineHeaders(this.config.headers(), options.headers),
      body,
      failedResponseHandler: this.failedResponseHandler,
      successfulResponseHandler: createEventSourceResponseHandler(
        this.chunkSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    })

    let finishReason: LanguageModelV3FinishReason = { unified: "other", raw: undefined }
    let usage: LanguageModelV3Usage = buildUsage(undefined, undefined)
    let isFirstChunk = true
    let hasStreamStarted = false
    let textId: string | undefined

    return {
      stream: response.pipeThrough(
        new TransformStream<
          ParseResult<z.infer<typeof this.chunkSchema>>,
          LanguageModelV3StreamPart
        >({
          transform(chunk, controller) {
            if (!chunk.success) {
              controller.enqueue({ type: "error", error: chunk.error })
              return
            }

            const value = chunk.value

            // Handle provider error objects streamed as chunks
            if ((value as any)?.object === "error") {
              controller.enqueue({ type: "error", error: (value as any).message })
              return
            }

            if (!hasStreamStarted) {
              hasStreamStarted = true
              controller.enqueue({
                type: "stream-start",
                warnings,
              })
            }

            if (isFirstChunk) {
              isFirstChunk = false
              const metadata = getResponseMetadata(value)
              controller.enqueue({
                type: "response-metadata",
                id: metadata.id,
                timestamp: metadata.timestamp,
                modelId: metadata.modelId,
              })
            }

            if (value.usage != null) {
              usage = buildUsage(
                value.usage.prompt_tokens,
                value.usage.completion_tokens,
              )
            }

            const choice = value.choices[0]

            if (choice?.finish_reason != null) {
              finishReason = mapQwenFinishReason(choice.finish_reason)
            }

            if (choice?.text != null) {
              if (!textId) {
                textId = generateId()
                controller.enqueue({ type: "text-start", id: textId })
              }
              controller.enqueue({
                type: "text-delta",
                id: textId,
                delta: choice.text,
              })
            }
          },

          flush(controller) {
            if (textId) {
              controller.enqueue({ type: "text-end", id: textId })
            }

            controller.enqueue({
              type: "finish",
              finishReason,
              usage,
            })
          },
        }),
      ),
      request: { body: JSON.stringify(body) },
      response: { headers: responseHeaders },
    }
  }
}

// limited version of the schema, focussed on what is needed for the implementation
// this approach limits breakages when the API changes and increases efficiency
/**
 * Creates a Zod schema to validate Qwen completion stream chunks.
 *
 * @param errorSchema - Schema to validate error objects.
 * @returns A union schema for a valid chunk or an error.
 */
function createQwenCompletionChunkSchema<ERROR_SCHEMA extends z.ZodType>(
  errorSchema: ERROR_SCHEMA,
) {
  return z.union([
    z.object({
      id: z.string().nullish(),
      created: z.number().nullish(),
      model: z.string().nullish(),
      choices: z.array(
        z.object({
          text: z.string(),
          finish_reason: z.string().nullish(),
          index: z.number(),
        }),
      ),
      usage: z
        .object({
          prompt_tokens: z.number(),
          completion_tokens: z.number(),
        })
        .nullish(),
    }),
    errorSchema,
  ])
}
