import type {
  APICallError,
  LanguageModelV1,
  LanguageModelV1CallWarning,
  LanguageModelV1FinishReason,
  LanguageModelV1StreamPart,
} from "@ai-sdk/provider"
import type {
  FetchFunction,
  ParseResult,
  ResponseHandler,
} from "@ai-sdk/provider-utils"
import type {
  QwenCompletionModelId,
  QwenCompletionSettings,
} from "./qwen-completion-settings"
import type {
  QwenErrorStructure,
} from "./qwen-error"
import {
  UnsupportedFunctionalityError,
} from "@ai-sdk/provider"
import {
  combineHeaders,
  createEventSourceResponseHandler,
  createJsonErrorResponseHandler,
  createJsonResponseHandler,
  postJsonToApi,
} from "@ai-sdk/provider-utils"
import { z } from "zod"
import { convertToQwenCompletionPrompt } from "./convert-to-qwen-completion-prompt"
import { getResponseMetadata } from "./get-response-metadata"
import { mapQwenFinishReason } from "./map-qwen-finish-reason"
import {
  defaultQwenErrorStructure,
} from "./qwen-error"

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
 * Implements the LanguageModelV1 interface and handles regular, streaming completions.
 */
export class QwenCompletionLanguageModel
implements LanguageModelV1 {
  readonly specificationVersion = "v1"
  readonly defaultObjectGenerationMode = undefined

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
    const errorStructure
        = config.errorStructure ?? defaultQwenErrorStructure
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
   * Constructs the arguments for API calls based on provided options.
   * 
   * @param options - Options containing generation mode, prompt, and other parameters.
   * @returns An object with args for the API call and any generated warnings.
   */
  private getArgs({
    mode,
      inputFormat,
      prompt,
      maxTokens,
      temperature,
      topP,
      topK,
      frequencyPenalty,
      presencePenalty,
      stopSequences: userStopSequences,
      responseFormat,
      seed,
      providerMetadata,
  }: Parameters<LanguageModelV1["doGenerate"]>[0]) {
    const type = mode.type

    const warnings: LanguageModelV1CallWarning[] = []

    // Warn if unsupported settings are used.
    if (topK != null) {
      warnings.push({
        type: "unsupported-setting",
        setting: "topK",
      })
    }

    if (responseFormat != null && responseFormat.type !== "text") {
      warnings.push({
        type: "unsupported-setting",
        setting: "responseFormat",
        details: "JSON response format is not supported.",
      })
    }

    // Convert prompt to Qwen-specific prompt info.
    const { prompt: completionPrompt, stopSequences }
        = convertToQwenCompletionPrompt({ prompt, inputFormat })

    const stop = [...(stopSequences ?? []), ...(userStopSequences ?? [])]

    const baseArgs = {
      // Model id and settings:
      model: this.modelId,
      echo: this.settings.echo,
      logit_bias: this.settings.logitBias,
      suffix: this.settings.suffix,
      user: this.settings.user,
      // Standardized settings:
      max_tokens: maxTokens,
      temperature,
      top_p: topP,
      frequency_penalty: frequencyPenalty,
      presence_penalty: presencePenalty,
      seed,
      ...providerMetadata?.[this.providerOptionsName],
      // Prompt and stop sequences:
      prompt: completionPrompt,
      stop: stop.length > 0 ? stop : undefined,
    }

    switch (type) {
      case "regular": {
        // Tools are not supported in "regular" mode.
        if (mode.tools?.length) {
          throw new UnsupportedFunctionalityError({
            functionality: "tools",
          })
        }

        if (mode.toolChoice) {
          throw new UnsupportedFunctionalityError({
            functionality: "toolChoice",
          })
        }

        return { args: baseArgs, warnings }
      }

      case "object-json": {
        throw new UnsupportedFunctionalityError({
          functionality: "object-json mode",
        })
      }

      case "object-tool": {
        throw new UnsupportedFunctionalityError({
          functionality: "object-tool mode",
        })
      }

      default: {
        const _exhaustiveCheck: never = type
        throw new Error(`Unsupported type: ${_exhaustiveCheck}`)
      }
    }
  }

  /**
   * Generates a completion response.
   *
   * @param options - Generation options including prompt and parameters.
   * @returns A promise resolving the generated text, usage, finish status, and metadata.
   */
  async doGenerate(
    options: Parameters<LanguageModelV1["doGenerate"]>[0],
  ): Promise<Awaited<ReturnType<LanguageModelV1["doGenerate"]>>> {
    const { args, warnings } = this.getArgs(options)

    const { responseHeaders, value: response } = await postJsonToApi({
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

    // Extract raw prompt and settings for debugging.
    const { prompt: rawPrompt, ...rawSettings } = args
    const choice = response.choices[0]

    return {
      text: choice.text,
      usage: {
        promptTokens: response.usage?.prompt_tokens ?? Number.NaN,
        completionTokens: response.usage?.completion_tokens ?? Number.NaN,
      },
      finishReason: mapQwenFinishReason(choice.finish_reason),
      rawCall: { rawPrompt, rawSettings },
      rawResponse: { headers: responseHeaders },
      response: getResponseMetadata(response),
      warnings,
      request: { body: JSON.stringify(args) },
    }
  }

  /**
   * Streams a completion response.
   *
   * @param options - Generation options including prompt and parameters.
   * @returns A promise resolving a stream of response parts and metadata.
   */
  async doStream(
    options: Parameters<LanguageModelV1["doStream"]>[0],
  ): Promise<Awaited<ReturnType<LanguageModelV1["doStream"]>>> {
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

    const { prompt: rawPrompt, ...rawSettings } = args

    let finishReason: LanguageModelV1FinishReason = "unknown"
    let usage: { promptTokens: number, completionTokens: number } = {
      promptTokens: Number.NaN,
      completionTokens: Number.NaN,
    }
    let isFirstChunk = true

    return {
      stream: response.pipeThrough(
        new TransformStream<
          ParseResult<z.infer<typeof this.chunkSchema>>,
          LanguageModelV1StreamPart
        >({
          transform(chunk, controller) {
            // Validate the current chunk and handle potential errors.
            if (!chunk.success) {
              finishReason = "error"
              controller.enqueue({ type: "error", error: chunk.error })
              return
            }

            const value = chunk.value

            // If the API returns an error inside the chunk.
            if ("error" in value) {
              finishReason = "error"
              controller.enqueue({ type: "error", error: value.error })
              return
            }

            if (isFirstChunk) {
              isFirstChunk = false

              // Send response metadata on first successful chunk.
              controller.enqueue({
                type: "response-metadata",
                ...getResponseMetadata(value),
              })
            }

            if (value.usage != null) {
              usage = {
                promptTokens: value.usage.prompt_tokens,
                completionTokens: value.usage.completion_tokens,
              }
            }

            const choice = value.choices[0]

            if (choice?.finish_reason != null) {
              finishReason = mapQwenFinishReason(
                choice.finish_reason,
              )
            }

            if (choice?.text != null) {
              // Enqueue text delta for streaming.
              controller.enqueue({
                type: "text-delta",
                textDelta: choice.text,
              })
            }
          },

          flush(controller) {
            // Signal the end of the stream, passing finish reason and usage data.
            controller.enqueue({
              type: "finish",
              finishReason,
              usage,
            })
          },
        }),
      ),
      rawCall: { rawPrompt, rawSettings },
      rawResponse: { headers: responseHeaders },
      warnings,
      request: { body: JSON.stringify(body) },
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
function createQwenCompletionChunkSchema<
  ERROR_SCHEMA extends z.ZodType,
>(errorSchema: ERROR_SCHEMA) {
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
