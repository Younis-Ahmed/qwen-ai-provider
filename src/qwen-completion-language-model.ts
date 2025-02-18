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
   * Generates the arguments for invoking the LanguageModelV1 doGenerate method.
   *
   * This function processes the given options to build a configuration object for the request. It converts the
   * input prompt to a Qwen-specific format, merges stop sequences from both the user and the prompt conversion,
   * and applies standardized settings for model generation. Additionally, it emits warnings for any unsupported
   * settings (e.g., topK and non-text response formats) and throws errors if unsupported functionalities
   * (such as tools, toolChoice, or specific modes) are detected.
   *
   * @param options - The configuration options for generating completion arguments.
   * @param options.mode - The mode for generation, specifying the type and any additional functionalities.
   * @param options.inputFormat - The format of the input prompt.
   * @param options.prompt - The prompt text to be processed and used for generating a completion.
   * @param options.maxTokens - The maximum number of tokens to generate.
   * @param options.temperature - The sampling temperature for generation randomness.
   * @param options.topP - The nucleus sampling probability threshold.
   * @param options.topK - The Top-K sampling parameter (unsupported; will trigger a warning if provided).
   * @param options.frequencyPenalty - The frequency penalty to reduce token repetition.
   * @param options.presencePenalty - The presence penalty to encourage novel token generation.
   * @param options.stopSequences - Additional stop sequences provided by the user.
   * @param options.responseFormat - The desired response format (non-text formats will trigger a warning).
   * @param options.seed - The seed for random number generation, ensuring deterministic outputs.
   * @param options.providerMetadata - Additional metadata to be merged into the provider-specific settings.
   *
   * @returns An object containing:
   *  - args: The built arguments object ready to be passed to the generation method.
   *  - warnings: A list of warnings for unsupported settings that were detected.
   *
   * @throws UnsupportedFunctionalityError If unsupported functionalities (tools, toolChoice, object-json mode,
   *         or object-tool mode) are specified in the mode configuration.
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
