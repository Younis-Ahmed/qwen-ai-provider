import type {
  LanguageModelV2,
  LanguageModelV2CallWarning,
} from "@ai-sdk/provider"
import type {
  QwenCompletionModelId,
  QwenCompletionSettings,
} from "./qwen-completion-settings"
import type { QwenModelConfig } from "./qwen-model-config"
import { generateId } from "@ai-sdk/provider-utils"
import { z } from "zod"
import { convertToQwenCompletionPrompt } from "./convert-to-qwen-completion-prompt"
import { getResponseMetadata } from "./get-response-metadata"
import { mapQwenFinishReason } from "./map-qwen-finish-reason"
import { QwenHttpClient } from "./qwen-http-client"
import { createQwenStreamTransformer } from "./qwen-stream"

/**
 * Configuration for the Qwen Completion Language Model.
 */
export interface QwenCompletionConfig extends QwenModelConfig {
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
 * A language model implementation for Qwen Completion API that follows the LanguageModelV2 interface.
 *
 * @param options.prompt - The input prompt messages to send to the model
 * @param options.maxOutputTokens - Maximum number of tokens to generate in the response
 * @param options.temperature - Controls randomness in the model's output (0-2)
 * @param options.topP - Nucleus sampling parameter that controls diversity (0-1)
 * @param options.topK - Not supported by Qwen - will generate a warning if used
 * @param options.frequencyPenalty - Penalizes frequent tokens (-2 to 2)
 * @param options.presencePenalty - Penalizes repeated tokens (-2 to 2)
 * @param options.providerOptions - Additional provider-specific options to include in the request
 * @param options.stopSequences - Array of sequences where the model should stop generating
 * @param options.seed - Random seed for deterministic generation
 */
export class QwenCompletionLanguageModel implements LanguageModelV2 {
  readonly specificationVersion = "v2" as const

  readonly supportedUrls: Record<string, RegExp[]> = {}

  readonly modelId: QwenCompletionModelId
  readonly settings: QwenCompletionSettings

  private readonly config: QwenCompletionConfig
  readonly http: QwenHttpClient
  readonly chunkSchema // type inferred via constructor

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
    this.http = new QwenHttpClient(config)
    this.chunkSchema = createQwenCompletionChunkSchema(
      this.http.baseErrorStructure.errorSchema,
    )
  }

  get provider(): string {
    return this.config.provider
  }

  get providerOptionsName(): string {
    return this.config.provider.split(".")[0].trim()
  }

  /**
   * Generates the arguments for invoking the LanguageModelV2 doGenerate method.
   *
   * @param options - The configuration options for generating completion arguments.
   */
  getArgs(options: Parameters<LanguageModelV2["doGenerate"]>[0]) {
    const warnings: LanguageModelV2CallWarning[] = []

    if (options.topK != null) {
      warnings.push({
        type: "unsupported-setting",
        setting: "topK",
      })
    }

    if (options.tools?.length) {
      warnings.push({
        type: "unsupported-setting",
        setting: "tools",
      })
    }

    if (options.toolChoice != null) {
      warnings.push({
        type: "unsupported-setting",
        setting: "toolChoice",
      })
    }

    if (
      options.responseFormat != null
      && options.responseFormat.type !== "text"
    ) {
      warnings.push({
        type: "unsupported-setting",
        setting: "responseFormat",
        details: "JSON response format is not supported.",
      })
    }

    const {
      prompt: completionPrompt,
      stopSequences,
    } = convertToQwenCompletionPrompt({
      prompt: options.prompt,
    })

    const stop = [...(stopSequences ?? []), ...(options.stopSequences ?? [])]

    const args = {
      // Model id and settings:
      model: this.modelId,
      echo: this.settings.echo,
      logit_bias: this.settings.logitBias,
      suffix: this.settings.suffix,
      user: this.settings.user,

      // Standardized settings:
      max_tokens: options.maxOutputTokens,
      temperature: options.temperature,
      top_p: options.topP,
      frequency_penalty: options.frequencyPenalty,
      presence_penalty: options.presencePenalty,
      seed: options.seed,
      ...options.providerOptions?.[this.providerOptionsName],

      // Prompt and stop sequences:
      prompt: completionPrompt,
      stop: stop.length > 0 ? stop : undefined,
    }

    return { args, warnings }
  }

  /**
   * Generates a completion response.
   *
   * @param options - Generation options including prompt and parameters.
   * @returns A promise resolving the generated text, usage, finish status, and metadata.
   */
  async doGenerate(
    options: Parameters<LanguageModelV2["doGenerate"]>[0],
  ): Promise<Awaited<ReturnType<LanguageModelV2["doGenerate"]>>> {
    const { args, warnings } = this.getArgs(options)

    const body = JSON.stringify(args)

    const { responseHeaders, value: response } = await this.http.generate({
      modelId: this.modelId,
      path: "/completions",
      body: args,
      schema: QwenCompletionResponseSchema,
      headers: options.headers,
      abortSignal: options.abortSignal,
    })

    const choice = response.choices[0]

    return {
      content: [
        {
          type: "text",
          text: choice.text,
        },
      ],
      usage: {
        inputTokens: response.usage?.prompt_tokens ?? Number.NaN,
        outputTokens: response.usage?.completion_tokens ?? Number.NaN,
        totalTokens:
          response.usage?.prompt_tokens != null
          && response.usage?.completion_tokens != null
            ? response.usage.prompt_tokens + response.usage.completion_tokens
            : undefined,
      },
      finishReason: mapQwenFinishReason(choice.finish_reason),
      request: { body },
      response: {
        ...getResponseMetadata(response),
        headers: responseHeaders,
      },
      warnings,
    }
  }

  /**
   * Streams a completion response.
   *
   * @param options - Generation options including prompt and parameters.
   * @returns A promise resolving a stream of response parts and metadata.
   */
  async doStream(
    options: Parameters<LanguageModelV2["doStream"]>[0],
  ): Promise<Awaited<ReturnType<LanguageModelV2["doStream"]>>> {
    const { args, warnings } = this.getArgs(options)

    const requestBody = {
      ...args,
      stream: true,
    }

    const body = JSON.stringify(requestBody)

    const { responseHeaders, value: response } = await this.http.stream({
      modelId: this.modelId,
      path: "/completions",
      body: requestBody,
      chunkSchema: this.chunkSchema,
      headers: options.headers,
      abortSignal: options.abortSignal,
    })

    const textStreamId = `text-${generateId()}`
    let sentTextStart = false

    return {
      stream: response.pipeThrough(
        new TransformStream(
          createQwenStreamTransformer({
            warnings,
            extractError: (value) => {
              const v = value
              return "error" in v ? v.error : undefined
            },
            onChunk: (value, controller) => {
              const choice = value.choices[0]

              if (choice?.text != null) {
                if (!sentTextStart) {
                  sentTextStart = true
                  controller.enqueue({
                    type: "text-start",
                    id: textStreamId,
                  })
                }

                controller.enqueue({
                  type: "text-delta",
                  id: textStreamId,
                  delta: choice.text,
                })
              }
            },
            onFlush: (controller) => {
              if (sentTextStart) {
                controller.enqueue({ type: "text-end", id: textStreamId })
              }
            },
          }),
        ),
      ),
      request: { body },
      response: { headers: responseHeaders },
    }
  }
}

// limited version of the schema, focussed on what is needed for the implementation
// this approach limits breakages when the API changes and increases efficiency
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
