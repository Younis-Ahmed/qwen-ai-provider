import type {
  APICallError,
  LanguageModelV1,
  LanguageModelV1CallWarning,
  LanguageModelV1FinishReason,
  LanguageModelV1ObjectGenerationMode,
  LanguageModelV1StreamPart,
} from "@ai-sdk/provider"
import type {
  FetchFunction,
  ParseResult,
  ResponseHandler,
} from "@ai-sdk/provider-utils"
import type {
  QwenChatModelId,
  QwenChatSettings,
} from "./qwen-chat-settings"
import type {
  QwenErrorStructure,
} from "./qwen-error"
import type { MetadataExtractor } from "./qwen-metadata-extractor"
import {
  InvalidResponseDataError,
} from "@ai-sdk/provider"
import {
  combineHeaders,
  createEventSourceResponseHandler,
  createJsonErrorResponseHandler,
  createJsonResponseHandler,
  generateId,
  isParsableJson,
  postJsonToApi,
} from "@ai-sdk/provider-utils"
import { z } from "zod"
import { convertToQwenChatMessages } from "./convert-to-qwen-chat-messages"
import { getResponseMetadata } from "./get-response-metadata"
import { mapQwenFinishReason } from "./map-qwen-finish-reason"
import {
  defaultQwenErrorStructure,
} from "./qwen-error"
import { prepareTools } from "./qwen-prepare-tools"

/**
 * Configuration for the Qwen Chat Language Model.
 * @interface QwenChatConfig
 */
export interface QwenChatConfig {
  provider: string
  headers: () => Record<string, string | undefined>
  url: (options: { modelId: string, path: string }) => string
  fetch?: FetchFunction
  errorStructure?: QwenErrorStructure<any>
  metadataExtractor?: MetadataExtractor

  /**
  Default object generation mode that should be used with this model when
  no mode is specified. Should be the mode with the best results for this
  model. `undefined` can be specified if object generation is not supported.
   */
  defaultObjectGenerationMode?: LanguageModelV1ObjectGenerationMode

  /**
   * Whether the model supports structured outputs.
   */
  supportsStructuredOutputs?: boolean
}

// limited version of the schema, focussed on what is needed for the implementation
// this approach limits breakages when the API changes and increases efficiency
const QwenChatResponseSchema = z.object({
  id: z.string().nullish(),
  created: z.number().nullish(),
  model: z.string().nullish(),
  choices: z.array(
    z.object({
      message: z.object({
        role: z.literal("assistant").nullish(),
        content: z.string().nullish(),
        reasoning_content: z.string().nullish(),
        tool_calls: z
          .array(
            z.object({
              id: z.string().nullish(),
              type: z.literal("function"),
              function: z.object({
                name: z.string(),
                arguments: z.string(),
              }),
            }),
          )
          .nullish(),
      }),
      finish_reason: z.string().nullish(),
    }),
  ),
  usage: z
    .object({
      prompt_tokens: z.number().nullish(),
      completion_tokens: z.number().nullish(),
    })
    .nullish(),
})

/**
 * Class representing the Qwen Chat language model.
 * Implements LanguageModelV1 providing text generation and streaming.
 */
/**
 * A language model implementation for Qwen Chat API that follows the LanguageModelV1 interface.
 * Handles both regular text generation and structured outputs through various modes.
 *
 * @param options.mode - The generation mode configuration that determines how the model generates responses:
 *                      'regular' for standard chat completion,
 *                      'object-json' for JSON-structured outputs,
 *                      'object-tool' for function-calling outputs
 * @param options.prompt - The input prompt messages to send to the model
 * @param options.maxTokens - Maximum number of tokens to generate in the response
 * @param options.temperature - Controls randomness in the model's output (0-2)
 * @param options.topP - Nucleus sampling parameter that controls diversity (0-1)
 * @param options.topK - Not supported by Qwen - will generate a warning if used
 * @param options.frequencyPenalty - Penalizes frequent tokens (-2 to 2)
 * @param options.presencePenalty - Penalizes repeated tokens (-2 to 2)
 * @param options.providerMetadata - Additional provider-specific metadata to include in the request
 * @param options.stopSequences - Array of sequences where the model should stop generating
 * @param options.responseFormat - Specifies the desired format of the response (e.g. JSON)
 * @param options.seed - Random seed for deterministic generation
 */
export class QwenChatLanguageModel implements LanguageModelV1 {
  readonly specificationVersion = "v1"

  readonly supportsStructuredOutputs: boolean

  readonly modelId: QwenChatModelId
  readonly settings: QwenChatSettings

  private readonly config: QwenChatConfig
  private readonly failedResponseHandler: ResponseHandler<APICallError>
  private readonly chunkSchema // type inferred via constructor

  /**
   * Constructs a new QwenChatLanguageModel.
   * @param modelId - The model identifier.
   * @param settings - Settings for the chat.
   * @param config - Model configuration.
   */
  constructor(
    modelId: QwenChatModelId,
    settings: QwenChatSettings,
    config: QwenChatConfig,
  ) {
    this.modelId = modelId
    this.settings = settings
    this.config = config

    // Initialize error handling using provided or default error structure.
    const errorStructure
        = config.errorStructure ?? defaultQwenErrorStructure
    this.chunkSchema = createQwenChatChunkSchema(
      errorStructure.errorSchema,
    )
    this.failedResponseHandler = createJsonErrorResponseHandler(errorStructure)

    this.supportsStructuredOutputs = config.supportsStructuredOutputs ?? false
  }

  /**
   * Getter for the default object generation mode.
   */
  get defaultObjectGenerationMode(): "json" | "tool" | undefined {
    return this.config.defaultObjectGenerationMode
  }

  /**
   * Getter for the provider name.
   */
  get provider(): string {
    return this.config.provider
  }

  /**
   * Internal getter that extracts the provider options name.
   * @private
   */
  private get providerOptionsName(): string {
    return this.config.provider.split(".")[0].trim()
  }

  /**
   * Generates the arguments and warnings required for a language model generation call.
   *
   * This function prepares the argument object based on the provided generation options and mode,
   * including any necessary warnings for unsupported settings. It handles different generation modes
   * such as regular, object-json, and object-tool.
   *
   * @param options.mode - The generation mode configuration containing the type and additional settings.
   * @param options.prompt - The prompt input used to generate chat messages.
   * @param options.maxTokens - The maximum number of tokens to generate.
   * @param options.temperature - The temperature setting to control randomness in generation.
   * @param options.topP - The nucleus sampling parameter (top-p) for token selection.
   * @param options.topK - The top-k sampling parameter; if provided, it triggers a warning as it is unsupported.
   * @param options.frequencyPenalty - The penalty applied to frequently occurring tokens.
   * @param options.presencePenalty - The penalty applied based on the presence of tokens.
   * @param options.providerMetadata - Additional metadata customized for the specific provider.
   * @param options.stopSequences - An array of sequences that will signal the generation to stop.
   * @param options.responseFormat - The desired response format; supports JSON schema formatting when structured outputs are enabled.
   * @param options.seed - An optional seed value for randomization.
   *
   * @returns An object containing:
   * - args: The arguments constructed for the language model generation request.
   * - warnings: A list of warnings related to unsupported or deprecated settings.
   */
  private getArgs({
    mode,
      prompt,
      maxTokens,
      temperature,
      topP,
      topK,
      frequencyPenalty,
      presencePenalty,
      providerMetadata,
      stopSequences,
      responseFormat,
      seed,
  }: Parameters<LanguageModelV1["doGenerate"]>[0]) {
    // Determine the type of generation mode.
    const type = mode.type

    const warnings: LanguageModelV1CallWarning[] = []

    // Warn if unsupported settings are used:
    if (topK != null) {
      warnings.push({
        type: "unsupported-setting",
        setting: "topK",
      })
    }

    if (
      responseFormat?.type === "json"
      && responseFormat.schema != null
      && !this.supportsStructuredOutputs
    ) {
      warnings.push({
        type: "unsupported-setting",
        setting: "responseFormat",
        details:
            "JSON response format schema is only supported with structuredOutputs",
      })
    }

    const baseArgs = {
      // model id:
      model: this.modelId,

      // model specific settings:
      user: this.settings.user,

      // standardized settings:
      max_tokens: maxTokens,
      temperature,
      top_p: topP,
      frequency_penalty: frequencyPenalty,
      presence_penalty: presencePenalty,
      response_format:
          responseFormat?.type === "json"
            ? this.supportsStructuredOutputs === true
            && responseFormat.schema != null
              ? {
                  type: "json_schema",
                  json_schema: {
                    schema: responseFormat.schema,
                    name: responseFormat.name ?? "response",
                    description: responseFormat.description,
                  },
                }
              : { type: "json_object" }
            : undefined,

      stop: stopSequences,
      seed,
      ...providerMetadata?.[this.providerOptionsName],

      // messages:
      messages: convertToQwenChatMessages(prompt),
    }

    // Handling various generation modes.
    switch (type) {
      case "regular": {
        const { tools, tool_choice, toolWarnings } = prepareTools({
          mode,
          structuredOutputs: this.supportsStructuredOutputs,
        })

        return {
          args: { ...baseArgs, tools, tool_choice },
          warnings: [...warnings, ...toolWarnings],
        }
      }

      case "object-json": {
        return {
          args: {
            ...baseArgs,
            response_format:
                this.supportsStructuredOutputs === true && mode.schema != null
                  ? {
                      type: "json_schema",
                      json_schema: {
                        schema: mode.schema,
                        name: mode.name ?? "response",
                        description: mode.description,
                      },
                    }
                  : { type: "json_object" },
          },
          warnings,
        }
      }

      case "object-tool": {
        return {
          args: {
            ...baseArgs,
            tool_choice: {
              type: "function",
              function: { name: mode.tool.name },
            },
            tools: [
              {
                type: "function",
                function: {
                  name: mode.tool.name,
                  description: mode.tool.description,
                  parameters: mode.tool.parameters,
                },
              },
            ],
          },
          warnings,
        }
      }

      default: {
        const _exhaustiveCheck: never = type
        throw new Error(`Unsupported type: ${_exhaustiveCheck}`)
      }
    }
  }

  /**
   * Generates a text response from the model.
   * @param options - Generation options.
   * @returns A promise resolving with the generation result.
   */
  async doGenerate(
    options: Parameters<LanguageModelV1["doGenerate"]>[0],
  ): Promise<Awaited<ReturnType<LanguageModelV1["doGenerate"]>>> {
    const { args, warnings } = this.getArgs({ ...options })

    const body = JSON.stringify(args)

    // Send request for generation using POST JSON.
    const {
      responseHeaders,
      value: responseBody,
      rawValue: parsedBody,
    } = await postJsonToApi({
      url: this.config.url({
        path: "/chat/completions",
        modelId: this.modelId,
      }),
      headers: combineHeaders(this.config.headers(), options.headers),
      body: args,
      failedResponseHandler: this.failedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(
        QwenChatResponseSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    })

    const { messages: rawPrompt, ...rawSettings } = args
    const choice = responseBody.choices[0]
    const providerMetadata = this.config.metadataExtractor?.extractMetadata?.({
      parsedBody,
    })

    // Return structured generation details.
    return {
      text: choice.message.content ?? undefined,
      reasoning: choice.message.reasoning_content ?? undefined,
      toolCalls: choice.message.tool_calls?.map(toolCall => ({
        toolCallType: "function",
        toolCallId: toolCall.id ?? generateId(),
        toolName: toolCall.function.name,
        args: toolCall.function.arguments!,
      })),
      finishReason: mapQwenFinishReason(choice.finish_reason),
      usage: {
        promptTokens: responseBody.usage?.prompt_tokens ?? Number.NaN,
        completionTokens: responseBody.usage?.completion_tokens ?? Number.NaN,
      },
      ...(providerMetadata && { providerMetadata }),
      rawCall: { rawPrompt, rawSettings },
      rawResponse: { headers: responseHeaders },
      response: getResponseMetadata(responseBody),
      warnings,
      request: { body },
    }
  }

  /**
   * Returns a stream of model responses.
   * @param options - Stream generation options.
   * @returns A promise resolving with the stream and additional metadata.
   */
  async doStream(
    options: Parameters<LanguageModelV1["doStream"]>[0],
  ): Promise<Awaited<ReturnType<LanguageModelV1["doStream"]>>> {
    if (this.settings.simulateStreaming) {
      // Simulate streaming by generating a full response and splitting it.
      const result = await this.doGenerate(options)
      const simulatedStream = new ReadableStream<LanguageModelV1StreamPart>({
        start(controller) {
          // Send metadata then text deltas.
          controller.enqueue({ type: "response-metadata", ...result.response })
          if (result.reasoning) {
            const reasoningText = typeof result.reasoning === 'string' 
              ? result.reasoning 
              : result.reasoning.map(item => item.type === 'text' ? item.text : '').join('')
            controller.enqueue({
              type: "reasoning",
              textDelta: reasoningText,
            })
          }
          if (result.text) {
            controller.enqueue({
              type: "text-delta",
              textDelta: result.text,
            })
          }
          if (result.toolCalls) {
            for (const toolCall of result.toolCalls) {
              controller.enqueue({
                type: "tool-call",
                ...toolCall,
              })
            }
          }
          controller.enqueue({
            type: "finish",
            finishReason: result.finishReason,
            usage: result.usage,
            logprobs: result.logprobs,
            providerMetadata: result.providerMetadata,
          })
          controller.close()
        },
      })
      return {
        stream: simulatedStream,
        rawCall: result.rawCall,
        rawResponse: result.rawResponse,
        warnings: result.warnings,
      }
    }

    const { args, warnings } = this.getArgs({ ...options })

    const requestBody = {
      ...args,
      stream: true,
      stream_options: {
        include_usage: true,
      },
    }

    // Set stream flag to true for the API.
    const body = JSON.stringify(requestBody)
    const metadataExtractor
        = this.config.metadataExtractor?.createStreamExtractor()

    const { responseHeaders, value: response } = await postJsonToApi({
      url: this.config.url({
        path: "/chat/completions",
        modelId: this.modelId,
      }),
      headers: combineHeaders(this.config.headers(), options.headers),
      body: requestBody,
      failedResponseHandler: this.failedResponseHandler,
      successfulResponseHandler: createEventSourceResponseHandler(
        this.chunkSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    })

    const { messages: rawPrompt, ...rawSettings } = args

    const toolCalls: Array<{
      id: string
      type: "function"
      function: {
        name: string
        arguments: string
      }
      hasFinished: boolean
    }> = []

    let finishReason: LanguageModelV1FinishReason = "unknown"
    let usage: {
      promptTokens: number | undefined
      completionTokens: number | undefined
    } = {
      promptTokens: undefined,
      completionTokens: undefined,
    }
    let isFirstChunk = true

    return {
      stream: response.pipeThrough(
        new TransformStream<
          ParseResult<z.infer<typeof this.chunkSchema>>,
          LanguageModelV1StreamPart
        >({
          // Transforms incoming chunks and maps them to stream parts.
          transform(chunk, controller) {
            // If validation fails, emit an error.
            if (!chunk.success) {
              finishReason = "error"
              controller.enqueue({ type: "error", error: chunk.error })
              return
            }
            const value = chunk.value

            metadataExtractor?.processChunk(chunk.rawValue)

            // If API sends an error field in the value.
            if ("error" in value) {
              finishReason = "error"
              controller.enqueue({ type: "error", error: value.error.message })
              return
            }

            if (isFirstChunk) {
              isFirstChunk = false

              controller.enqueue({
                type: "response-metadata",
                ...getResponseMetadata(value),
              })
            }

            if (value.usage != null) {
              usage = {
                promptTokens: value.usage.prompt_tokens ?? undefined,
                completionTokens: value.usage.completion_tokens ?? undefined,
              }
            }

            const choice = value.choices[0]

            if (choice?.finish_reason != null) {
              finishReason = mapQwenFinishReason(
                choice.finish_reason,
              )
            }

            if (choice?.delta == null) {
              return
            }

            const delta = choice.delta

            // Emit reasoning before the main content.
            if (delta.reasoning_content != null) {
              controller.enqueue({
                type: "reasoning",
                textDelta: delta.reasoning_content,
              })
            }

            if (delta.content != null) {
              controller.enqueue({
                type: "text-delta",
                textDelta: delta.content,
              })
            }

            // Process and merge tool call deltas.
            if (delta.tool_calls != null) {
              for (const toolCallDelta of delta.tool_calls) {
                const index = toolCallDelta.index

                if (toolCalls[index] == null) {
                  if (toolCallDelta.type !== "function") {
                    throw new InvalidResponseDataError({
                      data: toolCallDelta,
                      message: `Expected 'function' type.`,
                    })
                  }

                  if (toolCallDelta.id == null) {
                    throw new InvalidResponseDataError({
                      data: toolCallDelta,
                      message: `Expected 'id' to be a string.`,
                    })
                  }

                  if (toolCallDelta.function?.name == null) {
                    throw new InvalidResponseDataError({
                      data: toolCallDelta,
                      message: `Expected 'function.name' to be a string.`,
                    })
                  }

                  toolCalls[index] = {
                    id: toolCallDelta.id,
                    type: "function",
                    function: {
                      name: toolCallDelta.function.name,
                      arguments: toolCallDelta.function.arguments ?? "",
                    },
                    hasFinished: false,
                  }

                  const toolCall = toolCalls[index]

                  if (
                    toolCall.function?.name != null
                    && toolCall.function?.arguments != null
                  ) {
                    if (toolCall.function.arguments.length > 0) {
                      controller.enqueue({
                        type: "tool-call-delta",
                        toolCallType: "function",
                        toolCallId: toolCall.id,
                        toolName: toolCall.function.name,
                        argsTextDelta: toolCall.function.arguments,
                      })
                    }

                    // If the accumulated arguments are valid JSON, finish the tool call.
                    if (isParsableJson(toolCall.function.arguments)) {
                      controller.enqueue({
                        type: "tool-call",
                        toolCallType: "function",
                        toolCallId: toolCall.id ?? generateId(),
                        toolName: toolCall.function.name,
                        args: toolCall.function.arguments,
                      })
                      toolCall.hasFinished = true
                    }
                  }

                  continue
                }

                // Merge new delta with ongoing tool call.
                const toolCall = toolCalls[index]

                if (toolCall.hasFinished) {
                  continue
                }

                if (toolCallDelta.function?.arguments != null) {
                  toolCall.function!.arguments
                      += toolCallDelta.function?.arguments ?? ""
                }

                controller.enqueue({
                  type: "tool-call-delta",
                  toolCallType: "function",
                  toolCallId: toolCall.id,
                  toolName: toolCall.function.name,
                  argsTextDelta: toolCallDelta.function.arguments ?? "",
                })

                if (
                  toolCall.function?.name != null
                  && toolCall.function?.arguments != null
                  && isParsableJson(toolCall.function.arguments)
                ) {
                  controller.enqueue({
                    type: "tool-call",
                    toolCallType: "function",
                    toolCallId: toolCall.id ?? generateId(),
                    toolName: toolCall.function.name,
                    args: toolCall.function.arguments,
                  })
                  toolCall.hasFinished = true
                }
              }
            }
          },

          flush(controller) {
            // Build final metadata and finish streaming.
            const metadata = metadataExtractor?.buildMetadata()
            controller.enqueue({
              type: "finish",
              finishReason,
              usage: {
                promptTokens: usage.promptTokens ?? Number.NaN,
                completionTokens: usage.completionTokens ?? Number.NaN,
              },
              ...(metadata && { providerMetadata: metadata }),
            })
          },
        }),
      ),
      rawCall: { rawPrompt, rawSettings },
      rawResponse: { headers: responseHeaders },
      warnings,
      request: { body },
    }
  }
}

// limited version of the schema, focussed on what is needed for the implementation
// this approach limits breakages when the API changes and increases efficiency
function createQwenChatChunkSchema<ERROR_SCHEMA extends z.ZodType>(errorSchema: ERROR_SCHEMA) {
  return z.union([
    z.object({
      id: z.string().nullish(),
      created: z.number().nullish(),
      model: z.string().nullish(),
      choices: z.array(
        z.object({
          delta: z
            .object({
              role: z.enum(["assistant"]).nullish(),
              content: z.string().nullish(),
              reasoning_content: z.string().nullish(),
              tool_calls: z
                .array(
                  z.object({
                    index: z.number(),
                    id: z.string().nullish(),
                    type: z.literal("function").optional(),
                    function: z.object({
                      name: z.string().nullish(),
                      arguments: z.string().nullish(),
                    }),
                  }),
                )
                .nullish(),
            })
            .nullish(),
          finish_reason: z.string().nullish(),
        }),
      ),
      usage: z
        .object({
          prompt_tokens: z.number().nullish(),
          completion_tokens: z.number().nullish(),
        })
        .nullish(),
    }),
    errorSchema,
  ])
}
