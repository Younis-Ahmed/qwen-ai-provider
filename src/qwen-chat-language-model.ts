import type {
  APICallError,
  LanguageModelV2,
  LanguageModelV2CallOptions,
  LanguageModelV2CallWarning,
  LanguageModelV2Content,
  LanguageModelV2FinishReason,
  LanguageModelV2StreamPart,
} from "@ai-sdk/provider"
import type {
  FetchFunction,
  ParseResult,
  ResponseHandler,
} from "@ai-sdk/provider-utils"
import type { QwenChatModelId, QwenChatSettings } from "./qwen-chat-settings"
import type { QwenErrorStructure } from "./qwen-error"
import type { MetadataExtractor } from "./qwen-metadata-extractor"
import { InvalidResponseDataError } from "@ai-sdk/provider"
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
import { defaultQwenErrorStructure } from "./qwen-error"

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
 * A language model implementation for Qwen Chat API that follows the LanguageModelV2 interface.
 * Handles both regular text generation and structured outputs through various modes.
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
 * @param options.responseFormat - Specifies the desired format of the response (e.g. JSON)
 * @param options.seed - Random seed for deterministic generation
 */
export class QwenChatLanguageModel implements LanguageModelV2 {
  readonly specificationVersion = "v2"

  readonly supportedUrls: Record<string, RegExp[]> = {}

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
    const errorStructure = config.errorStructure ?? defaultQwenErrorStructure
    this.chunkSchema = createQwenChatChunkSchema(errorStructure.errorSchema)
    this.failedResponseHandler = createJsonErrorResponseHandler(errorStructure)

    this.supportsStructuredOutputs = config.supportsStructuredOutputs ?? false
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
   * Generates the arguments and warnings required for a language model generation call (V2).
   *
   * @param options - V2 call options
   * @param options.prompt - The prompt messages for the model
   * @param options.maxOutputTokens - Maximum number of tokens to generate
   * @param options.temperature - Sampling temperature
   * @param options.topP - Top-p sampling parameter
   * @param options.topK - Top-k sampling parameter
   * @param options.frequencyPenalty - Frequency penalty parameter
   * @param options.presencePenalty - Presence penalty parameter
   * @param options.providerOptions - Provider-specific options
   * @param options.stopSequences - Sequences where the model will stop generating
   * @param options.responseFormat - Expected format of the response
   * @param options.seed - Random seed for reproducibility
   * @param options.tools - Available tools for the model to use
   * @param options.toolChoice - Tool choice configuration
   * @returns An object containing the arguments and warnings
   */
  private getArgs({
    prompt,
    maxOutputTokens,
    temperature,
    topP,
    topK,
    frequencyPenalty,
    presencePenalty,
    providerOptions,
    stopSequences,
    responseFormat,
    seed,
    tools,
    toolChoice,
  }: LanguageModelV2CallOptions) {
    const warnings: LanguageModelV2CallWarning[] = []

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

    // Convert V2 tools to OpenAI format
    const openaiTools = tools
      ?.map((tool) => {
        if (tool.type === "function") {
          return {
            type: "function" as const,
            function: {
              name: tool.name,
              description: tool.description,
              parameters: tool.inputSchema,
            },
          }
        }
        // Provider-defined tools not supported yet
        warnings.push({
          type: "unsupported-tool",
          tool,
        })
        return null
      })
      .filter((t): t is NonNullable<typeof t> => t !== null)

    // Convert V2 tool choice to OpenAI format
    let openaiToolChoice: any
    if (toolChoice) {
      if (toolChoice.type === "auto") {
        openaiToolChoice = "auto"
      }
      else if (toolChoice.type === "none") {
        openaiToolChoice = "none"
      }
      else if (toolChoice.type === "required") {
        openaiToolChoice = "required"
      }
      else if (toolChoice.type === "tool") {
        openaiToolChoice = {
          type: "function",
          function: { name: toolChoice.toolName },
        }
      }
    }

    const args = {
      // model id:
      model: this.modelId,

      // model specific settings:
      user: this.settings.user,

      // standardized settings:
      max_tokens: maxOutputTokens,
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
      ...providerOptions?.[this.providerOptionsName],

      // messages:
      messages: convertToQwenChatMessages(prompt),

      // tools:
      tools: openaiTools && openaiTools.length > 0 ? openaiTools : undefined,
      tool_choice: openaiToolChoice,
    }

    return { args, warnings }
  }

  /**
   * Generates a text response from the model (V2).
   * @param options - Generation options.
   * @returns A promise resolving with the generation result.
   */
  async doGenerate(
    options: LanguageModelV2CallOptions,
  ): Promise<Awaited<ReturnType<LanguageModelV2["doGenerate"]>>> {
    const { args, warnings } = this.getArgs(options)

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

    const choice = responseBody.choices[0]
    const providerMetadata = this.config.metadataExtractor?.extractMetadata?.({
      parsedBody,
    })

    // Build V2 content array
    const content: LanguageModelV2Content[] = []

    // Add reasoning content if present
    if (choice.message.reasoning_content) {
      content.push({
        type: "reasoning",
        text: choice.message.reasoning_content,
      })
    }

    // Add text content if present
    if (choice.message.content) {
      content.push({
        type: "text",
        text: choice.message.content,
      })
    }

    // Add tool calls if present
    if (choice.message.tool_calls) {
      for (const toolCall of choice.message.tool_calls) {
        content.push({
          type: "tool-call",
          toolCallId: toolCall.id ?? generateId(),
          toolName: toolCall.function.name,
          input: toolCall.function.arguments!,
        })
      }
    }

    // Return V2 structured generation details
    return {
      content,
      finishReason: mapQwenFinishReason(choice.finish_reason),
      usage: {
        inputTokens: responseBody.usage?.prompt_tokens ?? undefined,
        outputTokens: responseBody.usage?.completion_tokens ?? undefined,
        totalTokens:
          (responseBody.usage?.prompt_tokens ?? 0)
          + (responseBody.usage?.completion_tokens ?? 0) || undefined,
      },
      ...(providerMetadata && { providerMetadata }),
      request: { body },
      response: {
        ...getResponseMetadata(responseBody),
        headers: responseHeaders,
        body: parsedBody,
      },
      warnings,
    }
  }

  /**
   * Returns a stream of model responses (V2).
   * @param options - Stream generation options.
   * @returns A promise resolving with the stream and additional metadata.
   */
  async doStream(
    options: LanguageModelV2CallOptions,
  ): Promise<Awaited<ReturnType<LanguageModelV2["doStream"]>>> {
    if (this.settings.simulateStreaming) {
      // Simulate streaming by generating a full response and splitting it.
      const result = await this.doGenerate(options)
      const simulatedStream = new ReadableStream<LanguageModelV2StreamPart>({
        start(controller) {
          controller.enqueue({
            type: "stream-start",
            warnings: result.warnings,
          })

          // Send metadata
          if (result.response) {
            controller.enqueue({
              type: "response-metadata",
              id: result.response.id,
              timestamp: result.response.timestamp,
              modelId: result.response.modelId,
            })
          }

          // Stream content parts
          for (const part of result.content) {
            if (part.type === "reasoning") {
              const id = generateId()
              controller.enqueue({ type: "reasoning-start", id })
              controller.enqueue({
                type: "reasoning-delta",
                id,
                delta: part.text,
              })
              controller.enqueue({ type: "reasoning-end", id })
            }
            else if (part.type === "text") {
              const id = generateId()
              controller.enqueue({ type: "text-start", id })
              controller.enqueue({ type: "text-delta", id, delta: part.text })
              controller.enqueue({ type: "text-end", id })
            }
            else if (part.type === "tool-call") {
              controller.enqueue({
                type: "tool-call",
                toolCallId: part.toolCallId,
                toolName: part.toolName,
                input: part.input,
              })
            }
          }

          controller.enqueue({
            type: "finish",
            finishReason: result.finishReason,
            usage: result.usage,
            providerMetadata: result.providerMetadata,
          })
          controller.close()
        },
      })
      return {
        stream: simulatedStream,
        request: result.request,
        response: result.response
          ? { headers: result.response.headers }
          : undefined,
      }
    }

    const { args, warnings } = this.getArgs(options)

    const requestBody = {
      ...args,
      stream: true,
      stream_options: {
        include_usage: true,
      },
    }

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

    const toolCalls: Array<{
      id: string
      name: string
      arguments: string
      hasFinished: boolean
    }> = []

    let finishReason: LanguageModelV2FinishReason = "unknown"
    let usage: {
      inputTokens: number | undefined
      outputTokens: number | undefined
      totalTokens: number | undefined
    } = {
      inputTokens: undefined,
      outputTokens: undefined,
      totalTokens: undefined,
    }
    let isFirstChunk = true
    let hasStreamStarted = false
    let textId: string | undefined
    let reasoningId: string | undefined

    return {
      stream: response.pipeThrough(
        new TransformStream<
          ParseResult<z.infer<typeof this.chunkSchema>>,
          LanguageModelV2StreamPart
        >({
          transform(chunk, controller) {
            if (!chunk.success) {
              controller.enqueue({ type: "error", error: chunk.error })
              return
            }
            const value = chunk.value

            metadataExtractor?.processChunk(chunk.rawValue)

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
              usage = {
                inputTokens: value.usage.prompt_tokens,
                outputTokens: value.usage.completion_tokens,
                totalTokens:
                  (value.usage.prompt_tokens ?? 0)
                  + (value.usage.completion_tokens ?? 0) || undefined,
              }
            }

            const choice = value.choices[0]

            if (choice?.finish_reason != null) {
              finishReason = mapQwenFinishReason(choice.finish_reason)
            }

            if (choice?.delta == null) {
              return
            }

            const delta = choice.delta

            // Handle reasoning content
            if (delta.reasoning_content != null) {
              if (!reasoningId) {
                reasoningId = generateId()
                controller.enqueue({
                  type: "reasoning-start",
                  id: reasoningId,
                })
              }
              controller.enqueue({
                type: "reasoning-delta",
                id: reasoningId,
                delta: delta.reasoning_content,
              })
            }

            // Handle text content
            if (delta.content != null) {
              if (!textId) {
                textId = generateId()
                controller.enqueue({ type: "text-start", id: textId })
              }
              controller.enqueue({
                type: "text-delta",
                id: textId,
                delta: delta.content,
              })
            }

            // Handle tool calls
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
                    name: toolCallDelta.function.name,
                    arguments: toolCallDelta.function.arguments ?? "",
                    hasFinished: false,
                  }

                  const toolCall = toolCalls[index]

                  // Emit tool-input-start
                  controller.enqueue({
                    type: "tool-input-start",
                    id: toolCall.id,
                    toolName: toolCall.name,
                  })

                  if (toolCall.arguments.length > 0) {
                    controller.enqueue({
                      type: "tool-input-delta",
                      id: toolCall.id,
                      delta: toolCall.arguments,
                    })
                  }

                  if (isParsableJson(toolCall.arguments)) {
                    controller.enqueue({
                      type: "tool-input-end",
                      id: toolCall.id,
                    })
                    controller.enqueue({
                      type: "tool-call",
                      toolCallId: toolCall.id,
                      toolName: toolCall.name,
                      input: toolCall.arguments,
                    })
                    toolCall.hasFinished = true
                  }

                  continue
                }

                const toolCall = toolCalls[index]

                if (toolCall.hasFinished) {
                  continue
                }

                if (toolCallDelta.function?.arguments != null) {
                  toolCall.arguments += toolCallDelta.function.arguments
                }

                controller.enqueue({
                  type: "tool-input-delta",
                  id: toolCall.id,
                  delta: toolCallDelta.function?.arguments ?? "",
                })

                if (isParsableJson(toolCall.arguments)) {
                  controller.enqueue({
                    type: "tool-input-end",
                    id: toolCall.id,
                  })
                  controller.enqueue({
                    type: "tool-call",
                    toolCallId: toolCall.id,
                    toolName: toolCall.name,
                    input: toolCall.arguments,
                  })
                  toolCall.hasFinished = true
                }
              }
            }
          },

          flush(controller) {
            // Close text and reasoning if they were started
            if (textId) {
              controller.enqueue({ type: "text-end", id: textId })
            }
            if (reasoningId) {
              controller.enqueue({ type: "reasoning-end", id: reasoningId })
            }

            // Emit final finish event
            const metadata = metadataExtractor?.buildMetadata()
            controller.enqueue({
              type: "finish",
              finishReason,
              usage,
              ...(metadata && { providerMetadata: metadata }),
            })
          },
        }),
      ),
      request: { body },
      response: { headers: responseHeaders },
    }
  }
}

// limited version of the schema, focussed on what is needed for the implementation
// this approach limits breakages when the API changes and increases efficiency
function createQwenChatChunkSchema<ERROR_SCHEMA extends z.ZodType>(
  errorSchema: ERROR_SCHEMA,
) {
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
                    type: z.literal("function").nullish(),
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
