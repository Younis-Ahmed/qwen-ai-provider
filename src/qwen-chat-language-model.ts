import type {
  LanguageModelV2,
  LanguageModelV2CallWarning,
  LanguageModelV2Content,
  LanguageModelV2StreamPart,
} from "@ai-sdk/provider"
import type {
  QwenChatModelId,
  QwenChatSettings,
} from "./qwen-chat-settings"
import type { MetadataExtractor } from "./qwen-metadata-extractor"
import type { QwenModelConfig } from "./qwen-model-config"
import {
  InvalidResponseDataError,
} from "@ai-sdk/provider"
import {
  generateId,
  isParsableJson,
} from "@ai-sdk/provider-utils"
import { z } from "zod"
import { convertToQwenChatMessages } from "./convert-to-qwen-chat-messages"
import { getResponseMetadata } from "./get-response-metadata"
import { mapQwenFinishReason } from "./map-qwen-finish-reason"
import { QwenHttpClient } from "./qwen-http-client"
import { prepareTools } from "./qwen-prepare-tools"
import { createQwenStreamTransformer } from "./qwen-stream"

/**
 * Configuration for the Qwen Chat Language Model.
 * @interface QwenChatConfig
 */
export interface QwenChatConfig extends QwenModelConfig {
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
  readonly specificationVersion = "v2" as const

  readonly supportedUrls: Record<string, RegExp[]> = {
    "image/*": [/^https?:\/\/.*$/],
    "video/*": [/^https?:\/\/.*$/],
    "audio/*": [/^https?:\/\/.*$/],
    "application/pdf": [/^https?:\/\/.*$/],
  }

  readonly supportsStructuredOutputs: boolean

  readonly modelId: QwenChatModelId
  readonly settings: QwenChatSettings

  private readonly config: QwenChatConfig
  readonly http: QwenHttpClient
  readonly chunkSchema // type inferred via constructor

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
    this.http = new QwenHttpClient(config)
    this.chunkSchema = createQwenChatChunkSchema(
      this.http.baseErrorStructure.errorSchema,
    )
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
  get providerOptionsName(): string {
    return this.config.provider.split(".")[0].trim()
  }

  /**
   * Generates the arguments and warnings required for a language model generation call.
   *
   * This function prepares the argument object based on the provided generation options,
   * including any necessary warnings for unsupported settings.
   *
   * @param options - The language model generation options.
   * @param options.prompt - The prompt input used to generate chat messages.
   * @param options.maxOutputTokens - The maximum number of tokens to generate.
   * @param options.temperature - The temperature setting to control randomness in generation.
   * @param options.topP - The nucleus sampling parameter (top-p) for token selection.
   * @param options.topK - The top-k sampling parameter; if provided, it triggers a warning as it is unsupported.
   * @param options.frequencyPenalty - The penalty applied to frequently occurring tokens.
   * @param options.presencePenalty - The penalty applied based on the presence of tokens.
   * @param options.providerOptions - Additional options customized for the specific provider.
   * @param options.stopSequences - An array of sequences that will signal the generation to stop.
   * @param options.responseFormat - The desired response format; supports JSON schema formatting when structured outputs are enabled.
   * @param options.seed - An optional seed value for randomization.
   *
   * @returns An object containing:
   * - args: The arguments constructed for the language model generation request.
   * - warnings: A list of warnings related to unsupported or deprecated settings.
   */
  getArgs(options: Parameters<LanguageModelV2["doGenerate"]>[0]) {
    const warnings: LanguageModelV2CallWarning[] = []

    // Warn if unsupported settings are used:
    if (options.topK != null) {
      warnings.push({
        type: "unsupported-setting",
        setting: "topK",
      })
    }

    const baseArgs = {
      // model id:
      model: this.modelId,

      // model specific settings:
      user: this.settings.user,

      // standardized settings:
      max_tokens: options.maxOutputTokens,
      temperature: options.temperature,
      top_p: options.topP,
      frequency_penalty: options.frequencyPenalty,
      presence_penalty: options.presencePenalty,
      stop: options.stopSequences,
      seed: options.seed,
      ...options.providerOptions?.[this.providerOptionsName],

      ...options.responseFormat?.type === "json"
        ? {
            response_format: this.supportsStructuredOutputs === true
              && options.responseFormat.schema != null
              ? {
                  type: "json_schema",
                  json_schema: {
                    schema: options.responseFormat.schema,
                    name: options.responseFormat.name ?? "response",
                    description: options.responseFormat.description,
                  },
                }
              : { type: "json_object" },
          }
        : {},

      // messages:
      messages: convertToQwenChatMessages(options.prompt),
    }

    const { tools, tool_choice, toolWarnings } = prepareTools({
      tools: options.tools,
      toolChoice: options.toolChoice,
    })

    return {
      args: { ...baseArgs, tools, tool_choice },
      warnings: [
        ...warnings,
        ...toolWarnings,
      ],
    }
  }

  /**
   * Generates a text response from the model.
   * @param options - Generation options.
   * @returns A promise resolving with the generation result.
   */
  async doGenerate(
    options: Parameters<LanguageModelV2["doGenerate"]>[0],
  ): Promise<Awaited<ReturnType<LanguageModelV2["doGenerate"]>>> {
    const { args, warnings } = this.getArgs(options)

    const body = JSON.stringify(args)

    const {
      responseHeaders,
      value: responseBody,
      rawValue: parsedBody,
    } = await this.http.generate({
      modelId: this.modelId,
      path: "/chat/completions",
      body: args,
      schema: QwenChatResponseSchema,
      headers: options.headers,
      abortSignal: options.abortSignal,
    })

    const choice = responseBody.choices[0]
    const providerMetadata = this.config.metadataExtractor?.extractMetadata?.({
      parsedBody,
    })

    const content: LanguageModelV2Content[] = []

    if (choice.message.reasoning_content != null) {
      content.push({
        type: "reasoning",
        text: choice.message.reasoning_content,
      })
    }

    if (choice.message.content != null) {
      content.push({
        type: "text",
        text: choice.message.content,
      })
    }

    if (choice.message.tool_calls != null) {
      for (const toolCall of choice.message.tool_calls) {
        content.push({
          type: "tool-call",
          toolCallId: toolCall.id ?? generateId(),
          toolName: toolCall.function.name,
          input: toolCall.function.arguments,
        })
      }
    }

    if (content.length === 0) {
      throw new InvalidResponseDataError({
        data: responseBody,
        message: "The model did not return any content.",
      })
    }

    return {
      content,
      finishReason: mapQwenFinishReason(choice.finish_reason),
      usage: {
        inputTokens: responseBody.usage?.prompt_tokens ?? Number.NaN,
        outputTokens: responseBody.usage?.completion_tokens ?? Number.NaN,
        totalTokens:
          responseBody.usage?.prompt_tokens != null
          && responseBody.usage?.completion_tokens != null
            ? responseBody.usage.prompt_tokens
            + responseBody.usage.completion_tokens
            : undefined,
      },
      ...(providerMetadata && { providerMetadata }),
      request: { body },
      response: {
        ...getResponseMetadata(responseBody),
        headers: responseHeaders,
      },
      warnings,
    }
  }

  /**
   * Returns a stream of model responses.
   * @param options - Stream generation options.
   * @returns A promise resolving with the stream and additional metadata.
   */
  async doStream(
    options: Parameters<LanguageModelV2["doStream"]>[0],
  ): Promise<Awaited<ReturnType<LanguageModelV2["doStream"]>>> {
    if (this.settings.simulateStreaming) {
      // Simulate streaming by generating a full response and splitting it.
      const result = await this.doGenerate(options)
      const simulatedStream = this.createSimulatedStream(result)
      return {
        stream: simulatedStream,
        request: {
          body: JSON.stringify({ ...options, stream: true }),
        },
        response: result.response,
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

    // Set stream flag to true for the API.
    const body = JSON.stringify(requestBody)
    const metadataExtractor
      = this.config.metadataExtractor?.createStreamExtractor()

    const { responseHeaders, value: response } = await this.http.stream({
      modelId: this.modelId,
      path: "/chat/completions",
      body: requestBody,
      chunkSchema: this.chunkSchema,
      headers: options.headers,
      abortSignal: options.abortSignal,
    })

    const toolCalls: Array<{
      id: string
      type: "function"
      function: {
        name: string
        arguments: string
      }
      hasFinished: boolean
    }> = []

    const toolInputStreamIds: Array<string> = []

    const textStreamId = `text-${generateId()}`
    const reasoningStreamId = `reasoning-${generateId()}`
    let sentTextStart = false
    let sentReasoningStart = false

    return {
      stream: response.pipeThrough(
        new TransformStream(
          createQwenStreamTransformer({
            warnings,
            onRawChunk: raw => metadataExtractor?.processChunk(raw),
            extractError: (value) => {
              const v = value
              return "error" in v
                ? v.error?.message
                : undefined
            },
            onChunk: (value, controller) => {
              const choice = value.choices[0]

              if (choice?.delta == null) {
                return
              }

              const delta = choice.delta

              // Emit reasoning before the main content.
              if (delta.reasoning_content != null) {
                if (!sentReasoningStart) {
                  sentReasoningStart = true
                  controller.enqueue({
                    type: "reasoning-start",
                    id: reasoningStreamId,
                  })
                }

                controller.enqueue({
                  type: "reasoning-delta",
                  id: reasoningStreamId,
                  delta: delta.reasoning_content,
                })
              }

              if (delta.content != null) {
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
                  delta: delta.content,
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
                    const toolInputStreamId = `tool-input-${generateId()}`
                    toolInputStreamIds[index] = toolInputStreamId

                    controller.enqueue({
                      type: "tool-input-start",
                      id: toolInputStreamId,
                      toolName: toolCall.function.name,
                    })

                    if (toolCall.function.arguments.length > 0) {
                      controller.enqueue({
                        type: "tool-input-delta",
                        id: toolInputStreamId,
                        delta: toolCall.function.arguments,
                      })
                    }

                    if (isParsableJson(toolCall.function.arguments)) {
                      controller.enqueue({
                        type: "tool-input-end",
                        id: toolInputStreamId,
                      })
                      controller.enqueue({
                        type: "tool-call",
                        toolCallId: toolCall.id ?? generateId(),
                        toolName: toolCall.function.name,
                        input: toolCall.function.arguments,
                      })
                      toolCall.hasFinished = true
                    }

                    continue
                  }

                  // Merge new delta with ongoing tool call.
                  const toolCall = toolCalls[index]
                  const toolInputStreamId = toolInputStreamIds[index]

                  if (toolCall.hasFinished) {
                    continue
                  }

                  if (toolCallDelta.function?.arguments != null) {
                    toolCall.function.arguments
                      += toolCallDelta.function?.arguments ?? ""
                  }

                  controller.enqueue({
                    type: "tool-input-delta",
                    id: toolInputStreamId,
                    delta: toolCallDelta.function.arguments ?? "",
                  })

                  if (
                    toolCall.function?.name != null
                    && toolCall.function?.arguments != null
                    && isParsableJson(toolCall.function.arguments)
                  ) {
                    controller.enqueue({
                      type: "tool-input-end",
                      id: toolInputStreamId,
                    })
                    controller.enqueue({
                      type: "tool-call",
                      toolCallId: toolCall.id ?? generateId(),
                      toolName: toolCall.function.name,
                      input: toolCall.function.arguments,
                    })
                    toolCall.hasFinished = true
                  }
                }
              }
            },
            onFlush: (controller) => {
              if (sentTextStart) {
                controller.enqueue({ type: "text-end", id: textStreamId })
              }

              if (sentReasoningStart) {
                controller.enqueue({
                  type: "reasoning-end",
                  id: reasoningStreamId,
                })
              }
            },
            buildStreamMetadata: () => metadataExtractor?.buildMetadata(),
          }),
        ),
      ),
      request: { body },
      response: { headers: responseHeaders },
    }
  }

  createSimulatedStream(result: Awaited<ReturnType<LanguageModelV2["doGenerate"]>>) {
    const streamId = `text-${generateId()}`
    const reasoningStreamId = `reasoning-${generateId()}`

    return new ReadableStream<LanguageModelV2StreamPart>({
      start(controller) {
        controller.enqueue({
          type: "stream-start",
          warnings: result.warnings,
        })

        controller.enqueue({
          type: "response-metadata",
          id: result.response?.id,
          timestamp: result.response?.timestamp,
          modelId: result.response?.modelId,
        })

        let sawText = false
        let sawReasoning = false

        for (const part of result.content) {
          switch (part.type) {
            case "reasoning": {
              if (!sawReasoning) {
                sawReasoning = true
                controller.enqueue({
                  type: "reasoning-start",
                  id: reasoningStreamId,
                })
              }

              controller.enqueue({
                type: "reasoning-delta",
                id: reasoningStreamId,
                delta: part.text,
              })
              break
            }

            case "text": {
              if (!sawText) {
                sawText = true
                controller.enqueue({ type: "text-start", id: streamId })
              }

              controller.enqueue({
                type: "text-delta",
                id: streamId,
                delta: part.text,
              })
              break
            }

            case "tool-call": {
              const toolInputId = `tool-input-${generateId()}`
              controller.enqueue({
                type: "tool-input-start",
                id: toolInputId,
                toolName: part.toolName,
              })
              controller.enqueue({
                type: "tool-input-delta",
                id: toolInputId,
                delta: part.input,
              })
              controller.enqueue({
                type: "tool-input-end",
                id: toolInputId,
              })
              controller.enqueue({
                type: "tool-call",
                toolCallId: part.toolCallId,
                toolName: part.toolName,
                input: part.input,
              })
              break
            }
          }
        }

        if (sawText) {
          controller.enqueue({ type: "text-end", id: streamId })
        }

        if (sawReasoning) {
          controller.enqueue({
            type: "reasoning-end",
            id: reasoningStreamId,
          })
        }

        controller.enqueue({
          type: "finish",
          finishReason: result.finishReason,
          usage: result.usage,
          ...(result.providerMetadata
            && { providerMetadata: result.providerMetadata }),
        })

        controller.close()
      },
    })
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
