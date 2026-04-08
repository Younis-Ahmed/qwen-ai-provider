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
import type { QwenChatModelId, QwenChatSettings } from "../config/chat"
import type { QwenErrorStructure } from "../error"
import type { MetadataExtractor } from "../utils/metadata-extractor"
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
import { defaultQwenErrorStructure } from "../error"
import { buildUsage } from "../utils/build-usage"
import { convertToQwenChatMessages } from "../utils/convert-to-chat-messages"
import { getResponseMetadata } from "../utils/get-response-metadata"
import { mapQwenFinishReason } from "../utils/map-finish-reason"

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
        reasoning: z.string().nullish(),
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
 * A language model implementation for Qwen Chat API that follows the LanguageModelV3 interface.
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
export class QwenChatLanguageModel implements LanguageModelV3 {
  readonly specificationVersion = "v3"

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
   * Generates the arguments and warnings required for a language model generation call (V3).
   *
   * @param options - V3 call options
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
  }: LanguageModelV3CallOptions) {
    const warnings: SharedV3Warning[] = []

    // Warn if unsupported settings are used:
    if (topK != null) {
      warnings.push({
        type: "unsupported",
        feature: "topK",
      })
    }

    if (
      responseFormat?.type === "json"
      && responseFormat.schema != null
      && !this.supportsStructuredOutputs
    ) {
      warnings.push({
        type: "unsupported",
        feature: "responseFormat",
        details:
          "JSON response format schema is only supported with structuredOutputs",
      })
    }

    // Convert V3 tools to OpenAI format
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
          type: "unsupported",
          feature: `tool type: ${tool.type}`,
        })
        return null
      })
      .filter((t): t is NonNullable<typeof t> => t !== null)

    // Convert V3 tool choice to OpenAI format
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

      // Fix: It will not output content
      extra_body: { enable_thinking: false },

      // messages:
      messages: convertToQwenChatMessages(prompt),

      // tools:
      tools: openaiTools && openaiTools.length > 0 ? openaiTools : undefined,
      tool_choice: openaiToolChoice,
    }

    return { args, warnings }
  }

  /**
   * Generates a text response from the model (V3).
   * @param options - Generation options.
   * @returns A promise resolving with the generation result.
   */
  async doGenerate(
    options: LanguageModelV3CallOptions,
  ): Promise<Awaited<ReturnType<LanguageModelV3["doGenerate"]>>> {
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

    // Build V3 content array
    const content: LanguageModelV3Content[] = []

    // Add reasoning content if present
    const reasoningText = choice.message.reasoning_content ?? choice.message.reasoning
    if (reasoningText) {
      content.push({
        type: "reasoning",
        text: reasoningText,
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
        const normalizedToolInput = normalizeToolCallArguments(
          toolCall.function.arguments,
        )
        content.push({
          type: "tool-call",
          toolCallId: toolCall.id ?? generateId(),
          toolName: toolCall.function.name,
          input: normalizedToolInput ?? toolCall.function.arguments,
        })
      }
    }
    else {
      // Fallback: some providers put pseudo tool tags into reasoning/content.
      const taggedToolCalls = extractToolCallsFromTaggedText(
        `${reasoningText ?? ""}\n${choice.message.content ?? ""}`,
      )
      for (const toolCall of taggedToolCalls) {
        content.push({
          type: "tool-call",
          toolCallId: generateId(),
          toolName: toolCall.toolName,
          input: toolCall.input,
        })
      }
    }

    // Return V3 structured generation details
    return {
      content,
      finishReason: mapQwenFinishReason(choice.finish_reason),
      usage: buildUsage(
        responseBody.usage?.prompt_tokens ?? undefined,
        responseBody.usage?.completion_tokens ?? undefined,
      ),
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
   * Returns a stream of model responses (V3).
   * @param options - Stream generation options.
   * @returns A promise resolving with the stream and additional metadata.
   */
  async doStream(
    options: LanguageModelV3CallOptions,
  ): Promise<Awaited<ReturnType<LanguageModelV3["doStream"]>>> {
    if (this.settings.simulateStreaming) {
      // Simulate streaming by generating a full response and splitting it.
      const result = await this.doGenerate(options)
      const simulatedStream = new ReadableStream<LanguageModelV3StreamPart>({
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

    let finishReason: LanguageModelV3FinishReason = { unified: "other", raw: undefined }
    let usage: LanguageModelV3Usage = buildUsage(undefined, undefined)
    let isFirstChunk = true
    let hasStreamStarted = false
    let textId: string | undefined
    let reasoningId: string | undefined
    let accumulatedReasoning = ""
    let accumulatedText = ""

    return {
      stream: response.pipeThrough(
        new TransformStream<
          ParseResult<z.infer<typeof this.chunkSchema>>,
          LanguageModelV3StreamPart
        >({
          transform(chunk, controller) {
            if (!chunk.success) {
              if (shouldRetryQwenStreamRequest(chunk.error)) {
                if (recoverPendingToolCalls(toolCalls, controller)) {
                  finishReason = mapQwenFinishReason("tool_calls")
                }
                // Qwen may emit a malformed terminal error chunk after content or tool deltas.
                // Treat this known provider error as non-fatal and let stream flush finalize.
                return
              }
              controller.enqueue({ type: "error", error: chunk.error })
              return
            }
            const value = chunk.value

            metadataExtractor?.processChunk(chunk.rawValue)

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

            const delta = choice?.delta

            if (delta == null)
              return

            // Handle reasoning content
            const reasoningText = delta.reasoning_content ?? delta.reasoning
            if (reasoningText != null) {
              accumulatedReasoning += reasoningText
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
                delta: reasoningText,
              })
            }

            // Handle text content
            if (delta.content != null) {
              accumulatedText += delta.content
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

            // Finalize any remaining tool calls at end-of-stream.
            for (let i = 0; i < toolCalls.length; i++) {
              const toolCall = toolCalls[i]
              if (toolCall == null || toolCall.hasFinished)
                continue
              const parsed = parseToolCallArguments(toolCall.arguments, {
                phase: "final",
              })
              if (parsed == null) {
                continue
              }

              controller.enqueue({
                type: "tool-input-end",
                id: toolCall.id,
              })
              controller.enqueue({
                type: "tool-call",
                toolCallId: toolCall.id,
                toolName: toolCall.name,
                input: parsed,
              })
              toolCall.hasFinished = true
            }

            const hasEmittedStructuredToolCall = toolCalls.some(tc => tc?.hasFinished)
            if (!hasEmittedStructuredToolCall) {
              const taggedToolCalls = extractToolCallsFromTaggedText(
                `${accumulatedReasoning}\n${accumulatedText}`,
              )
              for (const taggedToolCall of taggedToolCalls) {
                controller.enqueue({
                  type: "tool-call",
                  toolCallId: generateId(),
                  toolName: taggedToolCall.toolName,
                  input: taggedToolCall.input,
                })
              }
              if (taggedToolCalls.length > 0)
                finishReason = mapQwenFinishReason("tool_calls")
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
              reasoning: z.string().nullish(),
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

function parseToolCallArguments(
  rawArguments: string | null | undefined,
  options?: { phase?: "streaming" | "final" },
) {
  const phase = options?.phase ?? "streaming"

  // 1. 快速处理空值
  if (rawArguments == null)
    return null
  const input = rawArguments.trim()
  if (!input)
    return phase === "final" ? "{}" : null

  // 2. 尝试标准解析流程（JSON & Tag）
  const normalized = tryNormalizeJson(input)
  if (normalized != null)
    return normalized

  const parsedTagStrict = parseTagArgumentsAsJson(input, { requireComplete: true })
  if (parsedTagStrict != null)
    return parsedTagStrict

  // 流式处理阶段：若非标准格式则推迟，避免解析不完整片段
  if (phase !== "final")
    return null

  // 3. Final 阶段的容错解析
  // 尝试宽松的 Tag 解析
  const looseTag = parseTagArgumentsAsJson(input, { requireComplete: false })
  if (looseTag != null)
    return looseTag

  // 4. 启发式修复逻辑：处理缺失的花括号
  const hasStart = input.startsWith("{")
  const hasEnd = input.endsWith("}")
  const candidates: string[] = []

  if (!hasStart && !hasEnd) {
    candidates.push(`{${input}}`)
  }
  else if (hasStart && !hasEnd) {
    candidates.push(`${input}}`)
  }
  else if (!hasStart && hasEnd) {
    candidates.push(`{${input}}`, `{${input}`) // 优先尝试完全包装
  }

  for (const cand of candidates) {
    if (isParsableJson(cand))
      return cand
  }

  // 5. 噪声过滤 (例如 "}{" 或 "  {")
  if (/^[ \t\n{}]+$/.test(input))
    return "{}"

  // 6. 最终兜底：括号平衡补全
  // 仅计算非转义的括号，虽然简单但能处理 90% 的截断场景
  const openCount = (input.match(/\{/g) || []).length
  const closeCount = (input.match(/\}/g) || []).length
  const balance = openCount - closeCount

  if (balance !== 0) {
    const fixed = balance > 0
      ? input + "}".repeat(balance)
      : "{".repeat(Math.abs(balance)) + input

    if (isParsableJson(fixed)) {
      // 保持返回类型一致为 string
      return typeof fixed === "string"
        ? fixed
        : JSON.stringify(fixed)
    }
  }

  return null
}

function normalizeToolCallArguments(rawArguments: string | null | undefined) {
  if (rawArguments == null)
    return null

  if (!rawArguments.trim())
    return rawArguments

  const normalized = tryNormalizeJson(rawArguments)
  if (normalized != null)
    return normalized

  const parsedFromTags = parseTagArgumentsAsJson(rawArguments, { requireComplete: false })
  if (parsedFromTags != null)
    return parsedFromTags

  // Python proxy fallback: non-JSON/non-tag inputs become {}
  return "{}"
}

function extractToolCallsFromTaggedText(input: string | null | undefined) {
  if (!input)
    return []

  const toolCallPattern = /<tool_call>([\s\S]*?)<\/tool_call>/g
  const result: Array<{ toolName: string, input: string }> = []

  for (const match of input.matchAll(toolCallPattern)) {
    const body = match[1]?.trim()
    if (!body)
      continue

    const functionMatch = body.match(/<function=([\w-]+)>/i)
    const toolName = functionMatch?.[1]?.trim()
    if (!toolName)
      continue

    const args: Record<string, unknown> = {}
    const parameterPattern = /<parameter=([\w-]+)>([\s\S]*?)(?=<parameter=[\w-]+>|$)/g
    for (const [, rawKey, rawValue] of body.matchAll(parameterPattern)) {
      const key = rawKey.trim()
      const value = rawValue.trim()
      if (!key)
        continue

      if (key === "params" && isParsableJson(value)) {
        args[key] = JSON.parse(value)
      }
      else {
        args[key] = value
      }
    }

    result.push({
      toolName,
      input: JSON.stringify(args),
    })
  }

  return result
}

function tryNormalizeJson(input: string) {
  try {
    const parsed = JSON.parse(input)
    return JSON.stringify(parsed)
  }
  catch {
    return null
  }
}

function parseTagArgumentsAsJson(
  input: string,
  options: { requireComplete: boolean },
) {
  const pattern = /<parameter=(\w+)>([\s\S]*?)(?:<\/parameter>|$)/g
  const args: Record<string, string> = {}

  const matches = Array.from(input.matchAll(pattern))
  if (matches.length === 0)
    return null

  // 1. 完整性校验逻辑优化
  if (options.requireComplete) {
    // 检查是否所有匹配项都有闭合标签 (即 match[0] 必须以 </parameter> 结尾)
    const allClosed = matches.every(m => m[0].endsWith("</parameter>"))
    // 检查是否有函数级的结束标志
    // eslint-disable-next-line regexp/no-unused-capturing-group
    const hasFunctionEnd = /<\/(function|tool_call)>/.test(input)

    if (!allClosed || !hasFunctionEnd) {
      return null
    }
  }

  // 2. 提取并清洗数据
  for (const [, key, value] of matches) {
    args[key.trim()] = value.trim()
  }

  // 3. 最终结果转换
  // 如果没有任何参数被解析（例如只有标签头），返回 null
  return Object.keys(args).length > 0 ? JSON.stringify(args) : null
}

function shouldRetryQwenStreamRequest(error: unknown): boolean {
  const base = error instanceof Error ? error.message : String(error)
  const causeMessage
    = error instanceof Error
      && (error as any).cause
      && typeof (error as any).cause === "object"
      && "message" in (error as any).cause
      ? String((error as any).cause.message)
      : ""
  const nestedErrorMessage
    = error != null
      && typeof error === "object"
      && "error" in (error as any)
      && (error as any).error
      && typeof (error as any).error === "object"
      && "message" in (error as any).error
      ? String((error as any).error.message)
      : ""
  const normalized = `${base} ${causeMessage} ${nestedErrorMessage}`.toLowerCase()
  const isKnownServerCrash = normalized.includes("list index out of range")
  const isInternalServerError
    = normalized.includes("internalservererror")
      || normalized.includes("\"code\":500")
      || normalized.includes("status code 500")
      || normalized.includes("internal server error")
  const isTypeValidationBoundaryFailure
    = normalized.includes("type validation failed")
      || normalized.includes("ai_typevalidationerror")
  return isKnownServerCrash && (isInternalServerError || isTypeValidationBoundaryFailure)
}

function recoverPendingToolCalls(
  toolCalls: Array<{ id: string, name: string, arguments: string, hasFinished: boolean }>,
  controller: TransformStreamDefaultController<LanguageModelV3StreamPart>,
): boolean {
  let recoveredAnyToolCall = false
  for (const toolCall of toolCalls) {
    if (toolCall?.hasFinished)
      continue
    const recoveredInput = parseToolCallArguments(toolCall?.arguments, {
      phase: "final",
    })
    if (recoveredInput == null || toolCall?.id == null || toolCall?.name == null) {
      continue
    }
    controller.enqueue({
      type: "tool-input-end",
      id: toolCall.id,
    })
    controller.enqueue({
      type: "tool-call",
      toolCallId: toolCall.id,
      toolName: toolCall.name,
      input: recoveredInput,
    })
    toolCall.hasFinished = true
    recoveredAnyToolCall = true
  }
  return recoveredAnyToolCall
}
