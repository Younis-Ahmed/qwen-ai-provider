import type {
  LanguageModelV2CallWarning,
  LanguageModelV2FinishReason,
  LanguageModelV2StreamPart,
  SharedV2ProviderMetadata,
} from "@ai-sdk/provider"
import type { ParseResult } from "@ai-sdk/provider-utils"
import { getResponseMetadata } from "./get-response-metadata"
import { mapQwenFinishReason } from "./map-qwen-finish-reason"

/**
 * The shape shared by every Qwen chunk that is common to the chat and
 * completion streaming code paths. Model-specific fields are carried by the
 * richer `TVALID` type used at the call site.
 */
interface QwenChunkValue {
  id?: string | undefined | null
  created?: number | undefined | null
  model?: string | undefined | null
  usage?: {
    prompt_tokens?: number | null
    completion_tokens?: number | null
  } | null
  choices?: Array<{ finish_reason?: string | null }>
}

interface QwenStreamTransformerConfig<TVALID extends QwenChunkValue, TERROR> {
  /**
   * Warnings emitted on the first chunk via `stream-start`.
   */
  warnings: LanguageModelV2CallWarning[]

  /**
   * Extracts the error payload from a chunk, or `undefined` if the chunk is a
   * valid response. Lets callers decide how an API error chunk is shaped.
   */
  extractError: (value: TVALID | TERROR) => unknown | undefined

  /**
   * Emits the model-specific deltas for a valid chunk.
   */
  onChunk: (
    value: TVALID,
    controller: TransformStreamDefaultController<LanguageModelV2StreamPart>,
  ) => void

  /**
   * Optional hook for processing the raw (unparsed) chunk, e.g. metadata
   * extraction.
   */
  onRawChunk?: (raw: unknown) => void

  /**
   * Optional hook called on flush before the final `finish` part, e.g. to
   * close text/reasoning/tool part deltas.
   */
  onFlush?: (
    controller: TransformStreamDefaultController<LanguageModelV2StreamPart>,
  ) => void

  /**
   * Optional hook called on flush to build streaming provider metadata.
   */
  buildStreamMetadata?: () => SharedV2ProviderMetadata | undefined
}

/**
 * Builds a `Transformer` that applies the shared Qwen stream lifecycle: failed
 * parses and API error chunks become error parts, the first valid chunk emits
 * `stream-start` and `response-metadata`, usage + finish reason are accumulated
 * across chunks, and a final `finish` part closes the stream.
 *
 * Model-specific delta handling is delegated to `onChunk` (and `onFlush`).
 */
export function createQwenStreamTransformer<
  TVALID extends QwenChunkValue,
  TERROR = unknown,
>(
  config: QwenStreamTransformerConfig<TVALID, TERROR>,
): Transformer<ParseResult<TVALID | TERROR>, LanguageModelV2StreamPart> {
  let finishReason: LanguageModelV2FinishReason = "unknown"
  let usage: { promptTokens?: number, completionTokens?: number } = {}
  let isFirstChunk = true

  return {
    transform(chunk, controller) {
      // If validation fails, emit an error.
      if (!chunk.success) {
        finishReason = "error"
        controller.enqueue({ type: "error", error: chunk.error })
        return
      }

      const value = chunk.value
      config.onRawChunk?.(chunk.rawValue)

      // If the API sends an error inside the chunk.
      const error = config.extractError(value)
      if (error !== undefined) {
        finishReason = "error"
        controller.enqueue({ type: "error", error })
        return
      }

      const valid = value as TVALID

      if (isFirstChunk) {
        isFirstChunk = false
        controller.enqueue({ type: "stream-start", warnings: config.warnings })
        controller.enqueue({
          type: "response-metadata",
          ...getResponseMetadata(valid),
        })
      }

      if (valid.usage != null) {
        usage = {
          promptTokens: valid.usage.prompt_tokens ?? undefined,
          completionTokens: valid.usage.completion_tokens ?? undefined,
        }
      }

      const choice = valid.choices?.[0]
      if (choice?.finish_reason != null) {
        finishReason = mapQwenFinishReason(choice.finish_reason)
      }

      config.onChunk(valid, controller)
    },

    flush(controller) {
      config.onFlush?.(controller)

      const metadata = config.buildStreamMetadata?.()
      controller.enqueue({
        type: "finish",
        finishReason,
        usage: {
          inputTokens: usage.promptTokens ?? Number.NaN,
          outputTokens: usage.completionTokens ?? Number.NaN,
          totalTokens:
            usage.promptTokens != null && usage.completionTokens != null
              ? usage.promptTokens + usage.completionTokens
              : undefined,
        },
        ...(metadata
          ? { providerMetadata: metadata }
          : {}),
      })
    },
  }
}
