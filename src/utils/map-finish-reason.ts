import type { LanguageModelV3FinishReason } from "@ai-sdk/provider"

/**
 * Normalizes Qwen/OpenAI-style `finish_reason` strings for {@link LanguageModelV3FinishReason}.
 *
 * **Used by:** `QwenChatLanguageModel`, `QwenCompletionLanguageModel` (generate + stream).
 *
 * **Not applicable:** embedding and reranking APIs do not return `choices[].finish_reason`;
 * those models map success payloads only and surface HTTP/JSON errors without retries.
 *
 * @param finishReason - The original finish reason string.
 * @returns The mapped LanguageModelV3FinishReason.
 */
export function mapQwenFinishReason(
  finishReason: string | null | undefined,
): LanguageModelV3FinishReason {
  let unified: LanguageModelV3FinishReason["unified"]

  switch (finishReason) {
    case "stop":
      unified = "stop"
      break
    case "length":
      unified = "length"
      break
    case "tool_calls":
      unified = "tool-calls"
      break
    case "content_filter":
      unified = "content-filter"
      break
    default:
      unified = "other"
  }

  return {
    unified,
    raw: finishReason ?? undefined,
  }
}
