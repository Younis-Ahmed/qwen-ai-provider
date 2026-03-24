import type { LanguageModelV2FinishReason } from "@ai-sdk/provider"

/**
 * Maps the finish reason from the backend response to a standardized format.
 *
 * @param finishReason - The original finish reason string.
 * @returns The mapped LanguageModelV2FinishReason.
 */
export function mapQwenFinishReason(
  finishReason: string | null | undefined,
): LanguageModelV2FinishReason {
  switch (finishReason) {
    case "stop":
      return "stop"
    case "length":
      return "length"
    case "tool_calls":
      return "tool-calls"
    default:
      return "unknown"
  }
}
