import type { LanguageModelV3Usage } from "@ai-sdk/provider"

/**
 * Build V3 usage object from token counts.
 * @internal
 */
export function buildUsage(
  inputTokens: number | undefined,
  outputTokens: number | undefined,
): LanguageModelV3Usage {
  return {
    inputTokens: {
      total: inputTokens,
      noCache: undefined,
      cacheRead: undefined,
      cacheWrite: undefined,
    },
    outputTokens: {
      total: outputTokens,
      text: undefined,
      reasoning: undefined,
    },
  }
}
