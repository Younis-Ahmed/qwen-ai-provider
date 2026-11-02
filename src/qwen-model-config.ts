import type { FetchFunction } from "@ai-sdk/provider-utils"
import type { QwenErrorStructure } from "./qwen-error"

/**
 * Shared transport configuration for every Qwen model adapter.
 *
 * This is the single place the HTTP transport contract lives (URL building,
 * headers, fetch injection and error handling). Model adapters extend it only
 * with the fields that are genuinely specific to their kind.
 */
export interface QwenModelConfig {
  /**
   * The provider identifier, e.g. `qwen.chat`. Its first dot-segment is used
   * to look up provider options.
   */
  provider: string

  /**
   * Builds the request URL for a given model and API path.
   */
  url: (options: { modelId: string, path: string }) => string

  /**
   * Returns the default request headers (authentication, base headers).
   */
  headers: () => Record<string, string | undefined>

  /**
   * Custom fetch implementation (e.g. for testing or interception).
   */
  fetch?: FetchFunction

  /**
   * Error structure used to decode API error responses.
   */
  errorStructure?: QwenErrorStructure<any>
}
