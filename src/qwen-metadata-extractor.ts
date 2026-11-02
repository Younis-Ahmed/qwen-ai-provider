import type { SharedV2ProviderMetadata } from "@ai-sdk/provider"

/**
 * Interface for extracting provider-specific metadata from API responses.
 */
export interface MetadataExtractor {
  /**
   * Extracts metadata from a complete (non-streaming) API response.
   * @param param0 Object containing the parsed JSON response body.
   * @returns Provider metadata if available, otherwise undefined.
   */
  extractMetadata: ({
    parsedBody,
  }: {
    parsedBody: unknown
  }) => SharedV2ProviderMetadata | undefined

  /**
   * Creates a stream extractor to process and accumulate streaming response data.
   * @returns An object with processChunk and buildMetadata methods.
   */
  createStreamExtractor: () => {
    /**
     * Processes an individual chunk from the streaming response.
     * @param parsedChunk The parsed response chunk.
     */
    processChunk: (parsedChunk: unknown) => void

    /**
     * Builds and returns the complete metadata after processing all chunks.
     * @returns Provider metadata if available, otherwise undefined.
     */
    buildMetadata: () => SharedV2ProviderMetadata | undefined
  }
}
