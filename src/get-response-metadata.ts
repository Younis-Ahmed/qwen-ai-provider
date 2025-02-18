/**
 * Generates metadata for a response object.
 *
 * @param {object} params - Input parameters.
 * @param {string | undefined | null} params.id - A unique identifier for the response.
 * @param {string | undefined | null} params.model - Identifier for the model used.
 * @param {number | undefined | null} params.created - Unix timestamp (in seconds) of when the response was created.
 * @returns {object} An object containing normalized metadata.
 */
export function getResponseMetadata({
  id,
  model,
  created,
}: {
  id?: string | undefined | null
  created?: number | undefined | null
  model?: string | undefined | null
}) {
  // Normalize and construct the response metadata object.
  return {
    // Assign 'id' if provided; otherwise, leave as undefined.
    id: id ?? undefined,
    // Map 'model' to 'modelId' for improved clarity; assign if provided.
    modelId: model ?? undefined,
    // If 'created' is provided, convert the Unix timestamp (seconds) to a JavaScript Date object.
    timestamp: created != null ? new Date(created * 1000) : undefined,
  }
}
