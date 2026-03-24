import { createJsonErrorResponseHandler } from "@ai-sdk/provider-utils"
import { z } from "zod"

/**
 * Schema defining the structure of a Qwen error response.
 */
const qwenErrorDataSchema = z.object({
  object: z.literal("error"),
  message: z.string(),
  type: z.string(),
  param: z.string().nullable(),
  code: z.string().nullable(),
})

export type QwenErrorData = z.infer<typeof qwenErrorDataSchema>

/**
 * Interface for defining error structures for Qwen.
 */
export interface QwenErrorStructure<T> {
  /**
   * Zod schema to validate error data.
   */
  // Use ZodType to support both zod v3.25+ and v4
  errorSchema: z.ZodType<T>
  /**
   * Maps error details to a human-readable message.
   */
  errorToMessage: (error: T) => string
  /**
   * Determines if an error is retryable.
   */
  isRetryable?: (response: Response, error?: T) => boolean
}

// Create a handler for failed responses using the defined schema.
export const qwenFailedResponseHandler = createJsonErrorResponseHandler({
  errorSchema: qwenErrorDataSchema,
  errorToMessage: error => error.message,
})

export const defaultQwenErrorStructure: QwenErrorStructure<QwenErrorData> = {
  errorSchema: qwenErrorDataSchema,
  errorToMessage: data => data.message,
}
