import type { ZodSchema } from "zod"
import { createJsonErrorResponseHandler } from "@ai-sdk/provider-utils"
import { z } from "zod"

const qwenErrorDataSchema = z.object({
  object: z.literal("error"),
  message: z.string(),
  type: z.string(),
  param: z.string().nullable(),
  code: z.string().nullable(),
})

export type QwenErrorData = z.infer<typeof qwenErrorDataSchema>

export interface QwenErrorStructure<T> {
  errorSchema: ZodSchema<T>
  errorToMessage: (error: T) => string
  isRetryable?: (response: Response, error?: T) => boolean
}

export const qwenFailedResponseHandler = createJsonErrorResponseHandler({
  errorSchema: qwenErrorDataSchema,
  errorToMessage: error => error.message,
})

export const defaultQwenErrorStructure: QwenErrorStructure<QwenErrorData>
  = {
    errorSchema: qwenErrorDataSchema,
    errorToMessage: data => data.message,
  }
