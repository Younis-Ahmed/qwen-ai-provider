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

export const qwenFailedResponseHandler = createJsonErrorResponseHandler({
  errorSchema: qwenErrorDataSchema,
  errorToMessage: error => error.message,
})
