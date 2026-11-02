import { convertReadableStreamToArray } from "@ai-sdk/provider-utils/test"
import { createTestServer } from "@ai-sdk/test-server/with-vitest"
import { describe, expect, it, vi } from "vitest"
import { z } from "zod"
import { QwenHttpClient } from "./qwen-http-client"

vi.stubEnv("DASHSCOPE_API_KEY", "test-api-key-123")

const config = {
  provider: "qwen.chat",
  url: ({ path }: { modelId: string, path: string }) =>
    `https://my.api.com/v1${path}`,
  headers: () => ({ Authorization: "Bearer test-api-key" }),
}

const responseSchema = z.object({
  id: z.string(),
  choices: z.array(z.object({
    text: z.string(),
  })),
})

describe("qwenHttpClient", () => {
  describe("generate", () => {
    const server = createTestServer({
      "https://my.api.com/v1/chat/completions": {
        response: {
          type: "json-value",
          body: {
            id: "chatcmpl-123",
            choices: [
              { text: "Hello" },
            ],
          },
        },
      },
    })

    it("should post JSON and return parsed value, raw value and response headers", async () => {
      const client = new QwenHttpClient(config)

      const result = await client.generate({
        modelId: "qwen-plus",
        path: "/chat/completions",
        body: { model: "qwen-plus", messages: [] },
        schema: responseSchema,
      })

      expect(result.value).toStrictEqual({
        id: "chatcmpl-123",
        choices: [
          { text: "Hello" },
        ],
      })
      expect(result.rawValue).toStrictEqual({
        id: "chatcmpl-123",
        choices: [
          { text: "Hello" },
        ],
      })
      expect(result.responseHeaders).toBeDefined()
    })

    it("should combine default headers with per-call headers", async () => {
      const client = new QwenHttpClient(config)

      await client.generate({
        modelId: "qwen-plus",
        path: "/chat/completions",
        body: {},
        schema: responseSchema,
        headers: { "X-Custom": "yes" },
      })

      const requestHeaders = server.calls[0].requestHeaders
      expect(requestHeaders["x-custom"]).toBe("yes")
      expect(requestHeaders.authorization).toBe("Bearer test-api-key")
    })
  })

  describe("stream", () => {
    createTestServer({
      "https://my.api.com/v1/completions": {
        response: {
          type: "stream-chunks",
          chunks: [
            "data: {\"id\":\"1\",\"choices\":[{\"text\":\"Hello\",\"index\":0,\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"1\",\"choices\":[{\"text\":\"\",\"index\":0,\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
          ],
        },
      },
    })

    it("should return a stream of parsed event-source chunks", async () => {
      const client = new QwenHttpClient(config)
      const { value } = await client.stream({
        modelId: "qwen-plus",
        path: "/completions",
        body: { model: "qwen-plus", stream: true },
        chunkSchema: z.object({
          id: z.string().nullish(),
          choices: z.array(
            z.object({
              text: z.string(),
              finish_reason: z.string().nullish(),
            }),
          ),
        }),
      })

      const chunks = await convertReadableStreamToArray(value)
      expect(chunks.filter(chunk => chunk.success).map(chunk => chunk.value)).toStrictEqual([
        {
          id: "1",
          choices: [
            { text: "Hello", finish_reason: null },
          ],
        },
        {
          id: "1",
          choices: [
            { text: "", finish_reason: "stop" },
          ],
        },
      ])
    })
  })
})
