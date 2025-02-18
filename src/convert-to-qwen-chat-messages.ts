import type {
  LanguageModelV1Prompt,
  LanguageModelV1ProviderMetadata,
} from "@ai-sdk/provider"
import type { QwenChatPrompt } from "./qwen-api-types"
import {
  UnsupportedFunctionalityError,
} from "@ai-sdk/provider"
import { convertUint8ArrayToBase64 } from "@ai-sdk/provider-utils"

function getQwenMetadata(message: {
  providerMetadata?: LanguageModelV1ProviderMetadata
}) {
  return message?.providerMetadata?.qwen ?? {}
}

export function convertToQwenChatMessages(
  prompt: LanguageModelV1Prompt,
): QwenChatPrompt {
  const messages: QwenChatPrompt = []
  for (const { role, content, ...message } of prompt) {
    const metadata = getQwenMetadata({ ...message })
    switch (role) {
      case "system": {
        messages.push({ role: "system", content, ...metadata })
        break
      }

      case "user": {
        if (content.length === 1 && content[0].type === "text") {
          messages.push({
            role: "user",
            content: content[0].text,
            ...getQwenMetadata(content[0]),
          })
          break
        }

        messages.push({
          role: "user",
          content: content.map((part) => {
            const partMetadata = getQwenMetadata(part)
            switch (part.type) {
              case "text": {
                return { type: "text", text: part.text, ...partMetadata }
              }
              case "image": {
                return {
                  type: "image_url",
                  image_url: {
                    url:
                        part.image instanceof URL
                          ? part.image.toString()
                          : `data:${
                            part.mimeType ?? "image/jpeg"
                          };base64,${convertUint8ArrayToBase64(part.image)}`,
                  },
                  ...partMetadata,
                }
              }
              default: {
                // file part type is not supported
                throw new UnsupportedFunctionalityError({
                  functionality: "File content parts in user messages",
                })
              }
            }
          }),
          ...metadata,
        })

        break
      }

      case "assistant": {
        let text = ""
        const toolCalls: Array<{
          id: string
          type: "function"
          function: { name: string, arguments: string }
        }> = []

        for (const part of content) {
          const partMetadata = getQwenMetadata(part)
          switch (part.type) {
            case "text": {
              text += part.text
              break
            }
            case "tool-call": {
              toolCalls.push({
                id: part.toolCallId,
                type: "function",
                function: {
                  name: part.toolName,
                  arguments: JSON.stringify(part.args),
                },
                ...partMetadata,
              })
              break
            }
            default: {
              const _exhaustiveCheck: never = part
              throw new Error(`Unsupported part: ${_exhaustiveCheck}`)
            }
          }
        }

        messages.push({
          role: "assistant",
          content: text,
          tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
          ...metadata,
        })

        break
      }

      case "tool": {
        for (const toolResponse of content) {
          const toolResponseMetadata = getQwenMetadata(toolResponse)
          messages.push({
            role: "tool",
            tool_call_id: toolResponse.toolCallId,
            content: JSON.stringify(toolResponse.result),
            ...toolResponseMetadata,
          })
        }
        break
      }

      default: {
        const _exhaustiveCheck: never = role
        throw new Error(`Unsupported role: ${_exhaustiveCheck}`)
      }
    }
  }

  return messages
}
