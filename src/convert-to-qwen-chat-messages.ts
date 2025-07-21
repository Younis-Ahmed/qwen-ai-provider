import type {
  LanguageModelV1Prompt,
  LanguageModelV1ProviderMetadata,
} from "@ai-sdk/provider"
import type { QwenChatPrompt } from "./qwen-api-types"
import {
  UnsupportedFunctionalityError,
} from "@ai-sdk/provider"
import { convertUint8ArrayToBase64 } from "@ai-sdk/provider-utils"

// JSDoc for helper function to extract Qwen metadata.
/**
 * Extracts Qwen-specific metadata from a message.
 *
 * @param message - An object that may contain providerMetadata
 * @param message.providerMetadata - Provider-specific metadata containing Qwen configuration
 * @returns The Qwen metadata object or an empty object if none exists
 */

function getQwenMetadata(message: {
  providerMetadata?: LanguageModelV1ProviderMetadata
}) {
  return message?.providerMetadata?.qwen ?? {}
}

/**
 * Converts a generic language model prompt to Qwen chat messages.
 *
 * @param prompt The language model prompt to convert.
 * @returns An array of Qwen chat messages.
 */
export function convertToQwenChatMessages(
  prompt: LanguageModelV1Prompt,
): QwenChatPrompt {
  const messages: QwenChatPrompt = []
  // Iterate over each prompt message.
  for (const { role, content, ...message } of prompt) {
    const metadata = getQwenMetadata({ ...message })
    switch (role) {
      case "system": {
        // System messages are sent directly with metadata.
        messages.push({ role: "system", content, ...metadata })
        break
      }

      case "user": {
        if (content.length === 1 && content[0].type === "text") {
          // For a single text element, simplify the conversion.
          messages.push({
            role: "user",
            content: content[0].text,
            ...getQwenMetadata(content[0]),
          })
          break
        }
        // For multiple content parts, process each part.
        messages.push({
          role: "user",
          content: content.map((part) => {
            const partMetadata = getQwenMetadata(part)
            switch (part.type) {
              case "text": {
                // Plain text conversion.
                return { type: "text", text: part.text, ...partMetadata }
              }
              case "image": {
                // Convert images and encode if necessary.
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
                // Unsupported file content parts trigger an error.
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
        // Build text response and accumulate function/tool calls.
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
              // Append each text part.
              text += part.text
              break
            }
            case "tool-call": {
              // Convert tool calls to function calls with serialized arguments.
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
        // Process tool responses by converting result to JSON string.
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
        // Ensure all roles are handled.
        const _exhaustiveCheck: never = role
        throw new Error(`Unsupported role: ${_exhaustiveCheck}`)
      }
    }
  }

  return messages
}
