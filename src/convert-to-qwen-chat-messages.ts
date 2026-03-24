import type {
  LanguageModelV2Prompt,
  SharedV2ProviderOptions,
} from "@ai-sdk/provider"
import type { QwenChatPrompt } from "./qwen-api-types"
import { UnsupportedFunctionalityError } from "@ai-sdk/provider"
import { convertUint8ArrayToBase64 } from "@ai-sdk/provider-utils"

// JSDoc for helper function to extract Qwen options.
/**
 * Extracts Qwen-specific options from a message.
 *
 * @param message - An object that may contain providerOptions
 * @param message.providerOptions - Provider-specific options containing Qwen configuration
 * @returns The Qwen options object or an empty object if none exists
 */

function getQwenOptions(message: {
  providerOptions?: SharedV2ProviderOptions
}) {
  return message?.providerOptions?.qwen ?? {}
}

/**
 * Converts a generic language model prompt to Qwen chat messages.
 *
 * @param prompt The language model prompt to convert.
 * @returns An array of Qwen chat messages.
 */
export function convertToQwenChatMessages(
  prompt: LanguageModelV2Prompt,
): QwenChatPrompt {
  const messages: QwenChatPrompt = []
  // Iterate over each prompt message.
  for (const { role, content, ...message } of prompt) {
    const options = getQwenOptions({ ...message })
    switch (role) {
      case "system": {
        // System messages are sent directly with options.
        messages.push({ role: "system", content, ...options })
        break
      }

      case "user": {
        if (content.length === 1 && content[0].type === "text") {
          // For a single text element, simplify the conversion.
          messages.push({
            role: "user",
            content: content[0].text,
            ...getQwenOptions(content[0]),
          })
          break
        }
        // For multiple content parts, process each part.
        messages.push({
          role: "user",
          content: content.map((part) => {
            const partOptions = getQwenOptions(part)
            switch (part.type) {
              case "text": {
                // Plain text conversion.
                return { type: "text", text: part.text, ...partOptions }
              }
              case "file": {
                // Check if this is an image file
                if (part.mediaType && part.mediaType.startsWith("image/")) {
                  // Convert images and encode if necessary.
                  const data = part.data
                  let url: string
                  if (typeof data === "string") {
                    // Preserve full URLs and pre-formed data URLs; otherwise, treat as raw base64
                    if (data.startsWith("http") || data.startsWith("data:")) {
                      url = data
                    }
                    else {
                      url = `data:${part.mediaType};base64,${data}`
                    }
                  }
                  else if (data instanceof URL) {
                    url = data.toString()
                  }
                  else {
                    // Uint8Array
                    url = `data:${part.mediaType};base64,${convertUint8ArrayToBase64(data)}`
                  }
                  return {
                    type: "image_url",
                    image_url: { url },
                    ...partOptions,
                  }
                }
                // Non-image files are unsupported
                throw new UnsupportedFunctionalityError({
                  functionality:
                    "Non-image file content parts in user messages",
                })
              }
              default: {
                // Unsupported content parts trigger an error.
                const _exhaustiveCheck: never = part
                throw new UnsupportedFunctionalityError({
                  functionality: `Unsupported content part type: ${_exhaustiveCheck}`,
                })
              }
            }
          }),
          ...options,
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
          const partOptions = getQwenOptions(part)
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
                  arguments: JSON.stringify(part.input),
                },
                ...partOptions,
              })
              break
            }
            case "file": // Add cases in v5
            case "reasoning": {
              // Ignore or handle these part types as needed
              throw new UnsupportedFunctionalityError({
                functionality: `${part.type} content parts in assistant messages`,
              })
            }
            case "tool-result": {
              // Tool results should not appear in assistant messages
              throw new UnsupportedFunctionalityError({
                functionality:
                  "tool-result content parts in assistant messages",
              })
            }
            default: {
              // This branch should never occur.
              const _exhaustiveCheck: never = part
              throw new Error(`Unsupported part: ${_exhaustiveCheck}`)
            }
          }
        }

        messages.push({
          role: "assistant",
          content: text,
          tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
          ...options,
        })

        break
      }

      case "tool": {
        // Process tool responses by converting output to JSON string.
        for (const toolResponse of content) {
          const toolResponseOptions = getQwenOptions(toolResponse)
          const output = toolResponse.output

          // Extract the value from the V2 output format
          let toolContent: string
          if (output.type === "text" || output.type === "error-text") {
            toolContent = output.value
          }
          else {
            // json or error-json
            toolContent = JSON.stringify(output.value)
          }

          messages.push({
            role: "tool",
            tool_call_id: toolResponse.toolCallId,
            content: toolContent,
            ...toolResponseOptions,
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
