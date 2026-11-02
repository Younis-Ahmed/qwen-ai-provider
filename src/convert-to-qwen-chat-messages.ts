import type { LanguageModelV2Prompt } from "@ai-sdk/provider"
import type { QwenChatPrompt, QwenContentPart } from "./qwen-api-types"
import { UnsupportedFunctionalityError } from "@ai-sdk/provider"
import { convertUint8ArrayToBase64 } from "@ai-sdk/provider-utils"

/**
 * Convert message provider options to Qwen provider options.
 */
function getQwenProviderOptions(message: unknown): Record<string, unknown> {
  return (
    message as {
      providerOptions?: { qwen?: Record<string, unknown> } | undefined
    } | undefined
  )?.providerOptions?.qwen ?? {}
}

/**
 * Converts a prompt to a Qwen chat prompt.
 *
 * @param prompt - The prompt to convert.
 * @returns The converted Qwen chat prompt.
 */
export function convertToQwenChatMessages(
  prompt: LanguageModelV2Prompt,
): QwenChatPrompt {
  const messages: QwenChatPrompt = []

  for (const { role, content, ...message } of prompt) {
    const providerOptions = getQwenProviderOptions({ ...message })

    switch (role) {
      case "system": {
        messages.push({
          role: "system",
          content,
          ...providerOptions,
        })
        break
      }

      case "user": {
        if (content.length === 1 && content[0].type === "text") {
          messages.push({
            role: "user",
            content: content[0].text,
            ...getQwenProviderOptions(content[0]),
          })
          break
        }

        messages.push({
          role: "user",
          content: content.map(
            (part): QwenContentPart => {
              const partProviderOptions = getQwenProviderOptions(part)

              switch (part.type) {
                case "text": {
                  return {
                    type: "text",
                    text: part.text,
                    ...partProviderOptions,
                  }
                }

                case "file": {
                  const mediaType: string = part.mediaType ?? "image/jpeg"
                  const url = part.data instanceof URL
                    ? part.data.toString()
                    : typeof part.data === "string"
                      ? part.data
                      : `data:${mediaType};base64,${convertUint8ArrayToBase64(part.data)}`

                  if (mediaType.startsWith("image/")) {
                    return {
                      type: "image_url",
                      image_url: { url },
                      ...partProviderOptions,
                    }
                  }

                  if (mediaType.startsWith("video/")) {
                    return {
                      type: "video_url",
                      video_url: { url },
                      ...partProviderOptions,
                    }
                  }

                  if (mediaType.startsWith("audio/")) {
                    const format = mediaType.split("/")[1]?.replace("mpeg", "mp3") ?? "wav"

                    return {
                      type: "input_audio",
                      input_audio: { data: url, format },
                      ...partProviderOptions,
                    }
                  }

                  if (mediaType === "application/pdf") {
                    return {
                      type: "file",
                      file: { file_url: url },
                      ...partProviderOptions,
                    }
                  }

                  throw new UnsupportedFunctionalityError({
                    functionality: `file content part with media type "${mediaType}"`,
                  })
                }

                default: {
                  throw new Error("Unsupported part type")
                }
              }
            },
          ),
          ...providerOptions,
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
          const partProviderOptions = getQwenProviderOptions(part)

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
                  arguments: JSON.stringify(part.input),
                },
                ...partProviderOptions,
              })
              break
            }
            case "reasoning": {
              break
            }
            case "file":
            case "tool-result": {
              throw new UnsupportedFunctionalityError({
                functionality: `${part.type} content parts in assistant messages`,
              })
            }
          }
        }

        messages.push({
          role: "assistant",
          content: text,
          tool_calls: toolCalls.length > 0
            ? toolCalls
            : undefined,
          ...providerOptions,
        })
        break
      }

      case "tool": {
        for (const toolResponse of content) {
          const toolResponseProviderOptions = getQwenProviderOptions(toolResponse)
          const output = toolResponse.output

          messages.push({
            role: "tool",
            tool_call_id: toolResponse.toolCallId,
            content: output.type === "text" || output.type === "error-text"
              ? output.value
              : JSON.stringify(output.value),
            ...toolResponseProviderOptions,
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
