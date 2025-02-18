/* eslint-disable array-callback-return */
import type {
  LanguageModelV1Prompt,
} from "@ai-sdk/provider"
import {
  InvalidPromptError,
  UnsupportedFunctionalityError,
} from "@ai-sdk/provider"

/**
 * Converts a LanguageModelV1Prompt into a Qwen completion prompt.
 *
 * @param options - The configuration options
 * @param options.prompt - The input prompt in LanguageModelV1Prompt format
 * @param options.inputFormat - Either "prompt" (raw text input) or "messages" (chat messages)
 * @param options.user - Label for user messages (default: "user")
 * @param options.assistant - Label for assistant messages (default: "assistant")
 * @returns An object containing:
 *          - prompt: The constructed prompt string
 *          - stopSequences?: Array of strings to use as stop sequences
 * @throws {InvalidPromptError} When an unexpected system message is encountered
 * @throws {UnsupportedFunctionalityError} For unsupported content types such as images or tool-calls
 */
export function convertToQwenCompletionPrompt({
  prompt,
  inputFormat,
  user = "user",
  assistant = "assistant",
}: {
  prompt: LanguageModelV1Prompt
  inputFormat: "prompt" | "messages"
  user?: string
  assistant?: string
}): {
    prompt: string
    stopSequences?: string[]
  } {
  // If input is a straightforward prompt with one user message, return it directly.
  if (
    inputFormat === "prompt"
    && prompt.length === 1
    && prompt[0].role === "user"
    && prompt[0].content.length === 1
    && prompt[0].content[0].type === "text"
  ) {
    // Return the original text without transformation.
    return { prompt: prompt[0].content[0].text }
  }

  // Start assembling the text for a chat message format.
  let text = ""

  // If the first message is a system message, add its content first.
  if (prompt[0].role === "system") {
    text += `${prompt[0].content}\n\n`
    prompt = prompt.slice(1) // Remove the system message after processing.
  }

  // Process each message in the prompt.
  for (const { role, content } of prompt) {
    switch (role) {
      case "system": {
        // System messages are not expected beyond the first message.
        throw new InvalidPromptError({
          message: `Unexpected system message in prompt: ${content}`,
          prompt,
        })
      }

      case "user": {
        // Concatenate the parts of user messages.
        const userMessage = content
          .map((part) => {
            switch (part.type) {
              case "text": {
                return part.text
              }
              case "image": {
                // Images are not supported.
                throw new UnsupportedFunctionalityError({
                  functionality: "images",
                })
              }
              default: {
                throw new Error(`Unsupported content part type: ${part.type}`)
              }
            }
          })
          .join("")
        // Append user label and the message.
        text += `${user}:\n${userMessage}\n\n`
        break
      }

      case "assistant": {
        // Process assistant messages similarly.
        const assistantMessage = content
          .map((part) => {
            switch (part.type) {
              case "text": {
                return part.text
              }
              case "tool-call": {
                // Tool-call messages are unsupported.
                throw new UnsupportedFunctionalityError({
                  functionality: "tool-call messages",
                })
              }
            }
          })
          .join("")
        // Append assistant label and the message.
        text += `${assistant}:\n${assistantMessage}\n\n`
        break
      }

      case "tool": {
        // Tool messages are not supported.
        throw new UnsupportedFunctionalityError({
          functionality: "tool messages",
        })
      }

      default: {
        // Ensure exhaustive check.
        const _exhaustiveCheck: never = role
        throw new Error(`Unsupported role: ${_exhaustiveCheck}`)
      }
    }
  }

  // Append the final assistant signal to the chat completion.
  text += `${assistant}:\n`

  // Return the constructed prompt along with a stop sequence to separate future user inputs.
  return {
    prompt: text,
    stopSequences: [`\n${user}:`],
  }
}
