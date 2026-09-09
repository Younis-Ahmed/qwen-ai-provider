import type { LanguageModelV2Prompt } from "@ai-sdk/provider"
import { InvalidPromptError, UnsupportedFunctionalityError } from "@ai-sdk/provider"

/**
 * Convert V2 prompt to a Qwen completion prompt.
 */
export function convertToQwenCompletionPrompt({
  prompt,
  user = "user",
  assistant = "assistant",
}: {
  prompt: LanguageModelV2Prompt
  user?: string
  assistant?: string
}): { prompt: string, stopSequences: string[] } {
  let text = ""

  // First message is the system message
  if (prompt[0].role === "system") {
    text += `${prompt[0].content}

`
    prompt = prompt.slice(1)
  }

  for (const { role, content } of prompt) {
    switch (role) {
      case "system": {
        throw new InvalidPromptError({
          message: `Unexpected system message in prompt: ${content}`,
          prompt,
        })
      }
      case "user": {
        const userMessage = content
          .map((part) => {
            switch (part.type) {
              case "text": {
                return part.text
              }
              case "file": {
                throw new UnsupportedFunctionalityError({
                  functionality: "images",
                })
              }
              default: {
                throw new Error("Unsupported part type")
              }
            }
          })
          .filter(Boolean)
          .join("")

        text += `${user}:
${userMessage}

`
        break
      }
      case "assistant": {
        const assistantMessage = content
          .map((part) => {
            switch (part.type) {
              case "text": {
                return part.text
              }
              case "tool-call":
              case "tool-result":
              case "reasoning":
              case "file": {
                throw new UnsupportedFunctionalityError({
                  functionality: `${part.type} messages`,
                })
              }
              default: {
                throw new Error("Unsupported part type")
              }
            }
          })
          .join("")

        text += `${assistant}:
${assistantMessage}

`
        break
      }
      case "tool": {
        throw new UnsupportedFunctionalityError({
          functionality: "tool messages",
        })
      }
      default: {
        const _exhaustiveCheck: never = role
        throw new Error(`Unsupported role: ${_exhaustiveCheck}`)
      }
    }
  }

  text += `${assistant}:
`

  return {
    prompt: text,
    stopSequences: [`
${user}:`],
  }
}
