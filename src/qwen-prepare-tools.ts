import type {
  LanguageModelV1,
  LanguageModelV1CallWarning,
} from "@ai-sdk/provider"
import {
  UnsupportedFunctionalityError,
} from "@ai-sdk/provider"

/**
 * Prepares the tool configuration for language model generation.
 * @param param0 Object containing mode details and structured output flag.
 * @returns An object with tools, tool choice and any warnings.
 */
export function prepareTools({
  mode,
}: {
  mode: Parameters<LanguageModelV1["doGenerate"]>[0]["mode"] & {
    type: "regular"
  }
  structuredOutputs: boolean
}): {
  tools:
    | undefined
    | Array<{
      type: "function"
      function: {
        name: string
        description: string | undefined
        parameters: unknown
      }
    }>
  tool_choice:
    | { type: "function", function: { name: string } }
    | "auto"
    | "none"
    | "required"
    | undefined
  toolWarnings: LanguageModelV1CallWarning[]
} {
  // Normalize tools array by converting empty array to undefined.
  const tools = mode.tools?.length ? mode.tools : undefined
  const toolWarnings: LanguageModelV1CallWarning[] = []

  if (tools == null) {
    return { tools: undefined, tool_choice: undefined, toolWarnings }
  }

  const toolChoice = mode.toolChoice
  const qwenCompatTools: Array<{
    type: "function"
    function: {
      name: string
      description: string | undefined
      parameters: unknown
    }
  }> = []

  // Process each tool and format for compatibility.
  for (const tool of tools) {
    if (tool.type === "provider-defined") {
      // Warn if the tool is provider-defined.
      toolWarnings.push({ type: "unsupported-tool", tool })
    }
    else {
      qwenCompatTools.push({
        type: "function",
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters,
        },
      })
    }
  }

  if (toolChoice == null) {
    return { tools: qwenCompatTools, tool_choice: undefined, toolWarnings }
  }

  const type = toolChoice.type

  // Determine tool choice strategy.
  switch (type) {
    case "auto":
    case "none":
    case "required":
      return { tools: qwenCompatTools, tool_choice: type, toolWarnings }
    case "tool":
      return {
        tools: qwenCompatTools,
        tool_choice: {
          type: "function",
          function: {
            name: toolChoice.toolName,
          },
        },
        toolWarnings,
      }
    default: {
      // Exhaustive check to ensure all cases are handled.
      const _exhaustiveCheck: never = type
      throw new UnsupportedFunctionalityError({
        functionality: `Unsupported tool choice type: ${_exhaustiveCheck}`,
      })
    }
  }
}
