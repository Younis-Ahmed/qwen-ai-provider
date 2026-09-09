import type {
  LanguageModelV2CallWarning,
  LanguageModelV2FunctionTool,
  LanguageModelV2ProviderDefinedTool,
  LanguageModelV2ToolChoice,
} from "@ai-sdk/provider"
import {
  UnsupportedFunctionalityError,
} from "@ai-sdk/provider"

/**
 * Prepares the tool configuration for language model generation.
 * @param param0 Object containing tools and tool choice.
 * @param param0.tools The tools available for language model generation.
 * @param param0.toolChoice Specifies how the tool should be selected.
 * @returns An object with tools, tool_choice and any warnings.
 */
export function prepareTools({
  tools,
  toolChoice,
}: {
  tools?: Array<LanguageModelV2FunctionTool | LanguageModelV2ProviderDefinedTool>
  toolChoice?: LanguageModelV2ToolChoice
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
  toolWarnings: LanguageModelV2CallWarning[]
} {
  // Normalize tools array by converting empty array to undefined.
  const normalizedTools = tools?.length ? tools : undefined
  const toolWarnings: LanguageModelV2CallWarning[] = []

  if (normalizedTools == null) {
    return { tools: undefined, tool_choice: undefined, toolWarnings }
  }

  const qwenCompatTools: Array<{
    type: "function"
    function: {
      name: string
      description: string | undefined
      parameters: unknown
    }
  }> = []

  // Process each tool and format for compatibility.
  for (const tool of normalizedTools) {
    if (tool.type === "function") {
      qwenCompatTools.push({
        type: "function",
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.inputSchema,
        },
      })
    }
    else {
      // Warn if the tool is provider-defined.
      toolWarnings.push({ type: "unsupported-tool", tool })
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
