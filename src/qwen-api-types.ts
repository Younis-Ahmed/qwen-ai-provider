import type { JSONValue } from "@ai-sdk/provider"

export type QwenChatPrompt = Array<QwenMessage>

export type QwenMessage =
  | QwenSystemMessage
  | QwenUserMessage
  | QwenAssistantMessage
  | QwenToolMessage

// Allow for arbitrary additional properties for general purpose
// provider-metadata-specific extensibility.
type JsonRecord<T = never> = Record<
  string,
  JSONValue | JSONValue[] | T | T[] | undefined
>

export interface QwenSystemMessage extends JsonRecord {
  role: "system"
  content: string
}

export interface QwenUserMessage
  extends JsonRecord<QwenContentPart> {
  role: "user"
  content: string | Array<QwenContentPart>
}

export type QwenContentPart =
  | QwenContentPartText
  | QwenContentPartImage

export interface QwenContentPartImage extends JsonRecord {
  type: "image_url"
  image_url: { url: string }
}

export interface QwenContentPartText extends JsonRecord {
  type: "text"
  text: string
}

export interface QwenAssistantMessage
  extends JsonRecord<QwenMessageToolCall> {
  role: "assistant"
  content?: string | null
  tool_calls?: Array<QwenMessageToolCall>
}

export interface QwenMessageToolCall extends JsonRecord {
  type: "function"
  id: string
  function: {
    arguments: string
    name: string
  }
}

export interface QwenToolMessage extends JsonRecord {
  role: "tool"
  content: string
  tool_call_id: string
}
