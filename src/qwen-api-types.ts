import type { JSONValue } from "@ai-sdk/provider"

/**
 * Type representing a prompt for Qwen Chat built from an array of messages.
 */
export type QwenChatPrompt = Array<QwenMessage>

/**
 * Union type for all possible Qwen message types.
 */
export type QwenMessage =
  | QwenSystemMessage
  | QwenUserMessage
  | QwenAssistantMessage
  | QwenToolMessage

// Helper type for additional properties for metadata extensibility.
type JsonRecord<T = never> = Record<
  string,
  JSONValue | JSONValue[] | T | T[] | undefined
>

/**
 * System messages contain instructions/data set by the system.
 */
export interface QwenSystemMessage extends JsonRecord {
  role: "system"
  content: string
}

/**
 * User messages sent by the user to the model.
 */
export interface QwenUserMessage
  extends JsonRecord<QwenContentPart> {
  role: "user"
  content: string | Array<QwenContentPart>
}

/**
 * Represents a part of a message content.
 */
export type QwenContentPart =
  | QwenContentPartText
  | QwenContentPartImage

/**
 * Message part that contains an image URL.
 */
export interface QwenContentPartImage extends JsonRecord {
  type: "image_url"
  image_url: { url: string }
}

/**
 * Message part that contains text.
 */
export interface QwenContentPartText extends JsonRecord {
  type: "text"
  text: string
}

/**
 * Assistant messages response from the model.
 */
export interface QwenAssistantMessage
  extends JsonRecord<QwenMessageToolCall> {
  role: "assistant"
  content?: string | null
  tool_calls?: Array<QwenMessageToolCall>
}

/**
 * Represents a tool call embedded within an assistant message.
 */
export interface QwenMessageToolCall extends JsonRecord {
  type: "function"
  id: string
  function: {
    arguments: string
    name: string
  }
}

/**
 * Represents the response from a tool.
 */
export interface QwenToolMessage extends JsonRecord {
  role: "tool"
  content: string
  tool_call_id: string
}
