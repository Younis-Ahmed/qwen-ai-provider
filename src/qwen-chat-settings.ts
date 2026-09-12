// https://www.alibabacloud.com/help/en/model-studio/getting-started/models
export type QwenChatModelId
// Text Geeration
  = | "qwen2.5-14b-instruct-1m"
    | "qwen2.5-72b-instruct"
    | "qwen2.5-32b-instruct"
    | "qwen2.5-14b-instruct"
    | "qwen2.5-7b-instruct"
    | "qwen2-57b-a14b-instruct"
    | "qwen2.5-7b-instruct-1m"
    | "qwen-max"
    | "qwen-max-latest"
    | "qwen-max-2025-01-25"
    | "qwen-plus"
    | "qwen-plus-latest"
    | "qwen-plus-2025-01-25"
    | "qwen-turbo"
    | "qwen-turbo-latest"
    | "qwen-turbo-2024-11-01"
// Image/Video Understanding
    | "qwen-vl-max"
// Image Understanding
    | "qwen-vl-plus"
    | "qwen2.5-vl-72b-instruct"
    | "qwen2.5-vl-7b-instruct"
    | "qwen2.5-vl-3b-instruct"
// New multimodal models
    | "qwen3-vl-plus"
    | "qwen3-vl-flash"
    | "qwen3.8-max"
    | "qwen3.5-omni-plus"
    | "qwen3-omni-flash"
    | "qwen-audio-turbo"
    | (string & {})

export interface QwenChatSettings {
  /**
A unique identifier representing your end-user, which can help the provider to
monitor and detect abuse.
   */
  user?: string
  /**
Simulates streaming by using a normal generate call and returning it as a stream.
Enable this if the model that you are using does not support streaming.
Defaults to `false`.
   */
  simulateStreaming?: boolean
  /**
Whether the model supports structured outputs via JSON schema.
When true, the provider sends the full JSON schema in `response_format`
so the API can enforce it server-side.
Defaults to `true`.
   */
  supportsStructuredOutputs?: boolean
}
