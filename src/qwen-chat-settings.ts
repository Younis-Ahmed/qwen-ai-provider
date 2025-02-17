import type { OpenAICompatibleChatSettings } from "@ai-sdk/openai-compatible";

// https://www.alibabacloud.com/help/en/model-studio/getting-started/models
export type QwenChatModelId =
  // Text Geeration
  | "qwen2.5-14b-instruct-1m"
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
  | (string & {});

export interface QwenChatSettings extends OpenAICompatibleChatSettings {}
