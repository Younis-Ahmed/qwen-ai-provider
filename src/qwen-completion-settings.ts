/**
 * Module defining types and interfaces for Qwen completion model settings.
 *
 * @module qwen-completion-settings
 */
import type { QwenChatModelId, QwenChatSettings } from "./qwen-chat-settings"

/**
 * Alias for Qwen Chat Model ID used for completions.
 */
export type QwenCompletionModelId = QwenChatModelId

/**
 * Settings for Qwen completions, extending the base chat settings.
 *
 * @remarks
 * These settings control characteristics such as prompt echoing, token bias,
 * output suffix, and user identification.
 */
export interface QwenCompletionSettings extends QwenChatSettings {
  /**
   * Echo the input prompt along with the completion.
   *
   * @example
   * { echo: true }
   */
  echo?: boolean

  /**
   * Adjust the likelihood of specified tokens appearing in the completion.
   *
   * @remarks
   * Accepts an object mapping token IDs (from the qwen-plus tokenizer) to bias values.
   * The bias (ranging from -100 to 100) alters the model's logits prior to sampling.
   * Values near -100 or 100 can effectively ban or force the selection of tokens.
   *
   * @example
   * { logitBias: { "50256": -100 } }
   */
  logitBias?: Record<number, number>

  /**
   * Suffix for the generated completion.
   *
   * @remarks
   * This string is appended to the generated text after completion.
   *
   * @example
   * { suffix: " -- End" }
   */
  suffix?: string

  /**
   * A unique end-user identifier.
   *
   * @remarks
   * This identifier can assist in monitoring usage and detecting abuse.
   *
   * @example
   * { user: "user-1234" }
   */
  user?: string
}
