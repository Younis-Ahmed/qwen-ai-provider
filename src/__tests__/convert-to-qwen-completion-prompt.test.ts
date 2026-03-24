import { InvalidPromptError, UnsupportedFunctionalityError } from "@ai-sdk/provider"
import { describe, expect, it } from "vitest"
import { convertToQwenCompletionPrompt } from "../convert-to-qwen-completion-prompt"

describe("convertToQwenCompletionPrompt", () => {
  it("returns raw text when inputFormat is 'prompt' with single user text", () => {
    const { prompt } = convertToQwenCompletionPrompt({
      inputFormat: "prompt",
      prompt: [
        { role: "user", content: [{ type: "text", text: "Hello world" }] },
      ],
    })
    expect(prompt).toBe("Hello world")
  })

  it("constructs chat prompt with initial system, multiple user/assistant parts, and default labels", () => {
    const { prompt, stopSequences } = convertToQwenCompletionPrompt({
      inputFormat: "messages",
      prompt: [
        { role: "system", content: "Be helpful" },
        {
          role: "user",
          content: [
            { type: "text", text: "Hi" },
            { type: "text", text: ", there" },
          ],
        },
        { role: "assistant", content: [{ type: "text", text: "Hello!" }] },
      ],
    })

    expect(prompt).toBe("Be helpful\n\nuser:\nHi, there\n\nassistant:\nHello!\n\nassistant:\n")
    expect(stopSequences).toEqual(["\nuser:"])
  })

  it("supports system content provided as parts array (text-only)", () => {
    const { prompt } = convertToQwenCompletionPrompt({
      inputFormat: "messages",
      // Cast to any to simulate a system message with parts array
      prompt: [
        { role: "system", content: [
          { type: "text", text: "Be " },
          { type: "text", text: "helpful" },
        ] } as any,
        { role: "user", content: [{ type: "text", text: "Hi" }] },
      ],
    })

    expect(prompt).toBe("Be helpful\n\nuser:\nHi\n\nassistant:\n")
  })

  it("uses custom user/assistant labels and updates stopSequences accordingly", () => {
    const { prompt, stopSequences } = convertToQwenCompletionPrompt({
      inputFormat: "messages",
      user: "Human",
      assistant: "AI",
      prompt: [
        { role: "user", content: [{ type: "text", text: "Ping" }] },
        { role: "assistant", content: [{ type: "text", text: "Pong" }] },
      ],
    })

    expect(prompt).toBe("Human:\nPing\n\nAI:\nPong\n\nAI:\n")
    expect(stopSequences).toEqual(["\nHuman:"])
  })

  it("throws on unexpected system message after the first position", () => {
    expect(() =>
      convertToQwenCompletionPrompt({
        inputFormat: "messages",
        prompt: [
          { role: "user", content: [{ type: "text", text: "Hi" }] },
          { role: "system", content: "Too late" },
        ],
      }),
    ).toThrow(InvalidPromptError)
  })

  it("throws on user file content part", () => {
    expect(() =>
      convertToQwenCompletionPrompt({
        inputFormat: "messages",
        prompt: [
          {
            role: "user",
            content: [
              {
                type: "file",
                mediaType: "image/png",
                data: new Uint8Array([1, 2, 3]),
              },
            ],
          },
        ],
      }),
    ).toThrow(UnsupportedFunctionalityError)
  })

  it("throws on assistant tool-call", () => {
    expect(() =>
      convertToQwenCompletionPrompt({
        inputFormat: "messages",
        prompt: [
          { role: "user", content: [{ type: "text", text: "Hi" }] },
          {
            role: "assistant",
            content: [
              {
                type: "tool-call",
                toolCallId: "id",
                toolName: "t",
                input: {},
              },
            ],
          },
        ],
      }),
    ).toThrow(UnsupportedFunctionalityError)
  })

  it("throws on assistant file/reasoning/tool-result content parts", () => {
    for (const part of [
      { type: "file", mediaType: "image/png", data: new Uint8Array([1]) },
      { type: "reasoning", text: "think" },
      { type: "tool-result", toolCallId: "x", toolName: "y", output: { type: "text", value: "z" } },
    ] as const) {
      expect(() =>
        convertToQwenCompletionPrompt({
          inputFormat: "messages",
          prompt: [
            { role: "user", content: [{ type: "text", text: "Hi" }] },
            { role: "assistant", content: [part as any] },
          ],
        }),
      ).toThrow(UnsupportedFunctionalityError)
    }
  })
})
