import { describe, expect, it, vi } from "vitest"
import { convertToQwenChatMessages } from "../convert-to-qwen-chat-messages"

vi.stubEnv("DASHSCOPE_API_KEY", "test-api-key-123")

describe("user messages", () => {
  it("should convert messages with only a text part to a string content", async () => {
    const result = convertToQwenChatMessages([
      {
        role: "user",
        content: [{ type: "text", text: "Hello" }],
      },
    ])

    expect(result).toEqual([{ role: "user", content: "Hello" }])
  })

  it("should convert messages with image parts", async () => {
    const result = convertToQwenChatMessages([
      {
        role: "user",
        content: [
          { type: "text", text: "Hello" },
          {
            type: "file",
            data: new Uint8Array([0, 1, 2, 3]),
            mediaType: "image/png",
          },
        ],
      },
    ])

    expect(result).toEqual([
      {
        role: "user",
        content: [
          { type: "text", text: "Hello" },
          {
            type: "image_url",
            image_url: { url: "data:image/png;base64,AAECAw==" },
          },
        ],
      },
    ])
  })

  it("should handle URL-based images", async () => {
    const result = convertToQwenChatMessages([
      {
        role: "user",
        content: [
          {
            type: "file",
            data: new URL("https://example.com/image.jpg"),
            mediaType: "image/jpeg",
          },
        ],
      },
    ])

    expect(result).toEqual([
      {
        role: "user",
        content: [
          {
            type: "image_url",
            image_url: { url: "https://example.com/image.jpg" },
          },
        ],
      },
    ])
  })

  it("should preserve pre-formed data URLs for image parts", async () => {
    const dataUrl = "data:image/png;base64,AAECAw=="
    const result = convertToQwenChatMessages([
      {
        role: "user",
        content: [
          {
            type: "file",
            data: dataUrl,
            mediaType: "image/png",
          },
        ],
      },
    ])

    expect(result).toEqual([
      {
        role: "user",
        content: [
          {
            type: "image_url",
            image_url: { url: dataUrl },
          },
        ],
      },
    ])
  })
})

describe("tool calls", () => {
  it("should stringify arguments to tool calls", () => {
    const result = convertToQwenChatMessages([
      {
        role: "assistant",
        content: [
          {
            type: "tool-call",
            input: { foo: "bar123" },
            toolCallId: "quux",
            toolName: "thwomp",
          },
        ],
      },
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "quux",
            toolName: "thwomp",
            output: { type: "json", value: { oof: "321rab" } },
          },
        ],
      },
    ])

    expect(result).toEqual([
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            type: "function",
            id: "quux",
            function: {
              name: "thwomp",
              arguments: JSON.stringify({ foo: "bar123" }),
            },
          },
        ],
      },
      {
        role: "tool",
        content: JSON.stringify({ oof: "321rab" }),
        tool_call_id: "quux",
      },
    ])
  })

  it("should throw for tool-result parts in assistant messages", () => {
    expect(() =>
      convertToQwenChatMessages([
        {
          role: "assistant",
          content: [
            {
              type: "tool-result",
              toolCallId: "abc",
              toolName: "calc",
              output: { type: "text", value: "ok" },
            } as any,
          ],
        },
      ]),
    ).toThrow()
  })

  it("should throw on unknown assistant content part types", () => {
    expect(() =>
      convertToQwenChatMessages([
        {
          role: "assistant",
          content: [
            // Force an unknown part type to hit the default branch
            { type: "unknown", foo: 1 } as any,
          ],
        },
      ]),
    ).toThrow()
  })

  it("should convert tool role text outputs without JSON stringifying", () => {
    const result = convertToQwenChatMessages([
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "t1",
            toolName: "echo",
            output: { type: "text", value: "plain-text" },
          },
        ],
      },
    ])

    expect(result).toEqual([
      {
        role: "tool",
        tool_call_id: "t1",
        content: "plain-text",
      },
    ])
  })

  it("should throw on unsupported role values", () => {
    expect(() =>
      convertToQwenChatMessages([
        {
          // Cast to any to bypass TS type checks and hit runtime default
          role: "bogus" as any,
          content: [{ type: "text", text: "Hi" }],
        },
      ]),
    ).toThrowError(/Unsupported role/)
  })
})

describe("provider-specific metadata merging", () => {
  it("should merge system message metadata", async () => {
    const result = convertToQwenChatMessages([
      {
        role: "system",
        content: "You are a helpful assistant.",
        providerOptions: {
          qwen: {
            cacheControl: { type: "ephemeral" },
          },
        },
      },
    ])

    expect(result).toEqual([
      {
        role: "system",
        content: "You are a helpful assistant.",
        cacheControl: { type: "ephemeral" },
      },
    ])
  })

  it("should merge user message content metadata", async () => {
    const result = convertToQwenChatMessages([
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Hello",
            providerOptions: {
              qwen: {
                cacheControl: { type: "ephemeral" },
              },
            },
          },
        ],
      },
    ])

    expect(result).toEqual([
      {
        role: "user",
        content: "Hello",
        cacheControl: { type: "ephemeral" },
      },
    ])
  })

  it("should prioritize content-level metadata when merging", async () => {
    const result = convertToQwenChatMessages([
      {
        role: "user",
        providerOptions: {
          qwen: {
            messageLevel: true,
          },
        },
        content: [
          {
            type: "text",
            text: "Hello",
            providerOptions: {
              qwen: {
                contentLevel: true,
              },
            },
          },
        ],
      },
    ])

    expect(result).toEqual([
      {
        role: "user",
        content: "Hello",
        contentLevel: true,
      },
    ])
  })

  it("should handle tool calls with metadata", async () => {
    const result = convertToQwenChatMessages([
      {
        role: "assistant",
        content: [
          {
            type: "tool-call",
            toolCallId: "call1",
            toolName: "calculator",
            input: { x: 1, y: 2 },
            providerOptions: {
              qwen: {
                cacheControl: { type: "ephemeral" },
              },
            },
          },
        ],
      },
    ])

    expect(result).toEqual([
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            id: "call1",
            type: "function",
            function: {
              name: "calculator",
              arguments: JSON.stringify({ x: 1, y: 2 }),
            },
            cacheControl: { type: "ephemeral" },
          },
        ],
      },
    ])
  })

  it("should handle image content with metadata", async () => {
    const imageUrl = new URL("https://example.com/image.jpg")
    const result = convertToQwenChatMessages([
      {
        role: "user",
        content: [
          {
            type: "file",
            data: imageUrl,
            mediaType: "image/jpeg",
            providerOptions: {
              qwen: {
                cacheControl: { type: "ephemeral" },
              },
            },
          },
        ],
      },
    ])

    expect(result).toEqual([
      {
        role: "user",
        content: [
          {
            type: "image_url",
            image_url: { url: imageUrl.toString() },
            cacheControl: { type: "ephemeral" },
          },
        ],
      },
    ])
  })

  it("should omit non-qwen metadata", async () => {
    const result = convertToQwenChatMessages([
      {
        role: "system",
        content: "Hello",
        providerOptions: {
          someOtherProvider: {
            shouldBeIgnored: true,
          },
        },
      },
    ])

    expect(result).toEqual([
      {
        role: "system",
        content: "Hello",
      },
    ])
  })

  it("should handle a user message with multiple content parts (text + image)", () => {
    const result = convertToQwenChatMessages([
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Hello from part 1",
            providerOptions: {
              qwen: { sentiment: "positive" },
              leftoverKey: { foo: "some leftover data" },
            },
          },
          {
            type: "file",
            data: new Uint8Array([0, 1, 2, 3]),
            mediaType: "image/png",
            providerOptions: {
              qwen: { alt_text: "A sample image" },
            },
          },
        ],
        providerOptions: {
          qwen: { priority: "high" },
        },
      },
    ])

    expect(result).toEqual([
      {
        role: "user",
        priority: "high", // hoisted from message-level providerMetadata
        content: [
          {
            type: "text",
            text: "Hello from part 1",
            sentiment: "positive", // hoisted from part-level qwen
          },
          {
            type: "image_url",
            image_url: {
              url: "data:image/png;base64,AAECAw==",
            },
            alt_text: "A sample image",
          },
        ],
      },
    ])
  })

  it("should handle a user message with multiple text parts (flattening disabled)", () => {
    const result = convertToQwenChatMessages([
      {
        role: "user",
        content: [
          { type: "text", text: "Part 1" },
          { type: "text", text: "Part 2" },
        ],
      },
    ])

    // Because there are multiple text parts, the converter won't flatten them
    expect(result).toEqual([
      {
        role: "user",
        content: [
          { type: "text", text: "Part 1" },
          { type: "text", text: "Part 2" },
        ],
      },
    ])
  })

  it("should handle an assistant message with text plus multiple tool calls", () => {
    const result = convertToQwenChatMessages([
      {
        role: "assistant",
        content: [
          { type: "text", text: "Checking that now..." },
          {
            type: "tool-call",
            toolCallId: "call1",
            toolName: "searchTool",
            input: { query: "Weather" },
            providerOptions: {
              qwen: { function_call_reason: "user request" },
            },
          },
          { type: "text", text: "Almost there..." },
          {
            type: "tool-call",
            toolCallId: "call2",
            toolName: "mapsTool",
            input: { location: "Paris" },
          },
        ],
      },
    ])

    expect(result).toEqual([
      {
        role: "assistant",
        content: "Checking that now...Almost there...",
        tool_calls: [
          {
            id: "call1",
            type: "function",
            function: {
              name: "searchTool",
              arguments: JSON.stringify({ query: "Weather" }),
            },
            function_call_reason: "user request",
          },
          {
            id: "call2",
            type: "function",
            function: {
              name: "mapsTool",
              arguments: JSON.stringify({ location: "Paris" }),
            },
          },
        ],
      },
    ])
  })

  it("should handle a single tool role message with multiple tool-result parts", () => {
    const result = convertToQwenChatMessages([
      {
        role: "tool",
        providerOptions: {
          // this just gets omitted as we prioritize content-level metadata
          qwen: { responseTier: "detailed" },
        },
        content: [
          {
            type: "tool-result",
            toolCallId: "call123",
            toolName: "calculator",
            output: { type: "json", value: { stepOne: "data chunk 1" } },
          },
          {
            type: "tool-result",
            toolCallId: "call123",
            toolName: "calculator",
            providerOptions: {
              qwen: { partial: true },
            },
            output: { type: "json", value: { stepTwo: "data chunk 2" } },
          },
        ],
      },
    ])

    expect(result).toEqual([
      {
        role: "tool",
        tool_call_id: "call123",
        content: JSON.stringify({ stepOne: "data chunk 1" }),
      },
      {
        role: "tool",
        tool_call_id: "call123",
        content: JSON.stringify({ stepTwo: "data chunk 2" }),
        partial: true,
      },
    ])
  })

  it("should handle multiple content parts with multiple metadata layers", () => {
    const result = convertToQwenChatMessages([
      {
        role: "user",
        providerOptions: {
          qwen: { messageLevel: "global-metadata" },
          leftoverForMessage: { x: 123 },
        },
        content: [
          {
            type: "text",
            text: "Part A",
            providerOptions: {
              qwen: { textPartLevel: "localized" },
              leftoverForText: { info: "text leftover" },
            },
          },
          {
            type: "file",
            data: new Uint8Array([9, 8, 7, 6]),
            mediaType: "image/png",
            providerOptions: {
              qwen: { imagePartLevel: "image-data" },
            },
          },
        ],
      },
    ])

    expect(result).toEqual([
      {
        role: "user",
        messageLevel: "global-metadata",
        content: [
          {
            type: "text",
            text: "Part A",
            textPartLevel: "localized",
          },
          {
            type: "image_url",
            image_url: {
              url: "data:image/png;base64,CQgHBg==",
            },
            imagePartLevel: "image-data",
          },
        ],
      },
    ])
  })

  it("should handle different tool metadata vs. message-level metadata", () => {
    const result = convertToQwenChatMessages([
      {
        role: "assistant",
        providerOptions: {
          qwen: { globalPriority: "high" },
        },
        content: [
          { type: "text", text: "Initiating tool calls..." },
          {
            type: "tool-call",
            toolCallId: "callXYZ",
            toolName: "awesomeTool",
            input: { param: "someValue" },
            providerOptions: {
              qwen: {
                toolPriority: "critical",
              },
            },
          },
        ],
      },
    ])

    expect(result).toEqual([
      {
        role: "assistant",
        globalPriority: "high",
        content: "Initiating tool calls...",
        tool_calls: [
          {
            id: "callXYZ",
            type: "function",
            function: {
              name: "awesomeTool",
              arguments: JSON.stringify({ param: "someValue" }),
            },
            toolPriority: "critical",
          },
        ],
      },
    ])
  })

  it("should handle metadata collisions and overwrites in tool calls", () => {
    const result = convertToQwenChatMessages([
      {
        role: "assistant",
        providerOptions: {
          qwen: {
            cacheControl: { type: "default" },
            sharedKey: "assistantLevel",
          },
        },
        content: [
          {
            type: "tool-call",
            toolCallId: "collisionToolCall",
            toolName: "collider",
            input: { num: 42 },
            providerOptions: {
              qwen: {
                cacheControl: { type: "ephemeral" }, // overwrites top-level
                sharedKey: "toolLevel",
              },
            },
          },
        ],
      },
    ])

    expect(result).toEqual([
      {
        role: "assistant",
        cacheControl: { type: "default" },
        sharedKey: "assistantLevel",
        content: "",
        tool_calls: [
          {
            id: "collisionToolCall",
            type: "function",
            function: {
              name: "collider",
              arguments: JSON.stringify({ num: 42 }),
            },
            cacheControl: { type: "ephemeral" },
            sharedKey: "toolLevel",
          },
        ],
      },
    ])
  })
})
