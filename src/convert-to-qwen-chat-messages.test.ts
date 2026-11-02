import { describe, expect, it } from "vitest"
import { convertToQwenChatMessages } from "./convert-to-qwen-chat-messages"

vi.stubEnv("DASHSCOPE_API_KEY", "test-api-key-123")

describe("user messages", () => {
  it("should convert messages with only a text part to a string content", () => {
    const result = convertToQwenChatMessages([
      {
        role: "user",
        content: [
          { type: "text", text: "Hello" },
        ],
      },
    ])

    expect(result).toEqual([
      { role: "user", content: "Hello" },
    ])
  })

  it("should convert messages with image file parts", () => {
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

  it("should handle URL-based images", () => {
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

  it("should convert video file parts", () => {
    const result = convertToQwenChatMessages([
      {
        role: "user",
        content: [
          {
            type: "file",
            data: new URL("https://example.com/video.mp4"),
            mediaType: "video/mp4",
          },
        ],
      },
    ])

    expect(result).toEqual([
      {
        role: "user",
        content: [
          {
            type: "video_url",
            video_url: { url: "https://example.com/video.mp4" },
          },
        ],
      },
    ])
  })

  it("should convert audio file parts with the subtype as format", () => {
    const result = convertToQwenChatMessages([
      {
        role: "user",
        content: [
          {
            type: "file",
            data: new URL("https://example.com/audio.wav"),
            mediaType: "audio/wav",
          },
        ],
      },
    ])

    expect(result).toEqual([
      {
        role: "user",
        content: [
          {
            type: "input_audio",
            input_audio: {
              data: "https://example.com/audio.wav",
              format: "wav",
            },
          },
        ],
      },
    ])
  })

  it("should map mpeg audio format to mp3", () => {
    const result = convertToQwenChatMessages([
      {
        role: "user",
        content: [
          {
            type: "file",
            data: new URL("https://example.com/audio.mp3"),
            mediaType: "audio/mpeg",
          },
        ],
      },
    ])

    expect(result).toEqual([
      {
        role: "user",
        content: [
          {
            type: "input_audio",
            input_audio: {
              data: "https://example.com/audio.mp3",
              format: "mp3",
            },
          },
        ],
      },
    ])
  })

  it("should convert PDF file parts", () => {
    const result = convertToQwenChatMessages([
      {
        role: "user",
        content: [
          {
            type: "file",
            data: new URL("https://example.com/document.pdf"),
            mediaType: "application/pdf",
          },
        ],
      },
    ])

    expect(result).toEqual([
      {
        role: "user",
        content: [
          {
            type: "file",
            file: { file_url: "https://example.com/document.pdf" },
          },
        ],
      },
    ])
  })

  it("should throw for unsupported media types", () => {
    expect(() =>
      convertToQwenChatMessages([
        {
          role: "user",
          content: [
            {
              type: "file",
              data: new URL("https://example.com/file.zip"),
              mediaType: "application/zip",
            },
          ],
        },
      ]),
    ).toThrowError()
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
            output: { type: "text", value: "321rab" },
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
        content: "321rab",
        tool_call_id: "quux",
      },
    ])
  })

  it("should stringify non-text tool outputs", () => {
    const result = convertToQwenChatMessages([
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "quux",
            toolName: "thwomp",
            output: {
              type: "json",
              value: { url: "https://example.com" },
            },
          },
        ],
      },
    ])

    expect(result).toEqual([
      {
        role: "tool",
        tool_call_id: "quux",
        content: JSON.stringify({ url: "https://example.com" }),
      },
    ])
  })

  it("should use the value of error-text tool outputs", () => {
    const result = convertToQwenChatMessages([
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "quux",
            toolName: "thwomp",
            output: { type: "error-text", value: "Something broke" },
          },
        ],
      },
    ])

    expect(result).toEqual([
      {
        role: "tool",
        tool_call_id: "quux",
        content: "Something broke",
      },
    ])
  })
})

describe("reasoning", () => {
  it("should skip reasoning parts in assistant messages", () => {
    const result = convertToQwenChatMessages([
      {
        role: "assistant",
        content: [
          { type: "reasoning", text: "thinking..." },
          { type: "text", text: "Answer" },
        ],
      },
    ])

    expect(result).toEqual([
      {
        role: "assistant",
        content: "Answer",
        tool_calls: undefined,
      },
    ])
  })
})

describe("provider-specific options merging", () => {
  it("should merge system message options", () => {
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

  it("should merge user message content options", () => {
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

  it("should prioritize content-level options when merging", () => {
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

  it("should handle tool calls with options", () => {
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

  it("should handle file content with options", () => {
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
            image_url: {
              url: imageUrl.toString(),
            },
            cacheControl: { type: "ephemeral" },
          },
        ],
      },
    ])
  })

  it("should omit non-qwen options", () => {
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
        priority: "high", // hoisted from message-level providerOptions
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
          // this just gets omitted as we prioritize content-level options
          qwen: { responseTier: "detailed" },
        },
        content: [
          {
            type: "tool-result",
            toolCallId: "call123",
            toolName: "calculator",
            output: {
              type: "text",
              value: JSON.stringify({ stepOne: "data chunk 1" }),
            },
          },
          {
            type: "tool-result",
            toolCallId: "call123",
            toolName: "calculator",
            providerOptions: {
              qwen: { partial: true },
            },
            output: {
              type: "text",
              value: JSON.stringify({ stepTwo: "data chunk 2" }),
            },
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

  it("should handle multiple content parts with multiple option layers", () => {
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

  it("should handle different tool options vs. message-level options", () => {
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

  it("should handle options collisions and overwrites in tool calls", () => {
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

  it("should throw for unsupported assistant content parts", () => {
    expect(() =>
      convertToQwenChatMessages([
        {
          role: "assistant",
          content: [
            {
              type: "tool-result",
              toolCallId: "call1",
              toolName: "calculator",
              output: { type: "text", value: "1" },
            },
          ],
        },
      ]),
    ).toThrowError()
  })
})
