# qwen-ai-provider

Qwen AI Provider for interacting with Qwen models via Alibaba Cloud's Model Studio API.

## Requirements

This provider requires a valid Qwen API key and adherence to the API specifications provided by Alibaba Cloud.

## Installation

The qwen-ai-provider is available as an npm package. You can install it with:

```bash
npm i qwen-ai-provider
```

## Provider Instance

You can import the default provider instance `qwen` from `qwen-ai-provider`:

```ts
import { qwen } from "qwen-ai-provider"
```

If you need a customized setup, you can import `createQwen` from `qwen-ai-provider` and create a provider instance with your settings:

```ts
import { createQwen } from "qwen-ai-provider"

const qwenProvider = createQwen({
  // custom settings, e.g.:
  apiKey: "YOUR_QWEN_API_KEY",
  baseURL: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
  headers: { "Custom-Header": "value" },
})
```

The following optional settings can be provided:

- **baseURL** _string_

  Use a different URL prefix for API calls, e.g. to use proxy servers.
  The default is `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`.

- **apiKey** _string_

  Qwen API key, defaults to the `DASHSCOPE_API_KEY` environment variable.

- **headers** _Record<string,string>_

  Custom headers to include in requests.

- **queryParams** _Record<string,string>_

  Optional URL query parameters for API calls.

- **fetch** _FetchFunction_

  Custom fetch implementation, useful for testing or middleware.

## Models

The provider supports text generation/chat and embeddings.
The first argument is the model id. For example:

```ts
const chatModel = qwen("qwen-plus")
```

Other model functions include:

- `qwen.chatModel(modelId, settings)`
- `qwen.completion(modelId, settings)`
- `qwen.textEmbeddingModel(modelId, settings)`
- `qwen.languageModel(modelId, settings)` (alias for `chatModel`)

## Tested Models and Capabilities

This provider has been tested with Qwen models and supports a range of features:

| Feature              | Support                           |
| -------------------- | --------------------------------- |
| Text generation      | :white_check_mark:                |
| Streaming output     | :white_check_mark: (if supported) |
| Object generation    | :white_check_mark:                |
| Embedding generation | :white_check_mark:                |

## Usage Details

- **Chat Settings**

  The settings object allows you to fine-tune model behavior, including streaming, temperature control, and tool integration. See inline documentation in the source code for more details.

- **Tool Integration**

  Some Qwen models support tool usage for function calls. Configure the `tools` property in your settings as needed.

- **Versioning and Packaging**

  When publishing packages (e.g. on GitHub Packages), the provider supports pre-release versioning and packaging via GitHub Actions.

## Testing

This package includes a suite of tests using Vitest. To run tests:

```bash
npm run test
```

For more details, review the test files in the repository.
