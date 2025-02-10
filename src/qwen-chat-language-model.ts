import {
  LanguageModelV1,
  LanguageModelV1CallWarning,
  LanguageModelV1FinishReason,
  LanguageModelV1StreamPart,
} from '@ai-sdk/provider'
import {
  combineHeaders,
  createEventSourceResponseHandler,
  createJsonResponseHandler,
  FetchFunction,
  ParseResult,
  postJsonToApi,
} from '@ai-sdk/provider-utils'
import { z } from 'zod'
