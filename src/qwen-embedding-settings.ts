export type QwenEmbeddingModelId = "text-embedding-v3" | (string & {})

export interface QwenEmbeddingSettings {
  // settings from https://www.alibabacloud.com/help/en/model-studio/getting-started/models

  /**
A unique identifier representing your end-user, which can help vercel-sdk to
monitor and detect abuse. Learn more.
   */
  user?: string
  /**
   * The type of the text. Valid values: query and document. Default value: document.
   * If you want to perform queries on vectorized texts, we recommend that you specify this parameter to distinguish between the query and document types. If you only want to perform clustering and classification on them, you can use the default value.
   */
  text_type?: string

  /**
   * The vector dimension. Valid values: 1024, 768, and 512.
  Default value: 1024.
   */

  dimensions?: number

  /**
   * The type of output. Valid values: dense, sparse, dense&sparse.
   * Default value: dense, which specifies only dense vectors are returned.
   */
  output_type?: "dense" | "sparse" | "dense&sparse"
}
