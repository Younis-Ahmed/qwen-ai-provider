/**
 * Supported embedding model IDs.
 */
export type QwenEmbeddingModelId = "text-embedding-v3" | (string & {})

/**
 * Settings configuration for Qwen text embeddings.
 */
export interface QwenEmbeddingSettings {
	/**
	 * A unique identifier for the end-user used for monitoring and abuse detection.
	 */
	user?: string
	/**
	 * The type of text. Valid values are 'query' and 'document'.
	 * Default is 'document'. Use 'query' when performing text queries.
	 */
	text_type?: string

	/**
	 * The dimensionality of the output vector.
	 * Valid values include 1024, 768, and 512. Default is 1024.
	 */
	dimensions?: number

	/**
	 * Specifies the type of output vectors.
	 * Valid values: "dense", "sparse", or "dense&sparse". Default is "dense".
	 */
	output_type?: "dense" | "sparse" | "dense&sparse"
}
