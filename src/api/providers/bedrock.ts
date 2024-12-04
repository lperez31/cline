import AnthropicBedrock from "@anthropic-ai/bedrock-sdk"
import { Anthropic } from "@anthropic-ai/sdk"
import { BedrockRuntimeClient, InvokeModelWithResponseStreamCommand } from "@aws-sdk/client-bedrock-runtime"
import { ApiHandler } from "../"
import { ApiHandlerOptions, bedrockDefaultModelId, BedrockModelId, bedrockModels, ModelInfo } from "../../shared/api"
import { ApiStream } from "../transform/stream"

// https://docs.anthropic.com/en/api/claude-on-amazon-bedrock
export class AwsBedrockHandler implements ApiHandler {
	private options: ApiHandlerOptions
	private anthropicClient: AnthropicBedrock
	private bedrockClient: BedrockRuntimeClient

	constructor(options: ApiHandlerOptions) {
		this.options = options
		
		const awsConfig = {
			region: this.options.awsRegion || "us-east-1",
			...(this.options.awsAccessKey && this.options.awsSecretKey ? {
				credentials: {
					accessKeyId: this.options.awsAccessKey,
					secretAccessKey: this.options.awsSecretKey,
					...(this.options.awsSessionToken ? { sessionToken: this.options.awsSessionToken } : {})
				}
			} : {})
		}

		this.anthropicClient = new AnthropicBedrock({
			// Authenticate by either providing the keys below or use the default AWS credential providers, such as
			// using ~/.aws/credentials or the "AWS_SECRET_ACCESS_KEY" and "AWS_ACCESS_KEY_ID" environment variables.
			...(this.options.awsAccessKey ? { awsAccessKey: this.options.awsAccessKey } : {}),
			...(this.options.awsSecretKey ? { awsSecretKey: this.options.awsSecretKey } : {}),
			...(this.options.awsSessionToken ? { awsSessionToken: this.options.awsSessionToken } : {}),

			// awsRegion changes the aws region to which the request is made. By default, we read AWS_REGION,
			// and if that's not present, we default to us-east-1. Note that we do not read ~/.aws/config for the region.
			awsRegion: this.options.awsRegion,
		})

		this.bedrockClient = new BedrockRuntimeClient(awsConfig)
	}

	async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
		const model = this.getModel()
		const modelId = this.getModelId(model.id)

		// Handle Amazon Nova models differently
		if (modelId.includes("amazon.nova")) {
			const command = new InvokeModelWithResponseStreamCommand({
				modelId,
				contentType: "application/json",
				accept: "application/json",
				body: JSON.stringify({
					messages: [
						// Add system prompt as first user message
						{
							role: "user",
							content: [{ text: systemPrompt }]
						},
						// Map the rest of the messages
						...messages.map(msg => ({
							role: msg.role === "assistant" ? "assistant" : "user",
							content: [{ 
								text: typeof msg.content === "string" 
									? msg.content 
									: Array.isArray(msg.content) && msg.content.length > 0
										? (msg.content[0].type === "text" 
											? msg.content[0].text 
											: JSON.stringify(msg.content[0]))
										: "" 
							}]
						}))
					]
				})
			})

			const response = await this.bedrockClient.send(command)
			if (!response.body) {
				throw new Error("No response body received from Bedrock")
			}

			for await (const chunk of response.body) {
				if (chunk.chunk?.bytes) {
					try {
						const chunkText = new TextDecoder().decode(chunk.chunk.bytes)
						console.log('Debug - Received chunk:', chunkText)
						
						const response = JSON.parse(chunkText)
						console.log('Debug - Parsed response:', JSON.stringify(response, null, 2))

						if (response.metadata) {
							if (response.metadata.usage) {
								yield {
									type: "usage",
									inputTokens: response.usage.input_tokens || 0,
									outputTokens: response.usage.output_tokens || 0,
								}
							}
						}
						// Handle different response structures
						if (response.contentBlockDelta.delta.text) {
							yield {
								type: "text",
								text: response.contentBlockDelta.delta.text
							}
						}
					} catch (error) {
						console.error('Debug - Error processing chunk:', error)
						console.error('Debug - Problematic chunk:', chunk)
					}
				}
			}
			return
		}

		// Handle Anthropic models with existing implementation
		const stream = await this.anthropicClient.messages.create({
			model: modelId,
			max_tokens: model.info.maxTokens || 8192,
			temperature: 0,
			system: systemPrompt,
			messages,
			stream: true,
		})
		for await (const chunk of stream) {
			switch (chunk.type) {
				case "message_start":
					const usage = chunk.message.usage
					yield {
						type: "usage",
						inputTokens: usage.input_tokens || 0,
						outputTokens: usage.output_tokens || 0,
					}
					break
				case "message_delta":
					yield {
						type: "usage",
						inputTokens: 0,
						outputTokens: chunk.usage.output_tokens || 0,
					}
					break

				case "content_block_start":
					switch (chunk.content_block.type) {
						case "text":
							if (chunk.index > 0) {
								yield {
									type: "text",
									text: "\n",
								}
							}
							yield {
								type: "text",
								text: chunk.content_block.text,
							}
							break
					}
					break
				case "content_block_delta":
					switch (chunk.delta.type) {
						case "text_delta":
							yield {
								type: "text",
								text: chunk.delta.text,
							}
							break
					}
					break
			}
		}
	}

	private getModelId(modelId: string): string {
		// cross region inference requires prefixing the model id with the region
		if (this.options.awsUseCrossRegionInference) {
			let regionPrefix = (this.options.awsRegion || "").slice(0, 3)
			switch (regionPrefix) {
				case "us-":
					return `us.${modelId}`
				case "eu-":
					return `eu.${modelId}`
				default:
					// cross region inference is not supported in this region, falling back to default model
					return modelId
			}
		}
		return modelId
	}

	getModel(): { id: BedrockModelId; info: ModelInfo } {
		const modelId = this.options.apiModelId
		if (modelId && modelId in bedrockModels) {
			const id = modelId as BedrockModelId
			return { id, info: bedrockModels[id] }
		}
		return { id: bedrockDefaultModelId, info: bedrockModels[bedrockDefaultModelId] }
	}
}
