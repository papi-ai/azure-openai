<?php

/*
 * This file is part of PapiAI,
 * A simple but powerful PHP library for building AI agents.
 *
 * (c) Marcello Duarte <marcello.duarte@gmail.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

declare(strict_types=1);

namespace PapiAI\AzureOpenAI;

use Generator;
use PapiAI\Core\Contracts\EmbeddingProviderInterface;
use PapiAI\Core\Contracts\ProviderInterface;
use PapiAI\Core\EmbeddingResponse;
use PapiAI\Core\Exception\AuthenticationException;
use PapiAI\Core\Exception\ProviderException;
use PapiAI\Core\Exception\RateLimitException;
use PapiAI\Core\Message;
use PapiAI\Core\Response;
use PapiAI\Core\Role;
use PapiAI\Core\StreamChunk;
use PapiAI\Core\ToolCall;
use RuntimeException;

/**
 * Azure OpenAI API provider for PapiAI.
 *
 * Bridges PapiAI's core types (Message, Response, ToolCall) with Azure's OpenAI Service API,
 * handling format conversion in both directions. Uses the same API format as OpenAI but with
 * Azure-specific deployment-based endpoint structure. Supports chat completions, streaming,
 * tool calling, vision (multimodal), structured JSON output, and text embeddings.
 *
 * Authentication is via api-key header. All HTTP is done with ext-curl directly,
 * with no HTTP abstraction layer.
 *
 * URL pattern: {endpoint}/openai/deployments/{deployment}/{operation}?api-version={version}
 *
 * @see https://learn.microsoft.com/en-us/azure/ai-services/openai/reference
 */
class AzureOpenAIProvider implements ProviderInterface, EmbeddingProviderInterface
{
    /**
     * Create a new Azure OpenAI provider instance.
     *
     * @param string $apiKey     Azure OpenAI API key for authentication
     * @param string $endpoint   Azure resource endpoint URL (e.g. https://myresource.openai.azure.com)
     * @param string $deployment Model deployment name configured in Azure portal
     * @param string $apiVersion Azure OpenAI API version to use
     */
    public function __construct(
        private readonly string $apiKey,
        private readonly string $endpoint,
        private readonly string $deployment,
        private readonly string $apiVersion = '2024-06-01',
    ) {
    }

    /**
     * Send a chat completion request to the Azure OpenAI API.
     *
     * Converts PapiAI Messages to OpenAI's chat format, sends the request to the
     * configured deployment, and parses the response back into a core Response object.
     * Supports tools, vision, structured output, and custom generation parameters.
     *
     * @param array<Message> $messages Conversation history as PapiAI Message objects
     * @param array{
     *     model?: string,
     *     tools?: array,
     *     maxTokens?: int,
     *     temperature?: float,
     *     stopSequences?: array<string>,
     *     outputSchema?: array,
     * } $options Request options (model, tools, maxTokens, temperature, etc.)
     *
     * @return Response Parsed response containing text, tool calls, usage, and stop reason
     *
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When the API returns any other error (HTTP 4xx/5xx)
     * @throws RuntimeException        When the cURL request itself fails
     */
    public function chat(array $messages, array $options = []): Response
    {
        $payload = $this->buildPayload($messages, $options);
        $response = $this->request($payload);

        return Response::fromOpenAI($response, $messages);
    }

    /**
     * Stream a chat completion from the Azure OpenAI API using server-sent events.
     *
     * Yields StreamChunk objects as partial responses arrive. The final chunk
     * has isComplete=true. Only text content is streamed.
     *
     * @param array<Message> $messages Conversation history as PapiAI Message objects
     * @param array $options Request options (model, tools, maxTokens, temperature, etc.)
     *
     * @return iterable<StreamChunk> Stream of text chunks, ending with a completion marker
     *
     * @throws RuntimeException When the cURL request fails
     */
    public function stream(array $messages, array $options = []): iterable
    {
        $payload = $this->buildPayload($messages, $options);
        $payload['stream'] = true;

        foreach ($this->streamRequest($payload) as $event) {
            $delta = $event['choices'][0]['delta'] ?? [];
            if (isset($delta['content'])) {
                yield new StreamChunk($delta['content']);
            }
            if (($event['choices'][0]['finish_reason'] ?? null) !== null) {
                yield new StreamChunk('', isComplete: true);
            }
        }
    }

    /**
     * Whether this provider supports function/tool calling.
     *
     * Azure OpenAI supports tool calling via function definitions in the OpenAI format.
     */
    public function supportsTool(): bool
    {
        return true;
    }

    /**
     * Whether this provider supports vision (multimodal image input).
     *
     * Azure OpenAI supports image URLs and base64-encoded images in message content.
     */
    public function supportsVision(): bool
    {
        return true;
    }

    /**
     * Whether this provider supports structured JSON output via response_format.
     *
     * Azure OpenAI supports json_schema response format for structured output.
     */
    public function supportsStructuredOutput(): bool
    {
        return true;
    }

    /**
     * Generate text embeddings via the Azure OpenAI Embeddings API.
     *
     * @param string|array<string> $input  Text or array of texts to embed
     * @param array{
     *     model?: string,
     *     dimensions?: int,
     * } $options Embedding options (model defaults to the configured deployment)
     *
     * @return EmbeddingResponse Embedding vectors, model name, and usage statistics
     *
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When the API returns any other error (HTTP 4xx/5xx)
     * @throws RuntimeException        When the cURL request itself fails
     */
    public function embed(string|array $input, array $options = []): EmbeddingResponse
    {
        $model = $options['model'] ?? $this->deployment;
        $payload = [
            'model' => $model,
            'input' => is_array($input) ? $input : [$input],
        ];

        if (isset($options['dimensions'])) {
            $payload['dimensions'] = $options['dimensions'];
        }

        $response = $this->embeddingRequest($payload);

        $embeddings = array_map(
            fn (array $item) => $item['embedding'],
            $response['data']
        );

        return new EmbeddingResponse(
            embeddings: $embeddings,
            model: $response['model'] ?? $model,
            usage: $response['usage'] ?? [],
        );
    }

    /**
     * Get the unique identifier for this provider.
     *
     * @return string Provider name used for error reporting and identification
     */
    public function getName(): string
    {
        return 'azure-openai';
    }

    /**
     * Build the chat completions API URL.
     */
    private function buildChatUrl(): string
    {
        $base = rtrim($this->endpoint, '/');

        return "{$base}/openai/deployments/{$this->deployment}/chat/completions?api-version={$this->apiVersion}";
    }

    /**
     * Build the embeddings API URL.
     */
    private function buildEmbeddingsUrl(): string
    {
        $base = rtrim($this->endpoint, '/');

        return "{$base}/openai/deployments/{$this->deployment}/embeddings?api-version={$this->apiVersion}";
    }

    /**
     * Build the API request payload.
     */
    private function buildPayload(array $messages, array $options): array
    {
        $apiMessages = [];

        foreach ($messages as $message) {
            if ($message instanceof Message) {
                $apiMessages[] = $this->convertMessage($message);
            }
        }

        $payload = [
            'messages' => $apiMessages,
        ];

        if (isset($options['model'])) {
            $payload['model'] = $options['model'];
        }

        if (isset($options['maxTokens'])) {
            $payload['max_tokens'] = $options['maxTokens'];
        }

        if (isset($options['temperature'])) {
            $payload['temperature'] = $options['temperature'];
        }

        if (isset($options['stopSequences'])) {
            $payload['stop'] = $options['stopSequences'];
        }

        // Handle structured output / JSON mode
        if (isset($options['outputSchema'])) {
            $payload['response_format'] = [
                'type' => 'json_schema',
                'json_schema' => [
                    'name' => 'response',
                    'schema' => $options['outputSchema'],
                ],
            ];
        }

        // Handle tools
        if (isset($options['tools']) && !empty($options['tools'])) {
            $payload['tools'] = $this->convertTools($options['tools']);
        }

        return $payload;
    }

    /**
     * Convert a Message to OpenAI-compatible API format.
     */
    private function convertMessage(Message $message): array
    {
        $apiMessage = [
            'role' => $this->convertRole($message->role),
        ];

        if ($message->isTool()) {
            $apiMessage['role'] = 'tool';
            $apiMessage['content'] = $message->content;
            $apiMessage['tool_call_id'] = $message->toolCallId;
        } elseif ($message->hasToolCalls()) {
            $apiMessage['content'] = $message->getText() ?: null;
            $apiMessage['tool_calls'] = array_map(function (ToolCall $tc) {
                return [
                    'id' => $tc->id,
                    'type' => 'function',
                    'function' => [
                        'name' => $tc->name,
                        'arguments' => json_encode($tc->arguments),
                    ],
                ];
            }, $message->toolCalls);
        } elseif (is_array($message->content)) {
            $apiMessage['content'] = $this->convertMultimodalContent($message->content);
        } else {
            $apiMessage['content'] = $message->content;
        }

        return $apiMessage;
    }

    /**
     * Convert multimodal content to OpenAI-compatible format.
     */
    private function convertMultimodalContent(array $content): array
    {
        $parts = [];

        foreach ($content as $part) {
            if ($part['type'] === 'text') {
                $parts[] = ['type' => 'text', 'text' => $part['text']];
            } elseif ($part['type'] === 'image') {
                $source = $part['source'];
                if ($source['type'] === 'url') {
                    $parts[] = [
                        'type' => 'image_url',
                        'image_url' => ['url' => $source['url']],
                    ];
                } else {
                    $parts[] = [
                        'type' => 'image_url',
                        'image_url' => [
                            'url' => "data:{$source['media_type']};base64,{$source['data']}",
                        ],
                    ];
                }
            }
        }

        return $parts;
    }

    /**
     * Convert tools from PapiAI format to OpenAI-compatible format.
     */
    private function convertTools(array $tools): array
    {
        $openaiTools = [];

        foreach ($tools as $tool) {
            if (is_array($tool)) {
                $openaiTools[] = [
                    'type' => 'function',
                    'function' => [
                        'name' => $tool['name'],
                        'description' => $tool['description'],
                        'parameters' => $tool['input_schema'] ?? $tool['parameters'] ?? ['type' => 'object', 'properties' => []],
                    ],
                ];
            }
        }

        return $openaiTools;
    }

    /**
     * Convert Role to OpenAI-compatible role string.
     */
    private function convertRole(Role $role): string
    {
        return match ($role) {
            Role::System => 'system',
            Role::User => 'user',
            Role::Assistant => 'assistant',
            Role::Tool => 'tool',
        };
    }

    /**
     * Check the HTTP status code and throw appropriate exceptions.
     *
     * @param int $httpCode
     * @param array<string, mixed>|null $data
     * @param array<string, string> $responseHeaders
     *
     * @throws AuthenticationException
     * @throws RateLimitException
     * @throws ProviderException
     */
    private function handleErrorResponse(int $httpCode, ?array $data, array $responseHeaders = []): void
    {
        if ($httpCode < 400) {
            return;
        }

        if ($httpCode === 401) {
            throw new AuthenticationException(
                provider: 'azure-openai',
                statusCode: $httpCode,
                responseBody: $data,
            );
        }

        if ($httpCode === 429) {
            $retryAfter = isset($responseHeaders['retry-after'])
                ? (int) $responseHeaders['retry-after']
                : null;

            throw new RateLimitException(
                provider: 'azure-openai',
                retryAfterSeconds: $retryAfter,
                statusCode: $httpCode,
                responseBody: $data,
            );
        }

        $errorMessage = $data['error']['message'] ?? 'Unknown error';

        throw new ProviderException(
            message: "Azure OpenAI API error ({$httpCode}): {$errorMessage}",
            provider: 'azure-openai',
            statusCode: $httpCode,
            responseBody: $data,
        );
    }

    /**
     * Parse response headers from a cURL header callback.
     *
     * @param string $rawHeaders
     *
     * @return array<string, string>
     */
    private function parseResponseHeaders(string $rawHeaders): array
    {
        $headers = [];
        foreach (explode("\r\n", $rawHeaders) as $line) {
            if (str_contains($line, ':')) {
                [$key, $value] = explode(':', $line, 2);
                $headers[strtolower(trim($key))] = trim($value);
            }
        }

        return $headers;
    }

    /**
     * Make a chat completions API request via cURL.
     *
     * Sends a POST request to the Azure OpenAI chat completions endpoint
     * and returns the decoded JSON response.
     *
     * @param array<string, mixed> $payload JSON-serializable request body
     *
     * @return array<string, mixed> Decoded API response
     *
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When the API returns any other error (HTTP 4xx/5xx)
     * @throws RuntimeException        When the cURL request itself fails
     *
     * @codeCoverageIgnore
     */
    protected function request(array $payload): array
    {
        $url = $this->buildChatUrl();
        $ch = curl_init($url);

        $rawHeaders = '';
        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => [
                'Content-Type: application/json',
                'api-key: ' . $this->apiKey,
            ],
            CURLOPT_HEADERFUNCTION => function ($ch, $header) use (&$rawHeaders) {
                $rawHeaders .= $header;

                return strlen($header);
            },
        ]);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);

        curl_close($ch);

        if ($error !== '') {
            throw new RuntimeException("Azure OpenAI API request failed: {$error}");
        }

        $data = json_decode($response, true);
        $this->handleErrorResponse($httpCode, $data, $this->parseResponseHeaders($rawHeaders));

        return $data;
    }

    /**
     * Make an embeddings API request via cURL.
     *
     * Sends a POST request to the Azure OpenAI embeddings endpoint
     * and returns the decoded JSON response.
     *
     * @param array<string, mixed> $payload JSON-serializable request body
     *
     * @return array<string, mixed> Decoded API response
     *
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When the API returns any other error (HTTP 4xx/5xx)
     * @throws RuntimeException        When the cURL request itself fails
     *
     * @codeCoverageIgnore
     */
    protected function embeddingRequest(array $payload): array
    {
        $url = $this->buildEmbeddingsUrl();
        $ch = curl_init($url);

        $rawHeaders = '';
        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => [
                'Content-Type: application/json',
                'api-key: ' . $this->apiKey,
            ],
            CURLOPT_HEADERFUNCTION => function ($ch, $header) use (&$rawHeaders) {
                $rawHeaders .= $header;

                return strlen($header);
            },
        ]);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);

        curl_close($ch);

        if ($error !== '') {
            throw new RuntimeException("Azure OpenAI Embeddings API request failed: {$error}");
        }

        $data = json_decode($response, true);
        $this->handleErrorResponse($httpCode, $data, $this->parseResponseHeaders($rawHeaders));

        return $data;
    }

    /**
     * Make a streaming chat completions request via cURL.
     *
     * Buffers the full SSE response, then parses and yields each event as a
     * decoded JSON array. Stops when the [DONE] sentinel is received.
     *
     * @param array<string, mixed> $payload JSON-serializable request body (stream flag should be set)
     *
     * @return Generator<int, array<string, mixed>> Decoded SSE events from the API
     *
     * @codeCoverageIgnore
     */
    protected function streamRequest(array $payload): Generator
    {
        $url = $this->buildChatUrl();
        $ch = curl_init($url);

        $buffer = '';
        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_HTTPHEADER => [
                'Content-Type: application/json',
                'api-key: ' . $this->apiKey,
            ],
            CURLOPT_WRITEFUNCTION => function ($ch, $data) use (&$buffer) {
                $buffer .= $data;

                return strlen($data);
            },
        ]);

        curl_exec($ch);
        curl_close($ch);

        // Parse SSE events
        $lines = explode("\n", $buffer);
        foreach ($lines as $line) {
            $line = trim($line);
            if (str_starts_with($line, 'data: ')) {
                $json = substr($line, 6);
                if ($json === '[DONE]') {
                    break;
                }
                $event = json_decode($json, true);
                if ($event !== null) {
                    yield $event;
                }
            }
        }
    }
}
