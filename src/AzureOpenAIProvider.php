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
 * Azure OpenAI API Provider.
 *
 * Supports Azure-hosted OpenAI models with API key or AAD token authentication.
 *
 * API URL pattern: {endpoint}/openai/deployments/{deployment}/chat/completions?api-version={apiVersion}
 */
class AzureOpenAIProvider implements ProviderInterface, EmbeddingProviderInterface
{
    public function __construct(
        private readonly string $apiKey,
        private readonly string $endpoint,
        private readonly string $deployment,
        private readonly string $apiVersion = '2024-06-01',
    ) {
    }

    public function chat(array $messages, array $options = []): Response
    {
        $payload = $this->buildPayload($messages, $options);
        $response = $this->request($payload);

        return Response::fromOpenAI($response, $messages);
    }

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

    public function supportsTool(): bool
    {
        return true;
    }

    public function supportsVision(): bool
    {
        return true;
    }

    public function supportsStructuredOutput(): bool
    {
        return true;
    }

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
     * Make a chat completions API request.
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
     * Make an embeddings API request.
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
     * Make a streaming API request.
     *
     * @codeCoverageIgnore
     *
     * @return Generator<array>
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
