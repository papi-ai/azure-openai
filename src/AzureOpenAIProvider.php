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
use PapiAI\Core\Contracts\NamedToolSelectableInterface;
use PapiAI\Core\Contracts\ProviderInterface;
use PapiAI\Core\Effort;
use PapiAI\Core\EmbeddingResponse;
use PapiAI\Core\Exception\AuthenticationException;
use PapiAI\Core\Exception\ProviderException;
use PapiAI\Core\Exception\RateLimitException;
use PapiAI\Core\Exception\UnknownEffortException;
use PapiAI\Core\Message;
use PapiAI\Core\Response;
use PapiAI\Core\Role;
use PapiAI\Core\StopReason;
use PapiAI\Core\StreamChunk;
use PapiAI\Core\ToolCall;
use PapiAI\Core\ToolChoice;
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
 * Chat uses the v1 surface, {endpoint}/openai/v1/responses, with the deployment named in the body.
 * Embeddings use {endpoint}/openai/deployments/{deployment}/embeddings?api-version={version}.
 *
 * @see https://learn.microsoft.com/en-us/azure/ai-services/openai/reference
 *
 * @psalm-import-type ChatOptions from ProviderInterface
 */
class AzureOpenAIProvider implements ProviderInterface, EmbeddingProviderInterface, NamedToolSelectableInterface
{
    /**
     * The effort levels safe to send to any Azure deployment.
     *
     * Deliberately narrower than the OpenAI provider's. There the model name is in the request, so
     * the accepted levels can be worked out from it; here the request carries a **deployment
     * name**, which the customer chooses freely and which reveals nothing about the model behind
     * it. Since OpenAI rejects an unknown level outright rather than ignoring it, guessing would
     * turn a hint into a 400. These three are accepted by every reasoning model.
     *
     * @var non-empty-list<Effort>
     */
    private const OFFERED_EFFORT = [Effort::Low, Effort::Medium, Effort::High];

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
        private readonly ?Effort $defaultEffort = null,
    ) {
    }

    /**
     * Send a conversation to the Responses API and return the model's turn.
     *
     * @param Message[] $messages Conversation history
     * @param array     $options  Provider options (model, maxTokens, temperature, tools, toolChoice, effort, outputSchema)
     */
    public function chat(array $messages, array $options = []): Response
    {
        $response = $this->request($this->buildPayload($messages, $options));

        return $this->parseResponse($response, $messages);
    }

    /**
     * Stream a response, yielding text deltas and a final completion marker.
     *
     * @param Message[] $messages Conversation history
     * @param array     $options  Provider options
     *
     * @return iterable<StreamChunk>
     */
    public function stream(array $messages, array $options = []): iterable
    {
        $payload = $this->buildPayload($messages, $options);
        $payload['stream'] = true;

        foreach ($this->streamRequest($payload) as $event) {
            $type = $event['type'] ?? '';

            if ($type === 'response.output_text.delta') {
                yield new StreamChunk((string) ($event['delta'] ?? ''));
            }

            if (in_array($type, ['response.completed', 'response.incomplete', 'response.failed'], true)) {
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
     * The Responses endpoint on Azure's v1 surface. The deployment is named in the body, not the
     * path, and the v1 surface takes no api-version.
     */
    protected function buildChatUrl(): string
    {
        return rtrim($this->endpoint, '/') . '/openai/v1/responses';
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
     * Build the Responses API request body. `model` is the deployment name on Azure's v1 surface.
     * The history is sent in full each time (`store: false`).
     */
    private function buildPayload(array $messages, array $options): array
    {
        [$instructions, $input] = $this->splitConversation($messages);

        $payload = ['model' => $options['model'] ?? $this->deployment, 'input' => $input, 'store' => false];

        if ($instructions !== null) {
            $payload['instructions'] = $instructions;
        }

        return $payload + $this->generationSettings($options);
    }

    /**
     * The system message, which Responses takes as `instructions`, and everything else as items.
     *
     * @param Message[] $messages
     *
     * @return array{0: string|null, 1: list<array<string, mixed>>}
     */
    private function splitConversation(array $messages): array
    {
        $instructions = null;
        $input = [];

        foreach ($messages as $message) {
            if (!$message instanceof Message) {
                continue;
            }

            if ($message->isSystem()) {
                $instructions = $message->getText();

                continue;
            }

            foreach ($this->toInputItems($message) as $item) {
                $input[] = $item;
            }
        }

        return [$instructions, $input];
    }

    /**
     * The optional request fields: output ceiling, sampling, schema, tools, forced choice, effort.
     *
     * @return array<string, mixed>
     */
    private function generationSettings(array $options): array
    {
        $settings = [];

        if (isset($options['maxTokens'])) {
            $settings['max_output_tokens'] = $options['maxTokens'];
        }

        if (isset($options['temperature'])) {
            $settings['temperature'] = $options['temperature'];
        }

        if (isset($options['outputSchema'])) {
            $settings['text'] = ['format' => ['type' => 'json_schema', 'name' => 'response', 'schema' => $options['outputSchema']]];
        }

        if (!empty($options['tools'])) {
            $settings['tools'] = $this->convertTools($options['tools']);
        }

        // Validation lives in core and throws before any HTTP call, tools or no tools.
        if (isset($options['toolChoice'])) {
            $choice = ToolChoice::fromOption($options['toolChoice'], $options['tools'] ?? []);

            if (!empty($options['tools'])) {
                $settings['tool_choice'] = $this->toolChoiceFor($choice);
            }
        }

        $effort = $this->effortFor($options);

        if ($effort !== null) {
            $settings['reasoning'] = ['effort' => $effort->nearestOf(self::OFFERED_EFFORT)->value];
        }

        return $settings;
    }

    /**
     * Responses spells a named tool flat, and the modes as bare strings.
     *
     * @return array{type: string, name: string}|string
     */
    private function toolChoiceFor(ToolChoice $choice): array|string
    {
        if ($choice->toolName !== null) {
            return ['type' => 'function', 'name' => $choice->toolName];
        }

        return match ($choice->mode) {
            ToolChoice::NONE => 'none',
            ToolChoice::REQUIRED => 'required',
            default => 'auto',
        };
    }

    /**
     * Translate one PapiAI message into Responses input items.
     *
     * @return list<array<string, mixed>>
     */
    private function toInputItems(Message $message): array
    {
        if ($message->isTool()) {
            return [[
                'type' => 'function_call_output',
                'call_id' => $message->toolCallId,
                'output' => is_string($message->content) ? $message->content : json_encode($message->content),
            ]];
        }

        if ($message->hasToolCalls()) {
            $items = [];

            if ($message->getText() !== '') {
                $items[] = ['role' => 'assistant', 'content' => [['type' => 'output_text', 'text' => $message->getText()]]];
            }

            foreach ($message->toolCalls as $toolCall) {
                $items[] = ['type' => 'function_call', 'call_id' => $toolCall->id, 'name' => $toolCall->name, 'arguments' => json_encode($toolCall->arguments)];
            }

            return $items;
        }

        if (is_array($message->content)) {
            return [['role' => $this->convertRole($message->role), 'content' => $this->convertMultimodalContent($message->content)]];
        }

        if ($message->role === Role::Assistant) {
            return [['role' => 'assistant', 'content' => [['type' => 'output_text', 'text' => $message->content]]]];
        }

        return [['role' => $this->convertRole($message->role), 'content' => $message->content]];
    }

    /**
     * Read the model's turn out of a Responses body: text from message items, calls from
     * function_call items; reasoning items are skipped.
     *
     * @param array<string, mixed> $response
     * @param Message[]            $messages
     */
    private function parseResponse(array $response, array $messages): Response
    {
        $text = '';
        $toolCalls = [];

        foreach ($response['output'] ?? [] as $item) {
            $type = $item['type'] ?? '';

            if ($type === 'message') {
                foreach ($item['content'] ?? [] as $part) {
                    if (($part['type'] ?? '') === 'output_text') {
                        $text .= $part['text'] ?? '';
                    }
                }
            }

            if ($type === 'function_call') {
                $toolCalls[] = new ToolCall(
                    id: (string) ($item['call_id'] ?? $item['id'] ?? ''),
                    name: (string) ($item['name'] ?? ''),
                    arguments: json_decode((string) ($item['arguments'] ?? '{}'), true) ?? [],
                );
            }
        }

        $incomplete = ($response['status'] ?? 'completed') === 'incomplete';

        return new Response(
            text: $text,
            toolCalls: $toolCalls,
            messages: $messages,
            usage: $response['usage'] ?? [],
            stopReason: $incomplete ? (string) ($response['incomplete_details']['reason'] ?? 'incomplete') : ($toolCalls !== [] ? 'tool_calls' : 'stop'),
            stoppedBecause: $this->stoppedBecauseOf($response, $toolCalls !== []),
        );
    }

    /**
     * The effort this request asks for: the per-call option, else the provider default.
     *
     * @param array<string, mixed> $options The caller's request options
     *
     * @throws UnknownEffortException When the level is not one core defines
     */
    private function effortFor(array $options): ?Effort
    {
        if (!isset($options['effort'])) {
            return $this->defaultEffort;
        }

        $level = (string) $options['effort'];

        return Effort::tryFrom($level) ?? throw new UnknownEffortException($level);
    }

    /**
     * Convert multimodal parts to Responses input parts. A base64 image travels as a data URL.
     *
     * @param array<array{type: string, text?: string, source?: array}> $content
     *
     * @return list<array<string, string>>
     */
    private function convertMultimodalContent(array $content): array
    {
        $parts = [];

        foreach ($content as $part) {
            if ($part['type'] === 'text') {
                $parts[] = ['type' => 'input_text', 'text' => $part['text']];

                continue;
            }

            if ($part['type'] !== 'image') {
                continue;
            }

            $source = $part['source'];
            $parts[] = [
                'type' => 'input_image',
                'image_url' => $source['type'] === 'url' ? $source['url'] : sprintf('data:%s;base64,%s', $source['media_type'], $source['data']),
            ];
        }

        return $parts;
    }

    /**
     * Convert PapiAI tool definitions to the flat Responses tool shape.
     *
     * @param array<array{name: string, description: string, input_schema?: array, parameters?: array}> $tools
     *
     * @return list<array{type: string, name: string, description: string, parameters: array}>
     */
    private function convertTools(array $tools): array
    {
        $declared = [];

        foreach ($tools as $tool) {
            if (is_array($tool)) {
                $declared[] = [
                    'type' => 'function',
                    'name' => $tool['name'],
                    'description' => $tool['description'],
                    'parameters' => $tool['input_schema'] ?? $tool['parameters'] ?? ['type' => 'object', 'properties' => []],
                ];
            }
        }

        return $declared;
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

    /**
     * Why the model stopped, in core's vocabulary.
     *
     * @param array<string, mixed> $response
     */
    private function stoppedBecauseOf(array $response, bool $calledTools): StopReason
    {
        if ($calledTools) {
            return StopReason::ToolCalls;
        }

        return match ($response['status'] ?? 'completed') {
            'completed' => StopReason::Stop,
            'incomplete' => match ($response['incomplete_details']['reason'] ?? null) {
                'max_output_tokens' => StopReason::MaxTokens,
                'content_filter' => StopReason::Refused,
                default => StopReason::Other,
            },
            default => StopReason::Other,
        };
    }
}
