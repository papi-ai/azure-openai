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

use PapiAI\AzureOpenAI\AzureOpenAIProvider;
use PapiAI\Core\Contracts\EmbeddingProviderInterface;
use PapiAI\Core\Contracts\ProviderInterface;
use PapiAI\Core\EmbeddingResponse;
use PapiAI\Core\Message;
use PapiAI\Core\Response;
use PapiAI\Core\StreamChunk;
use PapiAI\Core\ToolCall;

/**
 * Test subclass that stubs HTTP methods for unit testing.
 */
class TestableAzureOpenAIProvider extends AzureOpenAIProvider
{
    public array $lastPayload = [];
    public array $lastEmbeddingPayload = [];
    public array $fakeResponse = [];
    public array $fakeEmbeddingResponse = [];
    public array $fakeStreamEvents = [];

    protected function request(array $payload): array
    {
        $this->lastPayload = $payload;

        return $this->fakeResponse;
    }

    protected function embeddingRequest(array $payload): array
    {
        $this->lastEmbeddingPayload = $payload;

        return $this->fakeEmbeddingResponse;
    }

    protected function streamRequest(array $payload): Generator
    {
        $this->lastPayload = $payload;

        foreach ($this->fakeStreamEvents as $event) {
            yield $event;
        }
    }
}

describe('AzureOpenAIProvider', function () {
    beforeEach(function () {
        $this->provider = new TestableAzureOpenAIProvider(
            apiKey: 'test-api-key',
            endpoint: 'https://myresource.openai.azure.com',
            deployment: 'gpt-4o',
        );
    });

    describe('construction', function () {
        it('implements ProviderInterface', function () {
            expect($this->provider)->toBeInstanceOf(ProviderInterface::class);
        });

        it('implements EmbeddingProviderInterface', function () {
            expect($this->provider)->toBeInstanceOf(EmbeddingProviderInterface::class);
        });

        it('returns azure-openai as name', function () {
            expect($this->provider->getName())->toBe('azure-openai');
        });
    });

    describe('capabilities', function () {
        it('supports tools', function () {
            expect($this->provider->supportsTool())->toBeTrue();
        });

        it('supports vision', function () {
            expect($this->provider->supportsVision())->toBeTrue();
        });

        it('supports structured output', function () {
            expect($this->provider->supportsStructuredOutput())->toBeTrue();
        });
    });

    describe('chat', function () {
        it('sends messages and returns a Response', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => 'Hello back!'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'gpt-4o',
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 5],
            ];

            $response = $this->provider->chat([Message::user('Hello')]);

            expect($response)->toBeInstanceOf(Response::class);
            expect($response->text)->toBe('Hello back!');
        });

        it('includes system message in messages array', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => 'OK'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'gpt-4o',
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 5],
            ];

            $this->provider->chat([
                Message::system('Be helpful'),
                Message::user('Hello'),
            ]);

            expect($this->provider->lastPayload['messages'])->toHaveCount(2);
            expect($this->provider->lastPayload['messages'][0]['role'])->toBe('system');
            expect($this->provider->lastPayload['messages'][0]['content'])->toBe('Be helpful');
            expect($this->provider->lastPayload['messages'][1]['role'])->toBe('user');
        });

        it('does not include model in payload by default (deployment is in URL)', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => 'OK'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'gpt-4o',
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 5],
            ];

            $this->provider->chat([Message::user('Hello')]);

            expect($this->provider->lastPayload)->not->toHaveKey('model');
        });

        it('overrides model and options from parameters', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => 'OK'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'gpt-4-turbo',
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 5],
            ];

            $this->provider->chat([Message::user('Hello')], [
                'model' => 'gpt-4-turbo',
                'maxTokens' => 8192,
                'temperature' => 0.5,
                'stopSequences' => ['END'],
            ]);

            expect($this->provider->lastPayload['model'])->toBe('gpt-4-turbo');
            expect($this->provider->lastPayload['max_tokens'])->toBe(8192);
            expect($this->provider->lastPayload['temperature'])->toBe(0.5);
            expect($this->provider->lastPayload['stop'])->toBe(['END']);
        });

        it('includes tools in payload converted to OpenAI format', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => 'OK'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'gpt-4o',
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 5],
            ];

            $tools = [
                [
                    'name' => 'get_weather',
                    'description' => 'Get weather',
                    'input_schema' => ['type' => 'object', 'properties' => []],
                ],
            ];

            $this->provider->chat([Message::user('Hello')], ['tools' => $tools]);

            $expected = [
                [
                    'type' => 'function',
                    'function' => [
                        'name' => 'get_weather',
                        'description' => 'Get weather',
                        'parameters' => ['type' => 'object', 'properties' => []],
                    ],
                ],
            ];
            expect($this->provider->lastPayload['tools'])->toBe($expected);
        });

        it('converts tool result messages', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => 'The weather is sunny'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'gpt-4o',
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 5],
            ];

            $this->provider->chat([
                Message::user('What is the weather?'),
                Message::assistant('Let me check', [
                    new ToolCall('tc_1', 'get_weather', ['city' => 'London']),
                ]),
                Message::toolResult('tc_1', ['temp' => 20]),
            ]);

            $messages = $this->provider->lastPayload['messages'];
            expect($messages)->toHaveCount(3);

            // Tool result message
            $toolMsg = $messages[2];
            expect($toolMsg['role'])->toBe('tool');
            expect($toolMsg['tool_call_id'])->toBe('tc_1');
        });

        it('handles response with tool calls', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => [
                            'role' => 'assistant',
                            'content' => 'Let me check',
                            'tool_calls' => [
                                [
                                    'id' => 'call_123',
                                    'type' => 'function',
                                    'function' => [
                                        'name' => 'get_weather',
                                        'arguments' => '{"city":"London"}',
                                    ],
                                ],
                            ],
                        ],
                        'finish_reason' => 'tool_calls',
                    ],
                ],
                'model' => 'gpt-4o',
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 20],
            ];

            $response = $this->provider->chat([Message::user('Weather?')]);

            expect($response->hasToolCalls())->toBeTrue();
            expect($response->toolCalls)->toHaveCount(1);
            expect($response->toolCalls[0]->name)->toBe('get_weather');
            expect($response->toolCalls[0]->arguments)->toBe(['city' => 'London']);
        });

        it('includes output schema as response_format', function () {
            $this->provider->fakeResponse = [
                'choices' => [
                    [
                        'message' => ['role' => 'assistant', 'content' => '{"name":"test"}'],
                        'finish_reason' => 'stop',
                    ],
                ],
                'model' => 'gpt-4o',
                'usage' => ['prompt_tokens' => 10, 'completion_tokens' => 5],
            ];

            $schema = ['type' => 'object', 'properties' => ['name' => ['type' => 'string']]];
            $this->provider->chat([Message::user('Hello')], ['outputSchema' => $schema]);

            expect($this->provider->lastPayload['response_format'])->toBe([
                'type' => 'json_schema',
                'json_schema' => [
                    'name' => 'response',
                    'schema' => $schema,
                ],
            ]);
        });
    });

    describe('stream', function () {
        it('yields StreamChunk for text deltas', function () {
            $this->provider->fakeStreamEvents = [
                ['choices' => [['delta' => ['content' => 'Hello'], 'finish_reason' => null]]],
                ['choices' => [['delta' => ['content' => ' world'], 'finish_reason' => null]]],
                ['choices' => [['delta' => [], 'finish_reason' => 'stop']]],
            ];

            $chunks = [];
            foreach ($this->provider->stream([Message::user('Hi')]) as $chunk) {
                $chunks[] = $chunk;
            }

            expect($chunks)->toHaveCount(3);
            expect($chunks[0])->toBeInstanceOf(StreamChunk::class);
            expect($chunks[0]->text)->toBe('Hello');
            expect($chunks[1]->text)->toBe(' world');
            expect($chunks[2]->isComplete)->toBeTrue();
        });

        it('sets stream flag in payload', function () {
            $this->provider->fakeStreamEvents = [
                ['choices' => [['delta' => [], 'finish_reason' => 'stop']]],
            ];

            iterator_to_array($this->provider->stream([Message::user('Hi')]));

            expect($this->provider->lastPayload['stream'])->toBeTrue();
        });
    });

    describe('embed', function () {
        it('embeds a single string input', function () {
            $this->provider->fakeEmbeddingResponse = [
                'data' => [
                    ['embedding' => [0.1, 0.2, 0.3], 'index' => 0],
                ],
                'model' => 'text-embedding-ada-002',
                'usage' => ['prompt_tokens' => 5, 'total_tokens' => 5],
            ];

            $response = $this->provider->embed('Hello world');

            expect($response)->toBeInstanceOf(EmbeddingResponse::class);
            expect($response->embeddings)->toBe([[0.1, 0.2, 0.3]]);
            expect($response->model)->toBe('text-embedding-ada-002');
            expect($this->provider->lastEmbeddingPayload['input'])->toBe(['Hello world']);
        });

        it('embeds an array of inputs', function () {
            $this->provider->fakeEmbeddingResponse = [
                'data' => [
                    ['embedding' => [0.1, 0.2], 'index' => 0],
                    ['embedding' => [0.3, 0.4], 'index' => 1],
                ],
                'model' => 'text-embedding-ada-002',
                'usage' => ['prompt_tokens' => 10, 'total_tokens' => 10],
            ];

            $response = $this->provider->embed(['Hello', 'World']);

            expect($response->embeddings)->toBe([[0.1, 0.2], [0.3, 0.4]]);
            expect($response->count())->toBe(2);
            expect($this->provider->lastEmbeddingPayload['input'])->toBe(['Hello', 'World']);
        });

        it('uses deployment as default model', function () {
            $this->provider->fakeEmbeddingResponse = [
                'data' => [
                    ['embedding' => [0.1], 'index' => 0],
                ],
                'model' => 'gpt-4o',
                'usage' => ['prompt_tokens' => 5, 'total_tokens' => 5],
            ];

            $this->provider->embed('Hello');

            expect($this->provider->lastEmbeddingPayload['model'])->toBe('gpt-4o');
        });

        it('passes custom dimensions', function () {
            $this->provider->fakeEmbeddingResponse = [
                'data' => [
                    ['embedding' => [0.1, 0.2], 'index' => 0],
                ],
                'model' => 'text-embedding-ada-002',
                'usage' => ['prompt_tokens' => 5, 'total_tokens' => 5],
            ];

            $this->provider->embed('Hello', ['dimensions' => 256]);

            expect($this->provider->lastEmbeddingPayload['dimensions'])->toBe(256);
        });

        it('parses usage from response', function () {
            $this->provider->fakeEmbeddingResponse = [
                'data' => [
                    ['embedding' => [0.1, 0.2, 0.3], 'index' => 0],
                ],
                'model' => 'text-embedding-ada-002',
                'usage' => ['prompt_tokens' => 8, 'total_tokens' => 8],
            ];

            $response = $this->provider->embed('Hello world');

            expect($response->usage)->toBe(['prompt_tokens' => 8, 'total_tokens' => 8]);
            expect($response->getPromptTokens())->toBe(8);
            expect($response->getTotalTokens())->toBe(8);
        });
    });
});
