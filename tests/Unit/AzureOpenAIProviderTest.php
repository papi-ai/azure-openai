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

            expect($response->usage->toArray())->toBe(['prompt_tokens' => 8, 'total_tokens' => 8]);
            expect($response->getPromptTokens())->toBe(8);
            expect($response->getTotalTokens())->toBe(8);
        });
    });
});
