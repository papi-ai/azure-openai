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
use PapiAI\Core\Message;

/**
 * Captures the request payload so effort mapping can be asserted without HTTP.
 */
class TestableAzureEffortProvider extends AzureOpenAIProvider
{
    public array $lastPayload = [];

    protected function request(array $payload): array
    {
        $this->lastPayload = $payload;

        return ['choices' => [['message' => ['role' => 'assistant', 'content' => 'ok'], 'finish_reason' => 'stop']]];
    }
}

describe('AzureOpenAIProvider reasoning effort', function () {
    beforeEach(function () {
        $this->provider = new TestableAzureEffortProvider(
            'test-api-key',
            'https://example.openai.azure.com',
            'gpt-5',
        );
        $this->chat = fn (array $options) => $this->provider->chat([Message::user('hi')], $options);
    });

    it('passes the level straight through, which is OpenAI\'s own vocabulary', function () {
        foreach (['low', 'medium', 'high'] as $level) {
            ($this->chat)(['effort' => $level]);

            expect($this->provider->lastPayload['reasoning_effort'])->toBe($level);
        }
    });

    it('sends nothing when the caller does not ask', function () {
        ($this->chat)([]);

        expect($this->provider->lastPayload)->not->toHaveKey('reasoning_effort');
    });

    it('rejects a level it does not recognise, before any HTTP call', function () {
        expect(fn () => ($this->chat)(['effort' => 'enormous']))
            ->toThrow(InvalidArgumentException::class, 'enormous');

        expect($this->provider->lastPayload)->toBe([]);
    });
});
