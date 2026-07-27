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
 * Captures the request payload so tool-choice mapping can be asserted without HTTP.
 */
class TestableAzureOpenAIToolChoiceProvider extends AzureOpenAIProvider
{
    public array $lastPayload = [];

    protected function request(array $payload): array
    {
        $this->lastPayload = $payload;

        return ['choices' => [['message' => ['role' => 'assistant', 'content' => 'ok'], 'finish_reason' => 'stop']]];
    }
}

describe('AzureOpenAIProvider tool choice', function () {
    beforeEach(function () {
        $this->provider = new TestableAzureOpenAIToolChoiceProvider(
            'test-api-key',
            'https://example.openai.azure.com',
            'gpt-4o',
        );
        $this->tools = [
            ['name' => 'get_weather', 'description' => 'Weather', 'parameters' => ['type' => 'object']],
        ];
    });

    it('maps auto/none/required to the OpenAI-compatible strings', function () {
        $this->provider->chat([Message::user('hi')], ['tools' => $this->tools, 'toolChoice' => 'auto']);
        expect($this->provider->lastPayload['tool_choice'])->toBe('auto');

        $this->provider->chat([Message::user('hi')], ['tools' => $this->tools, 'toolChoice' => 'none']);
        expect($this->provider->lastPayload['tool_choice'])->toBe('none');

        $this->provider->chat([Message::user('hi')], ['tools' => $this->tools, 'toolChoice' => 'required']);
        expect($this->provider->lastPayload['tool_choice'])->toBe('required');
    });

    it('maps a specific tool to the function form', function () {
        $this->provider->chat([Message::user('hi')], ['tools' => $this->tools, 'toolChoice' => ['name' => 'get_weather']]);

        expect($this->provider->lastPayload['tool_choice'])->toBe(['type' => 'function', 'function' => ['name' => 'get_weather']]);
    });

    it('emits no tool_choice when absent (backward compatible)', function () {
        $this->provider->chat([Message::user('hi')], ['tools' => $this->tools]);

        expect($this->provider->lastPayload)->not->toHaveKey('tool_choice');
    });

    it('throws for an unenforceable choice, before any HTTP call', function () {
        expect(fn () => $this->provider->chat([Message::user('hi')], ['toolChoice' => 'required']))
            ->toThrow(InvalidArgumentException::class);
        expect($this->provider->lastPayload)->toBe([]);
    });
});
