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
 * Captures the request payload so its encoded JSON can be asserted without HTTP.
 */
class TestableAzureOpenAISchemaEncodingProvider extends AzureOpenAIProvider
{
    public array $lastPayload = [];

    protected function request(array $payload): array
    {
        $this->lastPayload = $payload;

        return ['choices' => [['message' => ['role' => 'assistant', 'content' => 'ok'], 'finish_reason' => 'stop']]];
    }
}

describe('AzureOpenAIProvider request schemas', function () {
    beforeEach(function () {
        $this->provider = new TestableAzureOpenAISchemaEncodingProvider('test-api-key', 'https://example.openai.azure.com', 'gpt-4o');
        $this->body = function (array $options): string {
            $this->provider->chat([Message::user('hi')], $options);

            return json_encode($this->provider->lastPayload, JSON_THROW_ON_ERROR);
        };
    });

    it('states empty properties for a tool that names no parameters', function () {
        $body = ($this->body)(['tools' => [['name' => 'ping', 'description' => 'Ping']]]);

        expect($body)->toContain('{"type":"object","properties":{}}')->not->toContain('"properties":[]');
    });

    it('sends the empty properties of a tool as an object', function () {
        $tools = [['name' => 'ping', 'description' => 'Ping', 'parameters' => ['type' => 'object', 'properties' => []]]];

        expect(($this->body)(['tools' => $tools]))->toContain('{"type":"object","properties":{}}')->not->toContain('"properties":[]');
    });

    it('sends an empty object nested in a tool schema as an object', function () {
        $tools = [['name' => 'search', 'description' => 'Search', 'parameters' => [
            'type' => 'object',
            'properties' => [
                'filter' => ['type' => 'object', 'properties' => []],
                'rows' => ['type' => 'array', 'items' => ['type' => 'object']],
            ],
        ]]];

        expect(($this->body)(['tools' => $tools]))
            ->toContain('"filter":{"type":"object","properties":{}}')
            ->toContain('"items":{"type":"object","properties":{}}');
    });

    it('sends an empty object in an output schema as an object', function () {
        $schema = ['type' => 'object', 'properties' => ['meta' => ['type' => 'object', 'properties' => []]], 'required' => ['meta']];

        expect(($this->body)(['outputSchema' => $schema]))->toContain('"meta":{"type":"object","properties":{}');
    });
});
