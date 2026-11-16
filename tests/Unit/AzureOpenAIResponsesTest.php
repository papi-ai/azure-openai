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
use PapiAI\Core\StreamChunk;
use PapiAI\Core\ToolCall;

/**
 * Captures the request and replays a canned Responses body, so the wire format can be asserted
 * without HTTP.
 */
class TestableAzureResponsesProvider extends AzureOpenAIProvider
{
    public array $lastPayload = [];
    public string $lastUrl = '';
    public array $fakeResponse = ['output' => [], 'usage' => []];
    public array $fakeStreamEvents = [];

    protected function request(array $payload): array
    {
        $this->lastPayload = $payload;
        $this->lastUrl = $this->buildChatUrl();

        return $this->fakeResponse;
    }

    protected function streamRequest(array $payload): Generator
    {
        $this->lastPayload = $payload;
        $this->lastUrl = $this->buildChatUrl();

        yield from $this->fakeStreamEvents;
    }
}

describe('AzureOpenAIProvider on the Responses API', function () {
    beforeEach(function () {
        $this->provider = new TestableAzureResponsesProvider('key', 'https://myres.openai.azure.com', 'my-deployment');
        $this->chat = fn (array $messages, array $options = []) => $this->provider->chat($messages, $options);
    });

    it('posts to the v1 responses endpoint with the deployment in the body, not the path', function () {
        ($this->chat)([Message::user('Hi')]);

        expect($this->provider->lastUrl)->toBe('https://myres.openai.azure.com/openai/v1/responses');
        expect($this->provider->lastPayload['model'])->toBe('my-deployment');
    });

    it('sends input items, instructions, max_output_tokens and store false', function () {
        ($this->chat)([Message::system('Be terse.'), Message::user('Hi')], ['maxTokens' => 256]);

        expect($this->provider->lastPayload['instructions'])->toBe('Be terse.');
        expect($this->provider->lastPayload['input'])->toBe([['role' => 'user', 'content' => 'Hi']]);
        expect($this->provider->lastPayload['max_output_tokens'])->toBe(256);
        expect($this->provider->lastPayload['store'])->toBeFalse();
        expect($this->provider->lastPayload)->not->toHaveKey('messages');
    });

    it('declares tools flat and puts effort under reasoning', function () {
        ($this->chat)([Message::user('Hi')], [
            'tools' => [['name' => 'get_weather', 'description' => 'Weather', 'parameters' => ['type' => 'object', 'properties' => []]]],
            'effort' => 'high',
        ]);

        expect($this->provider->lastPayload['tools'][0]['name'])->toBe('get_weather');
        expect($this->provider->lastPayload['tools'][0])->not->toHaveKey('function');
        expect($this->provider->lastPayload['reasoning'])->toBe(['effort' => 'high']);
    });

    it('replays tool turns as function_call and function_call_output items', function () {
        ($this->chat)([
            Message::user('Weather?'),
            Message::assistant('', [new ToolCall('call_1', 'get_weather', ['city' => 'Lisbon'])]),
            Message::toolResult('call_1', 'sunny'),
        ]);

        expect($this->provider->lastPayload['input'][1])->toBe(['type' => 'function_call', 'call_id' => 'call_1', 'name' => 'get_weather', 'arguments' => '{"city":"Lisbon"}']);
        expect($this->provider->lastPayload['input'][2])->toBe(['type' => 'function_call_output', 'call_id' => 'call_1', 'output' => 'sunny']);
    });

    it('reads text and tool calls from the output items', function () {
        $this->provider->fakeResponse = [
            'status' => 'completed',
            'output' => [
                ['type' => 'function_call', 'call_id' => 'call_9', 'name' => 'get_weather', 'arguments' => '{"city":"Lisbon"}'],
                ['type' => 'message', 'role' => 'assistant', 'content' => [['type' => 'output_text', 'text' => 'Looking.']]],
            ],
            'usage' => ['input_tokens' => 3, 'output_tokens' => 2],
        ];

        $response = ($this->chat)([Message::user('Weather?')]);

        expect($response->text)->toBe('Looking.');
        expect($response->toolCalls[0]->id)->toBe('call_9');
        expect($response->stopReason)->toBe('tool_calls');
    });

    it('streams text deltas and completes on the terminal event', function () {
        $this->provider->fakeStreamEvents = [
            ['type' => 'response.output_text.delta', 'delta' => 'Hi'],
            ['type' => 'response.completed', 'response' => []],
        ];

        $chunks = iterator_to_array($this->provider->stream([Message::user('Hi')]));

        expect($this->provider->lastPayload['stream'])->toBeTrue();
        expect(array_map(fn (StreamChunk $c) => $c->text, $chunks))->toBe(['Hi', '']);
        expect(end($chunks)->isComplete)->toBeTrue();
    });
});
