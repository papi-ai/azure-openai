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

use PapiAI\Core\Contracts\ProviderInterface;
use PapiAI\Core\Factory\AbstractProviderFactory;
use PapiAI\Core\Factory\Setting;
use PapiAI\Core\Factory\SettingValues;

/**
 * Builds an AzureOpenAIProvider from plain settings, so an application can select it by name.
 */
final class AzureOpenAIProviderFactory extends AbstractProviderFactory
{
    public function name(): string
    {
        return 'azure-openai';
    }

    public function package(): string
    {
        return 'papi-ai/azure-openai';
    }

    public function settings(): array
    {
        return [
            new Setting('api_key', true, 'Azure OpenAI API key'),
            new Setting('endpoint', true, 'Resource endpoint, such as https://my-resource.openai.azure.com'),
            new Setting('deployment', true, 'Deployment name, sent as the model'),
            new Setting('api_version', false, 'API version for embeddings'),
            new Setting('effort', false, 'Default reasoning effort, used when a request sets none'),
        ];
    }

    protected function build(SettingValues $values): ProviderInterface
    {
        return new AzureOpenAIProvider(...$this->argumentsGiven([
            'apiKey' => $values->string('api_key'),
            'endpoint' => $values->string('endpoint'),
            'deployment' => $values->string('deployment'),
            'apiVersion' => $values->stringOrNull('api_version'),
            'defaultEffort' => $values->effort(),
        ]));
    }
}
