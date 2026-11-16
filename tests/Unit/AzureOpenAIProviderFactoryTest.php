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
use PapiAI\AzureOpenAI\AzureOpenAIProviderFactory;
use PapiAI\Core\Effort;
use PapiAI\Core\Exception\ConfigurationException;
use PapiAI\Core\Factory\ComposerDiscovery;

describe('AzureOpenAIProviderFactory', function () {
    beforeEach(function () {
        $this->factory = new AzureOpenAIProviderFactory();
        $this->property = fn (object $provider, string $name): mixed => (fn () => $this->{$name})->call($provider);
    });

    it('is named after its package', function () {
        expect($this->factory->name())->toBe('azure-openai');
        expect($this->factory->package())->toBe('papi-ai/azure-openai');
    });

    it('is declared in composer.json, where discovery looks for it', function () {
        $manifest = json_decode((string) file_get_contents(__DIR__ . '/../../composer.json'), true);

        expect($manifest['name'])->toBe($this->factory->package());
        expect($manifest['type'])->toBe(ComposerDiscovery::PACKAGE_TYPE);
        expect($manifest['extra']['papi']['factory'])->toBe(AzureOpenAIProviderFactory::class);
    });

    it('takes what the constructor takes', function () {
        expect(array_map(fn ($setting) => $setting->name, $this->factory->settings()))->toBe(['api_key', 'endpoint', 'deployment', 'api_version', 'effort']);
    });

    it('builds the provider from its settings', function () {
        $provider = $this->factory->create([
            'api_key' => 'k',
            'endpoint' => 'https://res.openai.azure.com',
            'deployment' => 'my-deployment',
            'api_version' => '2025-01-01',
            'effort' => 'high',
        ]);

        expect($provider)->toBeInstanceOf(AzureOpenAIProvider::class);
        expect(($this->property)($provider, 'apiKey'))->toBe('k');
        expect(($this->property)($provider, 'endpoint'))->toBe('https://res.openai.azure.com');
        expect(($this->property)($provider, 'deployment'))->toBe('my-deployment');
        expect(($this->property)($provider, 'apiVersion'))->toBe('2025-01-01');
        expect(($this->property)($provider, 'defaultEffort'))->toBe(Effort::High);
    });

    it('leaves what it is not given to the provider\'s own defaults', function () {
        $provider = $this->factory->create(['api_key' => 'k', 'endpoint' => 'https://res.openai.azure.com', 'deployment' => 'my-deployment']);

        expect(($this->property)($provider, 'apiVersion'))->toBe('2024-06-01');
    });

    it('refuses to build without its required settings', function () {
        expect(fn () => $this->factory->create(['endpoint' => 'https://res.openai.azure.com', 'deployment' => 'my-deployment']))
            ->toThrow(ConfigurationException::class, 'Provider "azure-openai" needs the "api_key" setting');
        expect(fn () => $this->factory->create(['api_key' => 'k', 'deployment' => 'my-deployment']))
            ->toThrow(ConfigurationException::class, 'Provider "azure-openai" needs the "endpoint" setting');
        expect(fn () => $this->factory->create(['api_key' => 'k', 'endpoint' => 'https://res.openai.azure.com']))
            ->toThrow(ConfigurationException::class, 'Provider "azure-openai" needs the "deployment" setting');
    });
});
