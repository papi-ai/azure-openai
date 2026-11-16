# Azure OpenAI

Azure OpenAI provider for PapiAI with AAD auth support.

## Installation

```bash
composer require papi-ai/azure-openai
```

## Usage

```php
use PapiAI\AzureOpenAI\AzureOpenAIProvider;
use PapiAI\Core\Message;

$provider = new AzureOpenAIProvider(
    apiKey: 'your-azure-api-key-or-aad-token',
    endpoint: 'https://myresource.openai.azure.com',
    deployment: 'gpt-4o',
);

$response = $provider->chat([
    Message::system('You are a helpful assistant.'),
    Message::user('Hello!'),
]);

echo $response->text;
```

Uses deployment-based URLs with `api-key` header authentication. Azure AD tokens are also supported.

## Embeddings

```php
$response = $provider->embed('Hello world', [
    'model' => 'text-embedding-ada-002',
]);

$vector = $response->first();
```

## Endpoint

Chat, streaming and tool calling go through the Responses API on Azure's v1 surface (`{endpoint}/openai/v1/responses`), with the deployment named as `model` in the body and no `api-version`. Embeddings stay on the deployment URL. Each request carries the full conversation (`store: false`).

## Capabilities

| Capability | Supported |
|---|---|
| Chat | Yes |
| Streaming | Yes |
| Tool calling | Yes |
| Vision | Yes |
| Structured output | Yes |
| Embeddings | Yes |

## Creating It by Name

An application that picks its provider from configuration can build this one without naming
its class:

```php
use PapiAI\Core\Factory\ComposerDiscovery;
use PapiAI\Core\Factory\ProviderRegistry;

$registry = new ProviderRegistry(new ComposerDiscovery());

$provider = $registry->create('azure-openai', [
    'api_key' => $_ENV['AZURE_OPENAI_API_KEY'],
    'endpoint' => $_ENV['AZURE_OPENAI_ENDPOINT'],
    'deployment' => $_ENV['AZURE_OPENAI_DEPLOYMENT'],
]);
```

It takes `api_key` (required), `endpoint` (required), `deployment` (required), `api_version` and
`effort`. Anything it is not given falls back to the constructor's defaults, and a missing or
unknown setting throws `ConfigurationException` before any request is made.

## Requirements

- PHP 8.2+
- `ext-curl`
- `papi-ai/papi-core` ^0.14
- An Azure OpenAI resource with a deployed model
