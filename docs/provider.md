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

## Capabilities

| Capability | Supported |
|---|---|
| Chat | Yes |
| Streaming | Yes |
| Tool calling | Yes |
| Vision | Yes |
| Structured output | Yes |
| Embeddings | Yes |

## Requirements

- PHP 8.2+
- `ext-curl`
- `papi-ai/papi-core` ^0.14
- An Azure OpenAI resource with a deployed model
