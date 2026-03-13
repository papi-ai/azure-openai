# Azure OpenAI Provider for PapiAI

[![CI](https://github.com/papi-ai/azure-openai/workflows/CI/badge.svg)](https://github.com/papi-ai/azure-openai/actions?query=workflow%3ACI) [![Latest Version](https://img.shields.io/packagist/v/papi-ai/azure-openai.svg)](https://packagist.org/packages/papi-ai/azure-openai) [![Total Downloads](https://img.shields.io/packagist/dt/papi-ai/azure-openai.svg)](https://packagist.org/packages/papi-ai/azure-openai) [![PHP Version](https://img.shields.io/packagist/php-v/papi-ai/azure-openai.svg)](https://packagist.org/packages/papi-ai/azure-openai) [![License](https://img.shields.io/packagist/l/papi-ai/azure-openai.svg)](https://packagist.org/packages/papi-ai/azure-openai)

Azure OpenAI provider with AAD auth support for the PapiAI agent library.

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

## Embeddings

```php
$response = $provider->embed('Hello world', [
    'model' => 'text-embedding-ada-002',
]);

$vector = $response->first();
```

## License

MIT
