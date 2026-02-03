package llm

import (
	"fmt"
	"net/http"
	"time"

	"github.com/sujit/ai-agent/pkg/config"
	"github.com/sujit/ai-agent/pkg/content"
)

// ProviderFactory creates providers from configuration.
type ProviderFactory struct {
	config *config.Config
}

// NewProviderFactory creates a new provider factory.
func NewProviderFactory(cfg *config.Config) *ProviderFactory {
	return &ProviderFactory{config: cfg}
}

// CreateProvider creates a provider from configuration.
func (f *ProviderFactory) CreateProvider(providerName string) (MultimodalProvider, error) {
	providerCfg, ok := f.config.GetProvider(providerName)
	if !ok {
		return nil, fmt.Errorf("provider configuration not found: %s", providerName)
	}

	if !providerCfg.Enabled {
		return nil, fmt.Errorf("provider is disabled: %s", providerName)
	}

	apiKey := f.config.GetAPIKey(providerName)
	if apiKey == "" {
		return nil, fmt.Errorf("API key not set for provider: %s (env: %s)", providerName, providerCfg.APIKeyEnv)
	}

	switch providerName {
	case "openai":
		return f.createOpenAIProvider(apiKey, providerCfg)
	case "anthropic":
		return f.createAnthropicProvider(apiKey, providerCfg)
	case "google":
		return f.createGoogleProvider(apiKey, providerCfg)
	case "deepseek":
		return f.createDeepSeekProvider(apiKey, providerCfg)
	case "mistral":
		return f.createMistralProvider(apiKey, providerCfg)
	case "xai":
		return f.createXAIProvider(apiKey, providerCfg)
	default:
		return nil, fmt.Errorf("unknown provider: %s", providerName)
	}
}

// CreateAllProviders creates all enabled providers that have API keys set.
func (f *ProviderFactory) CreateAllProviders() ([]MultimodalProvider, error) {
	var providers []MultimodalProvider

	for _, providerName := range f.config.GetEnabledProviders() {
		provider, err := f.CreateProvider(providerName)
		if err != nil {
			// Log warning but continue with other providers
			continue
		}
		providers = append(providers, provider)
	}

	return providers, nil
}

// createOpenAIProvider creates an OpenAI provider from config.
func (f *ProviderFactory) createOpenAIProvider(apiKey string, cfg *config.ProviderCfg) (*OpenAIMultimodalProvider, error) {
	models := f.convertModels(cfg.Models, ProviderOpenAI)

	timeout := time.Duration(cfg.TimeoutSeconds) * time.Second
	if timeout == 0 {
		timeout = 120 * time.Second
	}

	p := &OpenAIMultimodalProvider{
		BaseProvider: BaseProvider{
			ProviderType: ProviderOpenAI,
			APIKey:       apiKey,
			BaseURL:      cfg.BaseURL,
			DefaultModel: f.getDefaultModel(cfg.Models),
			Models:       models,
		},
		Client: &http.Client{Timeout: timeout},
	}

	// Apply custom headers for organization/project
	if org, ok := cfg.Headers["OpenAI-Organization"]; ok {
		p.Organization = org
	}
	if proj, ok := cfg.Headers["OpenAI-Project"]; ok {
		p.Project = proj
	}

	return p, nil
}

// createAnthropicProvider creates an Anthropic provider from config.
func (f *ProviderFactory) createAnthropicProvider(apiKey string, cfg *config.ProviderCfg) (*AnthropicMultimodalProvider, error) {
	models := f.convertModels(cfg.Models, ProviderAnthropic)

	timeout := time.Duration(cfg.TimeoutSeconds) * time.Second
	if timeout == 0 {
		timeout = 120 * time.Second
	}

	apiVersion := cfg.APIVersion
	if apiVersion == "" {
		apiVersion = "2023-06-01"
	}

	p := &AnthropicMultimodalProvider{
		BaseProvider: BaseProvider{
			ProviderType: ProviderAnthropic,
			APIKey:       apiKey,
			BaseURL:      cfg.BaseURL,
			DefaultModel: f.getDefaultModel(cfg.Models),
			Models:       models,
		},
		Client:      &http.Client{Timeout: timeout},
		APIVersion:  apiVersion,
		BetaHeaders: cfg.BetaFeatures,
	}

	return p, nil
}

// createGoogleProvider creates a Google provider from config.
func (f *ProviderFactory) createGoogleProvider(apiKey string, cfg *config.ProviderCfg) (*GoogleMultimodalProvider, error) {
	models := f.convertModels(cfg.Models, ProviderGoogle)

	timeout := time.Duration(cfg.TimeoutSeconds) * time.Second
	if timeout == 0 {
		timeout = 300 * time.Second
	}

	p := &GoogleMultimodalProvider{
		BaseProvider: BaseProvider{
			ProviderType: ProviderGoogle,
			APIKey:       apiKey,
			BaseURL:      cfg.BaseURL,
			DefaultModel: f.getDefaultModel(cfg.Models),
			Models:       models,
		},
		Client: &http.Client{Timeout: timeout},
	}

	// Configure Vertex AI if enabled
	if cfg.VertexAI != nil && cfg.VertexAI.Enabled {
		p.UseVertexAI = true
		p.Location = cfg.VertexAI.Location
		// Project ID would be loaded from environment
	}

	return p, nil
}

// createDeepSeekProvider creates a DeepSeek provider from config.
func (f *ProviderFactory) createDeepSeekProvider(apiKey string, cfg *config.ProviderCfg) (*DeepSeekProvider, error) {
	models := f.convertModels(cfg.Models, ProviderDeepSeek)

	timeout := time.Duration(cfg.TimeoutSeconds) * time.Second
	if timeout == 0 {
		timeout = 120 * time.Second
	}

	p := &DeepSeekProvider{
		BaseProvider: BaseProvider{
			ProviderType: ProviderDeepSeek,
			APIKey:       apiKey,
			BaseURL:      cfg.BaseURL,
			DefaultModel: f.getDefaultModel(cfg.Models),
			Models:       models,
		},
		Client: &http.Client{Timeout: timeout},
	}

	return p, nil
}

// createMistralProvider creates a Mistral provider from config.
func (f *ProviderFactory) createMistralProvider(apiKey string, cfg *config.ProviderCfg) (*MistralProvider, error) {
	models := f.convertModels(cfg.Models, ProviderMistral)

	timeout := time.Duration(cfg.TimeoutSeconds) * time.Second
	if timeout == 0 {
		timeout = 120 * time.Second
	}

	p := &MistralProvider{
		BaseProvider: BaseProvider{
			ProviderType: ProviderMistral,
			APIKey:       apiKey,
			BaseURL:      cfg.BaseURL,
			DefaultModel: f.getDefaultModel(cfg.Models),
			Models:       models,
		},
		Client: &http.Client{Timeout: timeout},
	}

	return p, nil
}

// createXAIProvider creates an xAI provider from config.
func (f *ProviderFactory) createXAIProvider(apiKey string, cfg *config.ProviderCfg) (*XAIProvider, error) {
	models := f.convertModels(cfg.Models, ProviderXAI)

	timeout := time.Duration(cfg.TimeoutSeconds) * time.Second
	if timeout == 0 {
		timeout = 120 * time.Second
	}

	p := &XAIProvider{
		BaseProvider: BaseProvider{
			ProviderType: ProviderXAI,
			APIKey:       apiKey,
			BaseURL:      cfg.BaseURL,
			DefaultModel: f.getDefaultModel(cfg.Models),
			Models:       models,
		},
		Client: &http.Client{Timeout: timeout},
	}

	return p, nil
}

// convertModels converts config model definitions to ModelInfo.
func (f *ProviderFactory) convertModels(models map[string]*config.ModelCfg, provider ProviderType) map[string]*ModelInfo {
	result := make(map[string]*ModelInfo)

	for id, cfg := range models {
		result[id] = &ModelInfo{
			ID:              cfg.ID,
			Name:            cfg.Name,
			Provider:        provider,
			Capabilities:    f.convertCapabilities(cfg.Capabilities),
			ContextWindow:   cfg.ContextWindow,
			MaxOutputTokens: cfg.MaxOutputTokens,
			InputCostPer1K:  cfg.InputCostPer1K,
			OutputCostPer1K: cfg.OutputCostPer1K,
			SupportedFormats: f.convertMediaTypes(cfg.SupportedFormats),
			Deprecated:      cfg.Deprecated,
			ReplacedBy:      cfg.ReplacedBy,
		}
	}

	return result
}

// convertCapabilities converts string capabilities to Capability type.
func (f *ProviderFactory) convertCapabilities(caps []string) []Capability {
	result := make([]Capability, 0, len(caps))
	for _, cap := range caps {
		result = append(result, Capability(cap))
	}
	return result
}

// convertMediaTypes converts string media types to MediaType.
func (f *ProviderFactory) convertMediaTypes(types []string) []content.MediaType {
	result := make([]content.MediaType, 0, len(types))
	for _, t := range types {
		result = append(result, content.MediaType(t))
	}
	return result
}

// getDefaultModel returns the first model ID as default if available.
func (f *ProviderFactory) getDefaultModel(models map[string]*config.ModelCfg) string {
	for id := range models {
		return id
	}
	return ""
}

// RoutingConfigFromConfig creates a RoutingConfig from configuration.
func RoutingConfigFromConfig(cfg *config.Config) *RoutingConfig {
	rc := &RoutingConfig{
		DefaultTextModel:       cfg.Routing.DefaultModels["text"],
		DefaultVisionModel:     cfg.Routing.DefaultModels["vision"],
		DefaultAudioModel:      cfg.Routing.DefaultModels["audio"],
		DefaultVideoModel:      cfg.Routing.DefaultModels["video"],
		DefaultDocumentModel:   cfg.Routing.DefaultModels["document"],
		DefaultCodeModel:       cfg.Routing.DefaultModels["code"],
		DefaultReasoningModel:  cfg.Routing.DefaultModels["reasoning"],
		DefaultEmbeddingModel:  cfg.Routing.DefaultModels["embedding"],
		DefaultImageGenModel:   cfg.Routing.DefaultModels["image_generation"],
		DefaultSpeechModel:     cfg.Routing.DefaultModels["speech"],
		DefaultTranscribeModel: cfg.Routing.DefaultModels["transcription"],
		PreferCheaperModels:    cfg.Routing.PreferCheaperModels,
		MaxCostPer1KTokens:     cfg.Routing.MaxCostPer1KTokens,
		DomainRoutes:           cfg.Routing.DomainRoutes,
	}

	// Convert preferred providers
	for _, p := range cfg.Routing.PreferredProviders {
		rc.PreferredProviders = append(rc.PreferredProviders, ProviderType(p))
	}

	// Convert fallback providers
	for _, p := range cfg.Routing.FallbackProviders {
		rc.FallbackProviders = append(rc.FallbackProviders, ProviderType(p))
	}

	return rc
}

// NewProviderRegistryFromConfig creates a fully configured provider registry from config.
func NewProviderRegistryFromConfig(cfg *config.Config) (*ProviderRegistry, error) {
	factory := NewProviderFactory(cfg)
	routingConfig := RoutingConfigFromConfig(cfg)

	registry := NewProviderRegistry(routingConfig)

	providers, err := factory.CreateAllProviders()
	if err != nil {
		return nil, err
	}

	for _, provider := range providers {
		if err := registry.RegisterProvider(provider); err != nil {
			// Log warning but continue
			continue
		}
	}

	return registry, nil
}
