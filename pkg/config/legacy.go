// Package config provides the legacy AgentConfiguration type for backward compatibility.
// New code should use the Config type from loader.go with JSON configuration files.
package config

import (
	"sync"
)

// AgentConfiguration provides configuration for the AI agent (legacy).
// Deprecated: Use Config from loader.go with JSON configuration instead.
type AgentConfiguration struct {
	mu sync.RWMutex

	// Core settings
	Name        string `json:"name"`
	Version     string `json:"version"`
	Environment string `json:"environment"`

	// Provider configurations (uses string keys for no circular imports)
	Providers map[string]*LegacyProviderConfig `json:"providers"`

	// Default model preferences (uses string keys)
	DefaultModels map[string]string `json:"default_models"`

	// Generation defaults
	Generation *LegacyGenerationDefaults `json:"generation"`

	// Memory settings
	Memory *LegacyMemorySettings `json:"memory"`

	// Rate limiting
	RateLimits *LegacyRateLimitConfig `json:"rate_limits"`

	// Logging and monitoring
	Logging *LegacyLoggingConfig `json:"logging"`

	// Security settings
	Security *LegacySecurityConfig `json:"security"`

	// Feature flags
	Features map[string]bool `json:"features"`

	// Custom settings
	Custom map[string]any `json:"custom"`
}

// LegacyProviderConfig holds configuration for a specific provider.
type LegacyProviderConfig struct {
	Enabled    bool              `json:"enabled"`
	APIKey     string            `json:"api_key"`
	BaseURL    string            `json:"base_url,omitempty"`
	OrgID      string            `json:"org_id,omitempty"`
	ProjectID  string            `json:"project_id,omitempty"`
	MaxRetries int               `json:"max_retries"`
	Models     map[string]bool   `json:"models"`
	Headers    map[string]string `json:"headers,omitempty"`
}

// LegacyGenerationDefaults holds default generation parameters.
type LegacyGenerationDefaults struct {
	Temperature      float64  `json:"temperature"`
	MaxTokens        int      `json:"max_tokens"`
	TopP             float64  `json:"top_p"`
	TopK             int      `json:"top_k"`
	FrequencyPenalty float64  `json:"frequency_penalty"`
	PresencePenalty  float64  `json:"presence_penalty"`
	StopSequences    []string `json:"stop_sequences"`
	StreamingEnabled bool     `json:"streaming_enabled"`
	RetryOnError     bool     `json:"retry_on_error"`
	MaxRetries       int      `json:"max_retries"`
}

// LegacyMemorySettings configures memory behavior.
type LegacyMemorySettings struct {
	Enabled         bool   `json:"enabled"`
	Strategy        string `json:"strategy"`
	MaxMessages     int    `json:"max_messages"`
	MaxTokens       int    `json:"max_tokens"`
	SummaryInterval int    `json:"summary_interval"`
	PersistPath     string `json:"persist_path,omitempty"`
	SemanticMemory  bool   `json:"semantic_memory"`
}

// LegacyRateLimitConfig configures global rate limiting.
type LegacyRateLimitConfig struct {
	Enabled            bool `json:"enabled"`
	RequestsPerMinute  int  `json:"requests_per_minute"`
	TokensPerMinute    int  `json:"tokens_per_minute"`
	ConcurrentRequests int  `json:"concurrent_requests"`
	QueueSize          int  `json:"queue_size"`
}

// LegacyLoggingConfig configures logging behavior.
type LegacyLoggingConfig struct {
	Level        string   `json:"level"`
	Format       string   `json:"format"`
	Output       string   `json:"output"`
	LogRequests  bool     `json:"log_requests"`
	LogResponses bool     `json:"log_responses"`
	LogTokens    bool     `json:"log_tokens"`
	LogCosts     bool     `json:"log_costs"`
	Redact       bool     `json:"redact"`
}

// LegacySecurityConfig configures security settings.
type LegacySecurityConfig struct {
	ContentFiltering bool     `json:"content_filtering"`
	MaxInputLength   int      `json:"max_input_length"`
	MaxOutputLength  int      `json:"max_output_length"`
	BlockedTopics    []string `json:"blocked_topics"`
	SanitizeHTML     bool     `json:"sanitize_html"`
	ValidateJSON     bool     `json:"validate_json"`
	PIIDetection     bool     `json:"pii_detection"`
}

// Validate validates the configuration.
func (ac *AgentConfiguration) Validate() []string {
	var errors []string

	if ac.Name == "" {
		errors = append(errors, "name is required")
	}

	if ac.Generation != nil {
		if ac.Generation.Temperature < 0 || ac.Generation.Temperature > 2 {
			errors = append(errors, "temperature must be between 0 and 2")
		}
		if ac.Generation.MaxTokens <= 0 {
			errors = append(errors, "max_tokens must be positive")
		}
	}

	return errors
}

// ConfigPresets provides pre-defined configuration presets.
type ConfigPresets struct{}

// Presets is the global presets instance.
var Presets = &ConfigPresets{}

// Development returns a development configuration.
func (cp *ConfigPresets) Development() *AgentConfiguration {
	return &AgentConfiguration{
		Name:        "AI Agent (Dev)",
		Version:     "1.0.0",
		Environment: "development",
		Providers:   make(map[string]*LegacyProviderConfig),
		DefaultModels: map[string]string{
			"text":     "gpt-4o-mini",
			"image":    "gemini-2.0-flash",
			"audio":    "gemini-2.0-flash",
			"video":    "gemini-2.5-pro",
			"document": "gemini-2.5-pro",
			"code":     "claude-sonnet-4-20250514",
		},
		Generation: &LegacyGenerationDefaults{
			Temperature:      0.7,
			MaxTokens:        4096,
			TopP:             0.9,
			StreamingEnabled: true,
			RetryOnError:     true,
			MaxRetries:       3,
		},
		Memory: &LegacyMemorySettings{
			Enabled:         true,
			Strategy:        "hybrid",
			MaxMessages:     50,
			MaxTokens:       100000,
			SummaryInterval: 20,
		},
		RateLimits: &LegacyRateLimitConfig{
			Enabled:            false,
			RequestsPerMinute:  60,
			TokensPerMinute:    100000,
			ConcurrentRequests: 10,
			QueueSize:          100,
		},
		Logging: &LegacyLoggingConfig{
			Level:        "debug",
			Format:       "text",
			Output:       "stdout",
			LogRequests:  true,
			LogResponses: true,
			LogTokens:    true,
			LogCosts:     true,
			Redact:       false,
		},
		Security: &LegacySecurityConfig{
			ContentFiltering: false,
			MaxInputLength:   1000000,
			MaxOutputLength:  100000,
			SanitizeHTML:     false,
			ValidateJSON:     true,
			PIIDetection:     false,
		},
		Features: map[string]bool{
			"rag":           true,
			"tools":         true,
			"streaming":     true,
			"caching":       false,
			"cost_tracking": true,
		},
		Custom: make(map[string]any),
	}
}

// Production returns a production configuration.
func (cp *ConfigPresets) Production() *AgentConfiguration {
	cfg := cp.Development()
	cfg.Name = "AI Agent (Prod)"
	cfg.Environment = "production"
	cfg.Logging.Level = "info"
	cfg.Logging.Format = "json"
	cfg.Logging.LogResponses = false
	cfg.Logging.Redact = true
	cfg.Security.ContentFiltering = true
	cfg.Security.PIIDetection = true
	cfg.RateLimits.Enabled = true
	cfg.Features["caching"] = true
	return cfg
}
