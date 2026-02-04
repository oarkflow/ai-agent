package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// ------------------------------
// Main Configuration Structures
// ------------------------------

// Config represents the complete application configuration loaded from JSON.
type Config struct {
	Version     string          `json:"version"`
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Providers   ProvidersConfig `json:"providers"`
	Routing     RoutingConfig   `json:"routing"`
	Generation  GenerationCfg   `json:"generation"`
	Memory      MemoryCfg       `json:"memory"`
	RateLimits  RateLimitsCfg   `json:"rate_limits"`
	Logging     LoggingCfg      `json:"logging"`
	Security    SecurityCfg     `json:"security"`
	Features    FeaturesCfg     `json:"features"`
	Agent       AgentCfg        `json:"agent"`

	// Loaded from separate files
	Prompts *PromptsConfig `json:"-"`
	Tools   *ToolsConfig   `json:"-"`
	Domains *DomainsConfig `json:"-"`

	// Internal
	configPath string
	loadedAt   time.Time
	mu         sync.RWMutex
}

// ProvidersConfig contains all provider configurations.
type ProvidersConfig map[string]*ProviderCfg

// ProviderCfg represents a single provider's configuration.
type ProviderCfg struct {
	Enabled        bool                  `json:"enabled"`
	APIKeyEnv      string                `json:"api_key_env"`
	BaseURL        string                `json:"base_url"`
	APIVersion     string                `json:"api_version,omitempty"`
	TimeoutSeconds int                   `json:"timeout_seconds"`
	MaxRetries     int                   `json:"max_retries"`
	Headers        map[string]string     `json:"headers,omitempty"`
	BetaFeatures   []string              `json:"beta_features,omitempty"`
	VertexAI       *VertexAICfg          `json:"vertex_ai,omitempty"`
	Models         map[string]*ModelCfg  `json:"models"`
	RateLimit      *ProviderRateLimitCfg `json:"rate_limit,omitempty"`
}

// VertexAICfg contains Vertex AI specific configuration.
type VertexAICfg struct {
	Enabled      bool   `json:"enabled"`
	ProjectIDEnv string `json:"project_id_env"`
	Location     string `json:"location"`
}

// ModelCfg represents a model's configuration.
type ModelCfg struct {
	ID               string   `json:"id"`
	Name             string   `json:"name"`
	Capabilities     []string `json:"capabilities"`
	ContextWindow    int      `json:"context_window"`
	MaxOutputTokens  int      `json:"max_output_tokens"`
	InputCostPer1K   float64  `json:"input_cost_per_1k"`
	OutputCostPer1K  float64  `json:"output_cost_per_1k"`
	SupportedFormats []string `json:"supported_formats,omitempty"`
	Deprecated       bool     `json:"deprecated,omitempty"`
	ReplacedBy       string   `json:"replaced_by,omitempty"`
}

// ProviderRateLimitCfg contains provider-specific rate limits.
type ProviderRateLimitCfg struct {
	RequestsPerMinute int `json:"requests_per_minute"`
	TokensPerMinute   int `json:"tokens_per_minute"`
	TokensPerDay      int `json:"tokens_per_day"`
}

// RoutingConfig contains routing configuration.
type RoutingConfig struct {
	DefaultModels       map[string]string `json:"default_models"`
	PreferCheaperModels bool              `json:"prefer_cheaper_models"`
	MaxCostPer1KTokens  float64           `json:"max_cost_per_1k_tokens"`
	PreferredProviders  []string          `json:"preferred_providers"`
	FallbackProviders   []string          `json:"fallback_providers"`
	DomainRoutes        map[string]string `json:"domain_routes"`
}

// GenerationCfg contains generation configuration.
type GenerationCfg struct {
	Defaults     GenerationDefaultsCfg            `json:"defaults"`
	TaskSpecific map[string]GenerationDefaultsCfg `json:"task_specific"`
}

// GenerationDefaultsCfg contains default generation parameters.
type GenerationDefaultsCfg struct {
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

// MemoryCfg contains memory configuration.
type MemoryCfg struct {
	Enabled          bool              `json:"enabled"`
	Strategy         string            `json:"strategy"`
	MaxMessages      int               `json:"max_messages"`
	MaxTokens        int               `json:"max_tokens"`
	SummaryInterval  int               `json:"summary_interval"`
	KeepSystemPrompt bool              `json:"keep_system_prompt"`
	PersistPath      string            `json:"persist_path"`
	SemanticMemory   SemanticMemoryCfg `json:"semantic_memory"`
}

// SemanticMemoryCfg contains semantic memory configuration.
type SemanticMemoryCfg struct {
	Enabled             bool    `json:"enabled"`
	VectorStoreType     string  `json:"vector_store_type"`
	SimilarityThreshold float64 `json:"similarity_threshold"`
	MaxResults          int     `json:"max_results"`
}

// RateLimitsCfg contains global rate limit configuration.
type RateLimitsCfg struct {
	Enabled            bool `json:"enabled"`
	RequestsPerMinute  int  `json:"requests_per_minute"`
	TokensPerMinute    int  `json:"tokens_per_minute"`
	ConcurrentRequests int  `json:"concurrent_requests"`
	QueueSize          int  `json:"queue_size"`
	BurstSize          int  `json:"burst_size"`
}

// LoggingCfg contains logging configuration.
type LoggingCfg struct {
	Level          string   `json:"level"`
	Format         string   `json:"format"`
	Output         string   `json:"output"`
	LogRequests    bool     `json:"log_requests"`
	LogResponses   bool     `json:"log_responses"`
	LogTokens      bool     `json:"log_tokens"`
	LogCosts       bool     `json:"log_costs"`
	Redact         bool     `json:"redact"`
	RedactPatterns []string `json:"redact_patterns"`
}

// SecurityCfg contains security configuration.
type SecurityCfg struct {
	ContentFiltering bool     `json:"content_filtering"`
	MaxInputLength   int      `json:"max_input_length"`
	MaxOutputLength  int      `json:"max_output_length"`
	BlockedTopics    []string `json:"blocked_topics"`
	AllowedDomains   []string `json:"allowed_domains"`
	SanitizeHTML     bool     `json:"sanitize_html"`
	ValidateJSON     bool     `json:"validate_json"`
	PIIDetection     bool     `json:"pii_detection"`
	PIIRedaction     bool     `json:"pii_redaction"`
}

// FeaturesCfg contains feature flags.
type FeaturesCfg struct {
	EnableRAG          bool `json:"enable_rag"`
	EnableTools        bool `json:"enable_tools"`
	EnableStreaming    bool `json:"enable_streaming"`
	EnableCaching      bool `json:"enable_caching"`
	EnableCostTracking bool `json:"enable_cost_tracking"`
	AutoPreprocess     bool `json:"auto_preprocess"`
}

// AgentCfg contains agent configuration.
type AgentCfg struct {
	Name         string `json:"name"`
	SystemPrompt string `json:"system_prompt"`
}

// ------------------------------
// Prompts Configuration
// ------------------------------

// PromptsConfig contains all prompt templates.
type PromptsConfig struct {
	Prompts map[string]*PromptCfg `json:"prompts"`
}

// PromptCfg represents a prompt template configuration.
type PromptCfg struct {
	ID          string              `json:"id"`
	Name        string              `json:"name"`
	Description string              `json:"description"`
	Version     string              `json:"version"`
	Category    string              `json:"category"`
	Template    string              `json:"template"`
	Variables   []PromptVariableCfg `json:"variables"`
	Config      *PromptSettingsCfg  `json:"config,omitempty"`
}

// PromptVariableCfg represents a template variable.
type PromptVariableCfg struct {
	Name        string   `json:"name"`
	Type        string   `json:"type"`
	Required    bool     `json:"required"`
	Default     string   `json:"default,omitempty"`
	Description string   `json:"description"`
	Enum        []string `json:"enum,omitempty"`
}

// PromptSettingsCfg contains prompt-specific settings.
type PromptSettingsCfg struct {
	PreferredProvider string  `json:"preferred_provider,omitempty"`
	PreferredModel    string  `json:"preferred_model,omitempty"`
	Temperature       float64 `json:"temperature,omitempty"`
	MaxTokens         int     `json:"max_tokens,omitempty"`
	OutputFormat      string  `json:"output_format,omitempty"`
}

// ------------------------------
// Tools Configuration
// ------------------------------

// ToolsConfig contains all tool configurations.
type ToolsConfig struct {
	Tools        map[string]*ToolCfg `json:"tools"`
	ToolSettings ToolSettingsCfg     `json:"tool_settings"`
}

// ToolCfg represents a tool configuration.
type ToolCfg struct {
	Name        string             `json:"name"`
	Description string             `json:"description"`
	Enabled     bool               `json:"enabled"`
	Parameters  *ToolParametersCfg `json:"parameters"`
	Handler     string             `json:"handler,omitempty"`
	Endpoint    *ToolEndpointCfg   `json:"endpoint,omitempty"`
	Security    *ToolSecurityCfg   `json:"security,omitempty"`
	Sandbox     *ToolSandboxCfg    `json:"sandbox,omitempty"`
	Provider    string             `json:"provider,omitempty"`
	Model       string             `json:"model,omitempty"`
}

// ToolParametersCfg represents tool parameters schema.
type ToolParametersCfg struct {
	Type       string                      `json:"type"`
	Properties map[string]*ToolPropertyCfg `json:"properties,omitempty"`
	Required   []string                    `json:"required,omitempty"`
}

// ToolPropertyCfg represents a tool property schema.
type ToolPropertyCfg struct {
	Type        string   `json:"type"`
	Description string   `json:"description,omitempty"`
	Enum        []string `json:"enum,omitempty"`
	Default     any      `json:"default,omitempty"`
	Minimum     *float64 `json:"minimum,omitempty"`
	Maximum     *float64 `json:"maximum,omitempty"`
}

// ToolEndpointCfg represents HTTP endpoint configuration.
type ToolEndpointCfg struct {
	Type    string            `json:"type"`
	URL     string            `json:"url"`
	Method  string            `json:"method"`
	Headers map[string]string `json:"headers,omitempty"`
}

// ToolSecurityCfg represents tool security settings.
type ToolSecurityCfg struct {
	AllowedPaths      []string `json:"allowed_paths,omitempty"`
	BlockedExtensions []string `json:"blocked_extensions,omitempty"`
	AllowedDomains    []string `json:"allowed_domains,omitempty"`
	BlockedDomains    []string `json:"blocked_domains,omitempty"`
	ReadOnly          bool     `json:"read_only,omitempty"`
	MaxRows           int      `json:"max_rows,omitempty"`
	MaxFileSizeMB     int      `json:"max_file_size_mb,omitempty"`
	AllowedDatabases  []string `json:"allowed_databases,omitempty"`
}

// ToolSandboxCfg represents sandbox settings for code execution.
type ToolSandboxCfg struct {
	Enabled       bool `json:"enabled"`
	MemoryLimitMB int  `json:"memory_limit_mb"`
	NetworkAccess bool `json:"network_access"`
}

// ToolSettingsCfg contains global tool settings.
type ToolSettingsCfg struct {
	MaxParallelCalls   int  `json:"max_parallel_calls"`
	DefaultTimeout     int  `json:"default_timeout"`
	RetryOnError       bool `json:"retry_on_error"`
	MaxRetries         int  `json:"max_retries"`
	LogToolCalls       bool `json:"log_tool_calls"`
	ValidateParameters bool `json:"validate_parameters"`
}

// ------------------------------
// Domains Configuration
// ------------------------------

// DomainsConfig contains domain configurations.
type DomainsConfig struct {
	Domains        map[string]*DomainCfg `json:"domains"`
	DomainSettings DomainSettingsCfg     `json:"domain_settings"`
}

// DomainCfg represents a domain configuration.
type DomainCfg struct {
	ID                    string            `json:"id"`
	Name                  string            `json:"name"`
	Description           string            `json:"description"`
	Enabled               bool              `json:"enabled"`
	Terminology           map[string]string `json:"terminology"`
	Guidelines            []string          `json:"guidelines"`
	PreferredModel        string            `json:"preferred_model,omitempty"`
	SystemPrompt          string            `json:"system_prompt,omitempty"`
	SystemPromptExtension string            `json:"system_prompt_extension,omitempty"`
}

// DomainSettingsCfg contains domain settings.
type DomainSettingsCfg struct {
	StoragePath         string  `json:"storage_path"`
	ChunkSize           int     `json:"chunk_size"`
	ChunkOverlap        int     `json:"chunk_overlap"`
	EmbeddingModel      string  `json:"embedding_model"`
	SimilarityThreshold float64 `json:"similarity_threshold"`
	MaxContextDocuments int     `json:"max_context_documents"`
	AutoDetectDomain    bool    `json:"auto_detect_domain"`
}

// ------------------------------
// Configuration Loader
// ------------------------------

// ConfigLoader handles loading and managing configuration files.
type ConfigLoader struct {
	basePath    string
	config      *Config
	mu          sync.RWMutex
	watcherStop chan struct{}
}

// NewConfigLoader creates a new configuration loader.
func NewConfigLoader(basePath string) *ConfigLoader {
	return &ConfigLoader{
		basePath: basePath,
	}
}

// Load loads all configuration files from the base path.
func (cl *ConfigLoader) Load() (*Config, error) {
	cl.mu.Lock()
	defer cl.mu.Unlock()

	config := &Config{
		configPath: cl.basePath,
		loadedAt:   time.Now(),
	}

	// Load main config
	mainConfigPath := filepath.Join(cl.basePath, "config.json")
	if err := cl.loadJSONFile(mainConfigPath, config); err != nil {
		return nil, fmt.Errorf("failed to load main config: %w", err)
	}

	// Load prompts config
	promptsPath := filepath.Join(cl.basePath, "prompts.json")
	if _, err := os.Stat(promptsPath); err == nil {
		config.Prompts = &PromptsConfig{}
		if err := cl.loadJSONFile(promptsPath, config.Prompts); err != nil {
			return nil, fmt.Errorf("failed to load prompts config: %w", err)
		}
	}

	// Load tools config
	toolsPath := filepath.Join(cl.basePath, "tools.json")
	if _, err := os.Stat(toolsPath); err == nil {
		config.Tools = &ToolsConfig{}
		if err := cl.loadJSONFile(toolsPath, config.Tools); err != nil {
			return nil, fmt.Errorf("failed to load tools config: %w", err)
		}
	}

	// Load domains config
	domainsPath := filepath.Join(cl.basePath, "domains.json")
	if _, err := os.Stat(domainsPath); err == nil {
		config.Domains = &DomainsConfig{}
		if err := cl.loadJSONFile(domainsPath, config.Domains); err != nil {
			return nil, fmt.Errorf("failed to load domains config: %w", err)
		}
	}

	// Resolve environment variables
	config.resolveEnvVars()

	cl.config = config
	return config, nil
}

// loadJSONFile loads a JSON file into the target struct.
func (cl *ConfigLoader) loadJSONFile(path string, target any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read file %s: %w", path, err)
	}

	if err := json.Unmarshal(data, target); err != nil {
		return fmt.Errorf("failed to parse JSON from %s: %w", path, err)
	}

	return nil
}

// resolveEnvVars resolves environment variable references in the configuration.
func (c *Config) resolveEnvVars() {
	for _, provider := range c.Providers {
		// Resolve API key from environment
		if provider.APIKeyEnv != "" {
			// The actual API key is retrieved at runtime using GetAPIKey()
		}

		// Resolve Vertex AI project ID
		if provider.VertexAI != nil && provider.VertexAI.ProjectIDEnv != "" {
			// The actual project ID is retrieved at runtime
		}

		// Resolve environment variables in headers
		for key, value := range provider.Headers {
			provider.Headers[key] = resolveEnvVar(value)
		}

		// Resolve base URL if it contains env vars
		provider.BaseURL = resolveEnvVar(provider.BaseURL)
	}
}

// resolveEnvVar resolves ${VAR_NAME} patterns in a string.
func resolveEnvVar(s string) string {
	if !strings.Contains(s, "${") {
		return s
	}

	result := s
	for {
		start := strings.Index(result, "${")
		if start == -1 {
			break
		}
		end := strings.Index(result[start:], "}")
		if end == -1 {
			break
		}
		end += start

		varName := result[start+2 : end]
		varValue := os.Getenv(varName)
		result = result[:start] + varValue + result[end+1:]
	}

	return result
}

// GetConfig returns the current configuration.
func (cl *ConfigLoader) GetConfig() *Config {
	cl.mu.RLock()
	defer cl.mu.RUnlock()
	return cl.config
}

// Reload reloads the configuration from disk.
func (cl *ConfigLoader) Reload() error {
	_, err := cl.Load()
	return err
}

// ------------------------------
// Configuration Access Methods
// ------------------------------

// GetAPIKey returns the API key for a provider.
func (c *Config) GetAPIKey(providerName string) string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	provider, ok := c.Providers[providerName]
	if !ok {
		return ""
	}

	return os.Getenv(provider.APIKeyEnv)
}

// GetProvider returns the configuration for a specific provider.
func (c *Config) GetProvider(name string) (*ProviderCfg, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	provider, ok := c.Providers[name]
	return provider, ok
}

// GetModel returns the configuration for a specific model across all providers.
func (c *Config) GetModel(modelID string) (*ModelCfg, string, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	for providerName, provider := range c.Providers {
		if model, ok := provider.Models[modelID]; ok {
			return model, providerName, true
		}
	}
	return nil, "", false
}

// GetDefaultModel returns the default model for a capability.
func (c *Config) GetDefaultModel(capability string) string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if model, ok := c.Routing.DefaultModels[capability]; ok {
		return model
	}
	return ""
}

// GetDomainRoute returns the preferred model for a domain.
func (c *Config) GetDomainRoute(domain string) string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if model, ok := c.Routing.DomainRoutes[domain]; ok {
		return model
	}
	return ""
}

// GetPrompt returns a prompt template by ID.
func (c *Config) GetPrompt(id string) (*PromptCfg, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.Prompts == nil {
		return nil, false
	}

	prompt, ok := c.Prompts.Prompts[id]
	return prompt, ok
}

// GetTool returns a tool configuration by name.
func (c *Config) GetTool(name string) (*ToolCfg, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.Tools == nil {
		return nil, false
	}

	tool, ok := c.Tools.Tools[name]
	return tool, ok
}

// GetDomain returns a domain configuration by ID.
func (c *Config) GetDomain(id string) (*DomainCfg, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.Domains == nil {
		return nil, false
	}

	domain, ok := c.Domains.Domains[id]
	return domain, ok
}

// GetEnabledProviders returns a list of enabled providers.
func (c *Config) GetEnabledProviders() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var providers []string
	for name, provider := range c.Providers {
		// Include provider if enabled AND (no API key required OR API key present in env)
		if provider.Enabled && (provider.APIKeyEnv == "" || os.Getenv(provider.APIKeyEnv) != "") {
			providers = append(providers, name)
		}
	}
	return providers
}

// GetGenerationConfig returns generation config for a task type.
func (c *Config) GetGenerationConfig(taskType string) GenerationDefaultsCfg {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if taskConfig, ok := c.Generation.TaskSpecific[taskType]; ok {
		// Merge with defaults
		merged := c.Generation.Defaults
		if taskConfig.Temperature != 0 {
			merged.Temperature = taskConfig.Temperature
		}
		if taskConfig.MaxTokens != 0 {
			merged.MaxTokens = taskConfig.MaxTokens
		}
		if taskConfig.TopP != 0 {
			merged.TopP = taskConfig.TopP
		}
		return merged
	}

	return c.Generation.Defaults
}

// Validate validates the configuration.
func (c *Config) Validate() []string {
	var errors []string

	// Check for at least one enabled provider
	hasEnabledProvider := false
	for _, provider := range c.Providers {
		if provider.Enabled {
			hasEnabledProvider = true
			break
		}
	}
	if !hasEnabledProvider {
		errors = append(errors, "no providers are enabled")
	}

	// Check default models exist
	for capability, modelID := range c.Routing.DefaultModels {
		if _, _, found := c.GetModel(modelID); !found {
			errors = append(errors, fmt.Sprintf("default model '%s' for capability '%s' not found", modelID, capability))
		}
	}

	// Validate generation defaults
	if c.Generation.Defaults.Temperature < 0 || c.Generation.Defaults.Temperature > 2 {
		errors = append(errors, "temperature must be between 0 and 2")
	}

	if c.Generation.Defaults.MaxTokens <= 0 {
		errors = append(errors, "max_tokens must be positive")
	}

	return errors
}

// ------------------------------
// Global Configuration Instance
// ------------------------------

var (
	globalConfig     *Config
	globalConfigOnce sync.Once
	globalConfigMu   sync.RWMutex
)

// LoadGlobalConfig loads the global configuration from the specified path.
func LoadGlobalConfig(configPath string) (*Config, error) {
	globalConfigMu.Lock()
	defer globalConfigMu.Unlock()

	loader := NewConfigLoader(configPath)
	config, err := loader.Load()
	if err != nil {
		return nil, err
	}

	globalConfig = config
	return config, nil
}

// GetGlobalConfig returns the global configuration instance.
func GetGlobalConfig() *Config {
	globalConfigMu.RLock()
	defer globalConfigMu.RUnlock()
	return globalConfig
}

// MustLoadConfig loads config and panics on error.
func MustLoadConfig(configPath string) *Config {
	config, err := LoadGlobalConfig(configPath)
	if err != nil {
		panic(fmt.Sprintf("failed to load configuration: %v", err))
	}
	return config
}
