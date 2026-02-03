package llm

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/sujit/ai-agent/pkg/content"
)

// ProviderRegistry manages multiple AI providers and routes requests.
type ProviderRegistry struct {
	mu        sync.RWMutex
	providers map[ProviderType]MultimodalProvider
	models    map[string]*RegisteredModel
	config    *RoutingConfig
}

// RegisteredModel contains model info with its provider.
type RegisteredModel struct {
	Info     *ModelInfo
	Provider MultimodalProvider
}

// RoutingConfig configures the routing behavior.
type RoutingConfig struct {
	// Default models for different use cases
	DefaultTextModel       string
	DefaultVisionModel     string
	DefaultAudioModel      string
	DefaultVideoModel      string
	DefaultDocumentModel   string
	DefaultCodeModel       string
	DefaultReasoningModel  string
	DefaultEmbeddingModel  string
	DefaultImageGenModel   string
	DefaultSpeechModel     string
	DefaultTranscribeModel string

	// Cost optimization
	PreferCheaperModels bool
	MaxCostPer1KTokens  float64

	// Capability preferences
	PreferredProviders []ProviderType
	FallbackProviders  []ProviderType

	// Domain-specific routing
	DomainRoutes map[string]string // domain -> model
}

// DefaultRoutingConfig returns a sensible default configuration.
// NOTE: These defaults are only used when not loading from config/config.json.
// When using JSON configuration, use RoutingConfigFromConfig() in factory.go instead.
func DefaultRoutingConfig() *RoutingConfig {
	return &RoutingConfig{
		// These are backward-compatible defaults; override via config/config.json
		DefaultTextModel:       "", // Will be set from config
		DefaultVisionModel:     "", // Will be set from config
		DefaultAudioModel:      "", // Will be set from config
		DefaultVideoModel:      "", // Will be set from config
		DefaultDocumentModel:   "", // Will be set from config
		DefaultCodeModel:       "", // Will be set from config
		DefaultReasoningModel:  "", // Will be set from config
		DefaultEmbeddingModel:  "", // Will be set from config
		DefaultImageGenModel:   "", // Will be set from config
		DefaultSpeechModel:     "", // Will be set from config
		DefaultTranscribeModel: "", // Will be set from config
		PreferCheaperModels:    false,
		DomainRoutes:           make(map[string]string),
	}
}

// NewProviderRegistry creates a new provider registry.
func NewProviderRegistry(config *RoutingConfig) *ProviderRegistry {
	if config == nil {
		config = DefaultRoutingConfig()
	}
	return &ProviderRegistry{
		providers: make(map[ProviderType]MultimodalProvider),
		models:    make(map[string]*RegisteredModel),
		config:    config,
	}
}

// RegisterProvider adds a provider to the registry.
func (r *ProviderRegistry) RegisterProvider(provider MultimodalProvider) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	providerType := provider.GetProviderType()
	r.providers[providerType] = provider

	// Register all models from this provider
	models, err := provider.ListModels(context.Background())
	if err != nil {
		return fmt.Errorf("failed to list models from %s: %w", providerType, err)
	}

	for i := range models {
		model := &models[i]
		r.models[model.ID] = &RegisteredModel{
			Info:     model,
			Provider: provider,
		}
	}

	return nil
}

// GetProvider returns a specific provider.
func (r *ProviderRegistry) GetProvider(providerType ProviderType) (MultimodalProvider, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	p, ok := r.providers[providerType]
	return p, ok
}

// GetModel returns a specific model and its provider.
func (r *ProviderRegistry) GetModel(modelID string) (*RegisteredModel, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	m, ok := r.models[modelID]
	return m, ok
}

// SelectModel automatically selects the best model for the given request.
func (r *ProviderRegistry) SelectModel(messages []*content.Message, requirements *ModelRequirements) (*RegisteredModel, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if requirements == nil {
		requirements = &ModelRequirements{}
	}

	// Analyze messages to determine required capabilities
	requiredCaps := r.analyzeRequirements(messages, requirements)

	// Find compatible models
	candidates := r.findCompatibleModels(requiredCaps, requirements)
	if len(candidates) == 0 {
		return nil, fmt.Errorf("no model found with required capabilities: %v", requiredCaps)
	}

	// Score and rank models
	scored := r.scoreModels(candidates, requirements)
	if len(scored) == 0 {
		return nil, fmt.Errorf("no suitable model found")
	}

	// Return best match
	return scored[0].Model, nil
}

// ModelRequirements specifies requirements for model selection.
type ModelRequirements struct {
	Capabilities      []Capability
	MinContextWindow  int
	MaxCostPer1K      float64
	PreferredModel    string
	PreferredProvider ProviderType
	Domain            string // For domain-specific routing
	TaskType          TaskType
	Speed             SpeedPreference
}

// TaskType represents the type of task.
type TaskType string

const (
	TaskGeneral      TaskType = "general"
	TaskCoding       TaskType = "coding"
	TaskReasoning    TaskType = "reasoning"
	TaskCreative     TaskType = "creative"
	TaskAnalysis     TaskType = "analysis"
	TaskSummary      TaskType = "summary"
	TaskTranslation  TaskType = "translation"
	TaskConversation TaskType = "conversation"
)

// SpeedPreference for latency vs quality tradeoff.
type SpeedPreference string

const (
	SpeedFast     SpeedPreference = "fast"
	SpeedBalanced SpeedPreference = "balanced"
	SpeedQuality  SpeedPreference = "quality"
)

// ScoredModel represents a model with its score.
type ScoredModel struct {
	Model *RegisteredModel
	Score float64
}

// analyzeRequirements determines required capabilities from messages.
func (r *ProviderRegistry) analyzeRequirements(messages []*content.Message, req *ModelRequirements) []Capability {
	caps := make(map[Capability]bool)

	// Always need text capability
	caps[CapText] = true

	// Add explicitly requested capabilities
	for _, c := range req.Capabilities {
		caps[c] = true
	}

	// Analyze message content
	for _, msg := range messages {
		for _, c := range msg.Contents {
			switch c.Type {
			case content.TypeImage:
				caps[CapVision] = true
			case content.TypeAudio:
				caps[CapAudio] = true
			case content.TypeVideo:
				caps[CapVideo] = true
			case content.TypeDocument:
				caps[CapDocument] = true
				if c.MediaType == content.MediaPDF {
					caps[CapVision] = true // PDF often needs vision
				}
			case content.TypeCode:
				caps[CapCodeGeneration] = true
			}
		}
	}

	// Add task-specific capabilities
	switch req.TaskType {
	case TaskCoding:
		caps[CapCodeGeneration] = true
	case TaskReasoning:
		caps[CapReasoning] = true
	}

	// Convert to slice
	result := make([]Capability, 0, len(caps))
	for c := range caps {
		result = append(result, c)
	}
	return result
}

// findCompatibleModels finds models with required capabilities.
func (r *ProviderRegistry) findCompatibleModels(requiredCaps []Capability, req *ModelRequirements) []*RegisteredModel {
	var candidates []*RegisteredModel

	for _, rm := range r.models {
		// Check if model has all required capabilities
		hasAll := true
		for _, reqCap := range requiredCaps {
			if !rm.Info.HasCapability(reqCap) {
				hasAll = false
				break
			}
		}
		if !hasAll {
			continue
		}

		// Check context window
		if req.MinContextWindow > 0 && rm.Info.ContextWindow < req.MinContextWindow {
			continue
		}

		// Check cost
		if req.MaxCostPer1K > 0 && rm.Info.InputCostPer1K > req.MaxCostPer1K {
			continue
		}

		// Check deprecated
		if rm.Info.Deprecated {
			continue
		}

		candidates = append(candidates, rm)
	}

	return candidates
}

// scoreModels scores and ranks compatible models.
func (r *ProviderRegistry) scoreModels(candidates []*RegisteredModel, req *ModelRequirements) []ScoredModel {
	scored := make([]ScoredModel, 0, len(candidates))

	for _, model := range candidates {
		score := 100.0

		// Preferred model bonus
		if req.PreferredModel != "" && model.Info.ID == req.PreferredModel {
			score += 1000
		}

		// Preferred provider bonus
		if req.PreferredProvider != "" && model.Info.Provider == req.PreferredProvider {
			score += 50
		}

		// Domain routing
		if req.Domain != "" {
			if domainModel, ok := r.config.DomainRoutes[req.Domain]; ok {
				if model.Info.ID == domainModel {
					score += 500
				}
			}
		}

		// Cost optimization
		if r.config.PreferCheaperModels {
			// Lower cost = higher score (inverted relationship)
			if model.Info.InputCostPer1K > 0 {
				score += 10 / model.Info.InputCostPer1K
			}
		}

		// Context window (larger is better)
		score += float64(model.Info.ContextWindow) / 100000

		// Task-specific scoring
		switch req.TaskType {
		case TaskCoding:
			if model.Info.HasCapability(CapCodeGeneration) {
				score += 30
			}
			// Claude and specialized coding models get bonus
			if strings.Contains(strings.ToLower(model.Info.Name), "claude") ||
				strings.Contains(strings.ToLower(model.Info.Name), "sonnet") {
				score += 20
			}
		case TaskReasoning:
			if model.Info.HasCapability(CapReasoning) {
				score += 50
			}
			// o1, o3 models get bonus
			if strings.HasPrefix(model.Info.ID, "o1") || strings.HasPrefix(model.Info.ID, "o3") {
				score += 30
			}
		}

		// Speed preference
		switch req.Speed {
		case SpeedFast:
			// Prefer smaller/faster models
			if strings.Contains(strings.ToLower(model.Info.Name), "mini") ||
				strings.Contains(strings.ToLower(model.Info.Name), "flash") ||
				strings.Contains(strings.ToLower(model.Info.Name), "haiku") {
				score += 40
			}
		case SpeedQuality:
			// Prefer larger/smarter models
			if strings.Contains(strings.ToLower(model.Info.Name), "opus") ||
				strings.Contains(strings.ToLower(model.Info.Name), "pro") ||
				strings.Contains(model.Info.ID, "4.5") {
				score += 40
			}
		}

		// Multimodal capability scoring
		multimodalCaps := 0
		for _, cap := range []Capability{CapVision, CapAudio, CapVideo, CapDocument} {
			if model.Info.HasCapability(cap) {
				multimodalCaps++
			}
		}
		score += float64(multimodalCaps) * 5

		scored = append(scored, ScoredModel{Model: model, Score: score})
	}

	// Sort by score descending
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].Score > scored[j].Score
	})

	return scored
}

// SmartRouter provides high-level routing for multimodal requests.
type SmartRouter struct {
	Registry *ProviderRegistry
}

// NewSmartRouter creates a new smart router.
func NewSmartRouter(registry *ProviderRegistry) *SmartRouter {
	return &SmartRouter{Registry: registry}
}

// Route selects the best model and generates a response.
func (sr *SmartRouter) Route(ctx context.Context, messages []*content.Message, config *GenerationConfig, requirements *ModelRequirements) (*GenerationResponse, error) {
	// Select best model
	model, err := sr.Registry.SelectModel(messages, requirements)
	if err != nil {
		return nil, fmt.Errorf("model selection failed: %w", err)
	}

	// Override model in config
	if config == nil {
		config = &GenerationConfig{}
	}
	config.Model = model.Info.ID

	// Generate response
	return model.Provider.Generate(ctx, messages, config)
}

// RouteStream selects the best model and streams a response.
func (sr *SmartRouter) RouteStream(ctx context.Context, messages []*content.Message, config *GenerationConfig, requirements *ModelRequirements) (<-chan StreamChunk, *RegisteredModel, error) {
	model, err := sr.Registry.SelectModel(messages, requirements)
	if err != nil {
		return nil, nil, fmt.Errorf("model selection failed: %w", err)
	}

	if config == nil {
		config = &GenerationConfig{}
	}
	config.Model = model.Info.ID

	ch, err := model.Provider.GenerateStream(ctx, messages, config)
	return ch, model, err
}

// RouteEmbed routes an embedding request to the best provider.
func (sr *SmartRouter) RouteEmbed(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
	// Find a provider with embedding capability
	for _, p := range sr.Registry.providers {
		if p.SupportsCapability(CapEmbedding) {
			return p.Embed(ctx, req)
		}
	}
	return nil, ErrCapabilityNotSupported
}

// RouteTranscribe routes a transcription request.
func (sr *SmartRouter) RouteTranscribe(ctx context.Context, req *TranscriptionRequest) (*TranscriptionResponse, error) {
	// Prefer Whisper for transcription
	if openai, ok := sr.Registry.GetProvider(ProviderOpenAI); ok {
		return openai.Transcribe(ctx, req)
	}
	// Fallback to Gemini
	if google, ok := sr.Registry.GetProvider(ProviderGoogle); ok {
		return google.Transcribe(ctx, req)
	}
	return nil, ErrCapabilityNotSupported
}

// RouteImageGen routes an image generation request.
func (sr *SmartRouter) RouteImageGen(ctx context.Context, req *ImageGenerationRequest) (*ImageGenerationResponse, error) {
	// Prefer DALL-E 3
	if openai, ok := sr.Registry.GetProvider(ProviderOpenAI); ok {
		return openai.GenerateImage(ctx, req)
	}
	// Fallback to Imagen
	if google, ok := sr.Registry.GetProvider(ProviderGoogle); ok {
		return google.GenerateImage(ctx, req)
	}
	return nil, ErrCapabilityNotSupported
}

// RouteSpeech routes a text-to-speech request.
func (sr *SmartRouter) RouteSpeech(ctx context.Context, req *SpeechRequest) (*SpeechResponse, error) {
	if openai, ok := sr.Registry.GetProvider(ProviderOpenAI); ok {
		return openai.GenerateSpeech(ctx, req)
	}
	return nil, ErrCapabilityNotSupported
}

// AutoDetectAndRoute automatically detects content type and routes appropriately.
func (sr *SmartRouter) AutoDetectAndRoute(ctx context.Context, input any, systemPrompt string) (*GenerationResponse, error) {
	var messages []*content.Message

	if systemPrompt != "" {
		messages = append(messages, content.NewSystemMessage(systemPrompt))
	}

	switch v := input.(type) {
	case string:
		messages = append(messages, content.NewUserMessage(v))
	case *content.Message:
		messages = append(messages, v)
	case []*content.Message:
		messages = append(messages, v...)
	case *content.Content:
		messages = append(messages, content.NewMultimodalMessage(content.RoleUser, v))
	case []*content.Content:
		messages = append(messages, content.NewMultimodalMessage(content.RoleUser, v...))
	default:
		return nil, fmt.Errorf("unsupported input type: %T", input)
	}

	// Analyze and route
	requirements := &ModelRequirements{}

	// Detect task type from content
	for _, msg := range messages {
		text := msg.GetText()
		if containsCodeIndicators(text) {
			requirements.TaskType = TaskCoding
			break
		}
		if containsReasoningIndicators(text) {
			requirements.TaskType = TaskReasoning
			break
		}
	}

	return sr.Route(ctx, messages, nil, requirements)
}

// Helper functions

func containsCodeIndicators(text string) bool {
	indicators := []string{
		"```", "func ", "def ", "class ", "function ",
		"import ", "package ", "const ", "let ", "var ",
		"debug", "error", "compile", "syntax",
		"implement", "refactor", "optimize code",
	}
	lower := strings.ToLower(text)
	for _, ind := range indicators {
		if strings.Contains(lower, strings.ToLower(ind)) {
			return true
		}
	}
	return false
}

func containsReasoningIndicators(text string) bool {
	indicators := []string{
		"solve", "calculate", "prove", "derive",
		"step by step", "explain why", "reasoning",
		"logical", "mathematically", "analyze",
		"what if", "compare and contrast",
	}
	lower := strings.ToLower(text)
	for _, ind := range indicators {
		if strings.Contains(lower, ind) {
			return true
		}
	}
	return false
}
