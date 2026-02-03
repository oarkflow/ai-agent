package llm

import (
	"context"

	"github.com/sujit/ai-agent/pkg/content"
)

// Capability represents what a model can do.
type Capability string

const (
	CapText           Capability = "text"
	CapVision         Capability = "vision"
	CapAudio          Capability = "audio"
	CapVideo          Capability = "video"
	CapDocument       Capability = "document"
	CapCodeGeneration Capability = "code_generation"
	CapCodeExecution  Capability = "code_execution"
	CapReasoning      Capability = "reasoning"
	CapFunctionCall   Capability = "function_calling"
	CapStreaming      Capability = "streaming"
	CapEmbedding      Capability = "embedding"
	CapImageGen       Capability = "image_generation"
	CapSpeechGen      Capability = "speech_generation"
	CapTranscription  Capability = "transcription"
	CapFineTuning     Capability = "fine_tuning"
	CapRealtime       Capability = "realtime"
	CapAgents         Capability = "agents"
	CapWebSearch      Capability = "web_search"
	CapFileSearch     Capability = "file_search"
)

// ProviderType identifies the LLM provider.
type ProviderType string

const (
	ProviderOpenAI      ProviderType = "openai"
	ProviderAnthropic   ProviderType = "anthropic"
	ProviderGoogle      ProviderType = "google"
	ProviderMistral     ProviderType = "mistral"
	ProviderCohere      ProviderType = "cohere"
	ProviderDeepSeek    ProviderType = "deepseek"
	ProviderXAI         ProviderType = "xai"
	ProviderMeta        ProviderType = "meta"
	ProviderAzureOpenAI ProviderType = "azure_openai"
	ProviderAWSBedrock  ProviderType = "aws_bedrock"
	ProviderOllama      ProviderType = "ollama"
	ProviderHuggingFace ProviderType = "huggingface"
	ProviderReplicate   ProviderType = "replicate"
	ProviderTogether    ProviderType = "together"
	ProviderGroq        ProviderType = "groq"
	ProviderFireworks   ProviderType = "fireworks"
)

// ModelInfo contains metadata about a model.
type ModelInfo struct {
	ID                string            `json:"id"`
	Name              string            `json:"name"`
	Provider          ProviderType      `json:"provider"`
	Version           string            `json:"version,omitempty"`
	Capabilities      []Capability      `json:"capabilities"`
	ContextWindow     int               `json:"context_window"`
	MaxOutputTokens   int               `json:"max_output_tokens"`
	InputCostPer1K    float64           `json:"input_cost_per_1k"`
	OutputCostPer1K   float64           `json:"output_cost_per_1k"`
	SupportedFormats  []content.MediaType `json:"supported_formats,omitempty"`
	Deprecated        bool              `json:"deprecated,omitempty"`
	ReplacedBy        string            `json:"replaced_by,omitempty"`
	ReleaseDate       string            `json:"release_date,omitempty"`
	Metadata          map[string]any    `json:"metadata,omitempty"`
}

// HasCapability checks if a model has a specific capability.
func (m *ModelInfo) HasCapability(cap Capability) bool {
	for _, c := range m.Capabilities {
		if c == cap {
			return true
		}
	}
	return false
}

// SupportsFormat checks if a model supports a specific media format.
func (m *ModelInfo) SupportsFormat(format content.MediaType) bool {
	for _, f := range m.SupportedFormats {
		if f == format {
			return true
		}
	}
	return false
}

// GenerationConfig holds configuration for text generation.
type GenerationConfig struct {
	Model           string            `json:"model,omitempty"`
	Temperature     float64           `json:"temperature,omitempty"`
	TopP            float64           `json:"top_p,omitempty"`
	TopK            int               `json:"top_k,omitempty"`
	MaxTokens       int               `json:"max_tokens,omitempty"`
	StopSequences   []string          `json:"stop_sequences,omitempty"`
	PresencePenalty float64           `json:"presence_penalty,omitempty"`
	FrequencyPenalty float64          `json:"frequency_penalty,omitempty"`
	Seed            *int              `json:"seed,omitempty"`
	ResponseFormat  *ResponseFormat   `json:"response_format,omitempty"`
	Tools           []Tool            `json:"tools,omitempty"`
	ToolChoice      any               `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool           `json:"parallel_tool_calls,omitempty"`
	SystemPrompt    string            `json:"system_prompt,omitempty"`
	Stream          bool              `json:"stream,omitempty"`
	Timeout         int               `json:"timeout,omitempty"` // Seconds
	RetryAttempts   int               `json:"retry_attempts,omitempty"`
	CacheControl    *CacheControl     `json:"cache_control,omitempty"`
	SafetySettings  []SafetySetting   `json:"safety_settings,omitempty"`
	Metadata        map[string]any    `json:"metadata,omitempty"`
}

// ResponseFormat specifies the output format.
type ResponseFormat struct {
	Type       string         `json:"type"` // "text", "json_object", "json_schema"
	JSONSchema *JSONSchema    `json:"json_schema,omitempty"`
}

// JSONSchema defines the schema for structured outputs.
type JSONSchema struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Schema      map[string]any `json:"schema"`
	Strict      bool           `json:"strict,omitempty"`
}

// Tool represents a tool/function that can be called by the model.
type Tool struct {
	Type     string   `json:"type"` // "function"
	Function Function `json:"function"`
}

// Function represents a callable function.
type Function struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"` // JSON Schema
	Strict      bool           `json:"strict,omitempty"`
}

// CacheControl for prompt caching (Anthropic).
type CacheControl struct {
	Type string `json:"type"` // "ephemeral"
}

// SafetySetting for content safety (Google).
type SafetySetting struct {
	Category  string `json:"category"`
	Threshold string `json:"threshold"`
}

// GenerationResponse represents the response from generation.
type GenerationResponse struct {
	ID              string                  `json:"id,omitempty"`
	Model           string                  `json:"model"`
	Message         *content.Message        `json:"message"`
	FinishReason    string                  `json:"finish_reason,omitempty"`
	Usage           *Usage                  `json:"usage,omitempty"`
	ToolCalls       []content.ToolCall      `json:"tool_calls,omitempty"`
	Citations       []Citation              `json:"citations,omitempty"`
	Metadata        map[string]any          `json:"metadata,omitempty"`
}

// Usage contains token usage information.
type Usage struct {
	InputTokens      int     `json:"input_tokens"`
	OutputTokens     int     `json:"output_tokens"`
	TotalTokens      int     `json:"total_tokens"`
	CachedTokens     int     `json:"cached_tokens,omitempty"`
	ReasoningTokens  int     `json:"reasoning_tokens,omitempty"`
	InputCost        float64 `json:"input_cost,omitempty"`
	OutputCost       float64 `json:"output_cost,omitempty"`
	TotalCost        float64 `json:"total_cost,omitempty"`
}

// Citation for grounded responses.
type Citation struct {
	StartIndex int    `json:"start_index"`
	EndIndex   int    `json:"end_index"`
	URL        string `json:"url,omitempty"`
	Title      string `json:"title,omitempty"`
	License    string `json:"license,omitempty"`
}

// StreamChunk represents a chunk in a streaming response.
type StreamChunk struct {
	ID           string              `json:"id,omitempty"`
	Delta        string              `json:"delta"`
	FinishReason string              `json:"finish_reason,omitempty"`
	Usage        *Usage              `json:"usage,omitempty"`
	ToolCall     *content.ToolCall   `json:"tool_call,omitempty"`
	Error        error               `json:"error,omitempty"`
}

// EmbeddingRequest represents an embedding request.
type EmbeddingRequest struct {
	Input          []string `json:"input"`
	Model          string   `json:"model"`
	EncodingFormat string   `json:"encoding_format,omitempty"` // "float" or "base64"
	Dimensions     int      `json:"dimensions,omitempty"`
}

// EmbeddingResponse represents embedding results.
type EmbeddingResponse struct {
	Embeddings [][]float64 `json:"embeddings"`
	Model      string      `json:"model"`
	Usage      *Usage      `json:"usage,omitempty"`
}

// TranscriptionRequest for audio transcription.
type TranscriptionRequest struct {
	Audio         *content.Content `json:"audio"`
	Model         string           `json:"model,omitempty"`
	Language      string           `json:"language,omitempty"`
	Prompt        string           `json:"prompt,omitempty"`
	ResponseFormat string          `json:"response_format,omitempty"` // "json", "text", "srt", "vtt"
	Temperature   float64          `json:"temperature,omitempty"`
	Timestamps    bool             `json:"timestamps,omitempty"`
}

// TranscriptionResponse contains transcription results.
type TranscriptionResponse struct {
	Text     string              `json:"text"`
	Language string              `json:"language,omitempty"`
	Duration float64             `json:"duration,omitempty"`
	Segments []TranscriptSegment `json:"segments,omitempty"`
	Words    []TranscriptWord    `json:"words,omitempty"`
}

// TranscriptSegment represents a segment of transcribed audio.
type TranscriptSegment struct {
	ID        int     `json:"id"`
	Start     float64 `json:"start"`
	End       float64 `json:"end"`
	Text      string  `json:"text"`
	Tokens    []int   `json:"tokens,omitempty"`
	AvgLogProb float64 `json:"avg_logprob,omitempty"`
}

// TranscriptWord represents a word with timing.
type TranscriptWord struct {
	Word  string  `json:"word"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}

// ImageGenerationRequest for image generation.
type ImageGenerationRequest struct {
	Prompt         string `json:"prompt"`
	Model          string `json:"model,omitempty"`
	N              int    `json:"n,omitempty"`
	Size           string `json:"size,omitempty"`
	Quality        string `json:"quality,omitempty"`
	Style          string `json:"style,omitempty"`
	ResponseFormat string `json:"response_format,omitempty"` // "url" or "b64_json"
}

// ImageGenerationResponse contains generated images.
type ImageGenerationResponse struct {
	Images  []*content.Content `json:"images"`
	Model   string             `json:"model"`
	Usage   *Usage             `json:"usage,omitempty"`
}

// SpeechRequest for text-to-speech.
type SpeechRequest struct {
	Input          string  `json:"input"`
	Model          string  `json:"model,omitempty"`
	Voice          string  `json:"voice"`
	ResponseFormat string  `json:"response_format,omitempty"` // "mp3", "opus", "aac", "flac"
	Speed          float64 `json:"speed,omitempty"`
}

// SpeechResponse contains generated speech.
type SpeechResponse struct {
	Audio *content.Content `json:"audio"`
	Model string           `json:"model"`
}

// MultimodalProvider is the enhanced provider interface for multimodal AI.
type MultimodalProvider interface {
	// Core text generation
	Generate(ctx context.Context, messages []*content.Message, config *GenerationConfig) (*GenerationResponse, error)

	// Streaming generation
	GenerateStream(ctx context.Context, messages []*content.Message, config *GenerationConfig) (<-chan StreamChunk, error)

	// Embeddings
	Embed(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error)

	// Audio transcription
	Transcribe(ctx context.Context, req *TranscriptionRequest) (*TranscriptionResponse, error)

	// Image generation
	GenerateImage(ctx context.Context, req *ImageGenerationRequest) (*ImageGenerationResponse, error)

	// Text-to-speech
	GenerateSpeech(ctx context.Context, req *SpeechRequest) (*SpeechResponse, error)

	// Provider info
	GetProviderType() ProviderType
	GetModelInfo(model string) (*ModelInfo, error)
	ListModels(ctx context.Context) ([]ModelInfo, error)

	// Capabilities
	GetCapabilities() []Capability
	SupportsCapability(cap Capability) bool
}

// BaseProvider provides common functionality for providers.
type BaseProvider struct {
	ProviderType ProviderType
	APIKey       string
	BaseURL      string
	DefaultModel string
	Models       map[string]*ModelInfo
}

// GetProviderType returns the provider type.
func (b *BaseProvider) GetProviderType() ProviderType {
	return b.ProviderType
}

// GetModelInfo returns info about a specific model.
func (b *BaseProvider) GetModelInfo(model string) (*ModelInfo, error) {
	if info, ok := b.Models[model]; ok {
		return info, nil
	}
	return nil, ErrModelNotFound
}

// Common errors
var (
	ErrModelNotFound    = NewLLMError("model not found", "")
	ErrCapabilityNotSupported = NewLLMError("capability not supported", "")
	ErrInvalidInput     = NewLLMError("invalid input", "")
	ErrRateLimited      = NewLLMError("rate limited", "")
	ErrContextTooLong   = NewLLMError("context too long", "")
	ErrContentFiltered  = NewLLMError("content filtered", "")
)

// LLMError represents an error from an LLM provider.
type LLMError struct {
	Message    string `json:"message"`
	Code       string `json:"code,omitempty"`
	StatusCode int    `json:"status_code,omitempty"`
	Retryable  bool   `json:"retryable,omitempty"`
}

func (e *LLMError) Error() string {
	return e.Message
}

// NewLLMError creates a new LLM error.
func NewLLMError(message, code string) *LLMError {
	return &LLMError{
		Message: message,
		Code:    code,
	}
}
