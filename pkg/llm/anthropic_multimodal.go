package llm

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/sujit/ai-agent/pkg/content"
)

// ------------------------------
// Provider Definition
// ------------------------------

type AnthropicMultimodalProvider struct {
	BaseProvider
	Client      *http.Client
	APIVersion  string
	BetaHeaders []string
}

type AnthropicOption func(*AnthropicMultimodalProvider)

// NewAnthropicMultimodalProvider creates a new Anthropic provider.
// When using JSON configuration, prefer using ProviderFactory.CreateProvider("anthropic") instead.
func NewAnthropicMultimodalProvider(apiKey string, opts ...AnthropicOption) *AnthropicMultimodalProvider {
	p := &AnthropicMultimodalProvider{
		BaseProvider: BaseProvider{
			ProviderType: ProviderAnthropic,
			APIKey:       apiKey,
			BaseURL:      "",                          // Set via config or WithAnthropicBaseURL option
			DefaultModel: "",                          // Set via config or WithAnthropicModel option
			Models:       make(map[string]*ModelInfo), // Loaded from config
		},
		Client:     &http.Client{Timeout: 120 * time.Second},
		APIVersion: "", // Set via config or WithAnthropicAPIVersion option
	}

	for _, opt := range opts {
		opt(p)
	}

	// Apply defaults only if not set via options (backward compatibility)
	if p.BaseURL == "" {
		p.BaseURL = "https://api.anthropic.com/v1"
	}
	if p.DefaultModel == "" {
		p.DefaultModel = "claude-sonnet-4-20250514"
	}
	if p.APIVersion == "" {
		p.APIVersion = "2023-06-01"
	}

	return p
}

func WithAnthropicBeta(features ...string) AnthropicOption {
	return func(p *AnthropicMultimodalProvider) {
		p.BetaHeaders = append(p.BetaHeaders, features...)
	}
}

func WithAnthropicBaseURL(url string) AnthropicOption {
	return func(p *AnthropicMultimodalProvider) {
		p.BaseURL = url
	}
}

// WithAnthropicModel sets the default model.
func WithAnthropicModel(model string) AnthropicOption {
	return func(p *AnthropicMultimodalProvider) {
		p.DefaultModel = model
	}
}

// WithAnthropicModels sets the available models from config.
func WithAnthropicModels(models map[string]*ModelInfo) AnthropicOption {
	return func(p *AnthropicMultimodalProvider) {
		p.Models = models
	}
}

// WithAnthropicAPIVersion sets the API version.
func WithAnthropicAPIVersion(version string) AnthropicOption {
	return func(p *AnthropicMultimodalProvider) {
		p.APIVersion = version
	}
}

// WithAnthropicTimeout sets the HTTP client timeout.
func WithAnthropicTimeout(timeout time.Duration) AnthropicOption {
	return func(p *AnthropicMultimodalProvider) {
		p.Client = &http.Client{Timeout: timeout}
	}
}

// ------------------------------
// Helper Methods
// ------------------------------

func (p *AnthropicMultimodalProvider) setHeaders(req *http.Request) {
	req.Header.Set("x-api-key", p.APIKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("anthropic-version", p.APIVersion)

	if len(p.BetaHeaders) > 0 {
		for _, beta := range p.BetaHeaders {
			req.Header.Add("anthropic-beta", beta)
		}
	}
}

func (p *AnthropicMultimodalProvider) doRequest(ctx context.Context, method, path string, body any) ([]byte, error) {
	jsonData, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, method, p.BaseURL+path, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	p.setHeaders(req)

	resp, err := p.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	return respBody, nil
}

func ConvertImageToBase64(imagePath string) (string, content.MediaType, error) {
	img, err := content.NewImageFromFile(imagePath)
	if err != nil {
		return "", "", err
	}
	return base64.StdEncoding.EncodeToString(img.Data), img.MediaType, nil
}

// ------------------------------
// Capabilities
// ------------------------------

func (p *AnthropicMultimodalProvider) GetCapabilities() []Capability {
	return []Capability{
		CapText, CapVision, CapDocument, CapCodeGeneration,
		CapFunctionCall, CapStreaming, CapReasoning, CapAgents,
	}
}

func (p *AnthropicMultimodalProvider) SupportsCapability(cap Capability) bool {
	for _, c := range p.GetCapabilities() {
		if c == cap {
			return true
		}
	}
	return false
}

// ------------------------------
// Not Supported
// ------------------------------

func (p *AnthropicMultimodalProvider) GenerateSpeech(ctx context.Context, req *SpeechRequest) (*SpeechResponse, error) {
	return nil, ErrCapabilityNotSupported
}

func (p *AnthropicMultimodalProvider) GenerateImage(ctx context.Context, req *ImageGenerationRequest) (*ImageGenerationResponse, error) {
	return nil, ErrCapabilityNotSupported
}

func (p *AnthropicMultimodalProvider) Transcribe(ctx context.Context, req *TranscriptionRequest) (*TranscriptionResponse, error) {
	return nil, ErrCapabilityNotSupported
}

func (p *AnthropicMultimodalProvider) Embed(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
	return nil, ErrCapabilityNotSupported
}

// ------------------------------
// Models
// ------------------------------

func (p *AnthropicMultimodalProvider) ListModels(ctx context.Context) ([]ModelInfo, error) {
	// If models are loaded from config, return those
	if len(p.Models) > 0 {
		models := make([]ModelInfo, 0, len(p.Models))
		for _, m := range p.Models {
			models = append(models, *m)
		}
		return models, nil
	}
	// Return empty if no models configured (use config/config.json to define models)
	return []ModelInfo{}, nil
}

// ------------------------------
// Core Generation Methods
// ------------------------------

// Generate performs text generation with multimodal input support.
func (p *AnthropicMultimodalProvider) Generate(ctx context.Context, messages []*content.Message, config *GenerationConfig) (*GenerationResponse, error) {
	model := p.DefaultModel
	if config != nil && config.Model != "" {
		model = config.Model
	}

	// Convert messages, extracting system prompt
	var systemPrompt string
	var convertedMessages []map[string]any

	for _, msg := range messages {
		if msg.Role == content.RoleSystem {
			systemPrompt = msg.GetText()
			continue
		}
		convertedMessages = append(convertedMessages, p.convertMessage(msg))
	}

	reqBody := map[string]any{
		"model":      model,
		"messages":   convertedMessages,
		"max_tokens": 4096,
	}

	if systemPrompt != "" {
		reqBody["system"] = systemPrompt
	}

	if config != nil {
		if config.Temperature != 0 {
			reqBody["temperature"] = config.Temperature
		}
		if config.MaxTokens > 0 {
			reqBody["max_tokens"] = config.MaxTokens
		}
		if config.TopP != 0 {
			reqBody["top_p"] = config.TopP
		}
		if len(config.StopSequences) > 0 {
			reqBody["stop_sequences"] = config.StopSequences
		}
	}

	body, err := p.doRequest(ctx, "POST", "/messages", reqBody)
	if err != nil {
		return nil, err
	}

	var resp struct {
		ID      string `json:"id"`
		Model   string `json:"model"`
		Type    string `json:"type"`
		Role    string `json:"role"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text,omitempty"`
		} `json:"content"`
		StopReason string `json:"stop_reason"`
		Usage      struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Extract text content
	var responseText string
	for _, c := range resp.Content {
		if c.Type == "text" {
			responseText += c.Text
		}
	}

	return &GenerationResponse{
		ID:           resp.ID,
		Model:        resp.Model,
		Message:      content.NewTextMessage(content.RoleAssistant, responseText),
		FinishReason: resp.StopReason,
		Usage: &Usage{
			InputTokens:  resp.Usage.InputTokens,
			OutputTokens: resp.Usage.OutputTokens,
			TotalTokens:  resp.Usage.InputTokens + resp.Usage.OutputTokens,
		},
	}, nil
}

// GenerateStream performs streaming text generation.
func (p *AnthropicMultimodalProvider) GenerateStream(ctx context.Context, messages []*content.Message, config *GenerationConfig) (<-chan StreamChunk, error) {
	ch := make(chan StreamChunk)

	go func() {
		defer close(ch)

		resp, err := p.Generate(ctx, messages, config)
		if err != nil {
			ch <- StreamChunk{Error: err}
			return
		}

		ch <- StreamChunk{
			ID:           resp.ID,
			Delta:        resp.Message.GetText(),
			FinishReason: resp.FinishReason,
			Usage:        resp.Usage,
		}
	}()

	return ch, nil
}

// convertMessage converts a single content.Message to Anthropic format.
func (p *AnthropicMultimodalProvider) convertMessage(msg *content.Message) map[string]any {
	converted := map[string]any{
		"role": string(msg.Role),
	}

	if msg.IsMultimodal() {
		parts := make([]map[string]any, 0)
		for _, c := range msg.Contents {
			switch c.Type {
			case content.TypeText:
				parts = append(parts, map[string]any{
					"type": "text",
					"text": c.Text,
				})
			case content.TypeImage:
				parts = append(parts, map[string]any{
					"type": "image",
					"source": map[string]string{
						"type":       "base64",
						"media_type": string(c.MediaType),
						"data":       base64.StdEncoding.EncodeToString(c.Data),
					},
				})
			case content.TypeDocument:
				if c.MediaType == content.MediaPDF {
					parts = append(parts, map[string]any{
						"type": "document",
						"source": map[string]string{
							"type":       "base64",
							"media_type": "application/pdf",
							"data":       base64.StdEncoding.EncodeToString(c.Data),
						},
					})
				}
			}
		}
		converted["content"] = parts
	} else {
		converted["content"] = msg.GetText()
	}

	return converted
}
