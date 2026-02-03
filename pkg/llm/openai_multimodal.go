package llm

import (
	"bytes"
	"context"
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

type OpenAIMultimodalProvider struct {
	BaseProvider
	Client       *http.Client
	Organization string
	Project      string
}

type OpenAIOption func(*OpenAIMultimodalProvider)

// NewOpenAIMultimodalProvider creates a new OpenAI provider.
// When using JSON configuration, prefer using ProviderFactory.CreateProvider("openai") instead.
func NewOpenAIMultimodalProvider(apiKey string, opts ...OpenAIOption) *OpenAIMultimodalProvider {
	p := &OpenAIMultimodalProvider{
		BaseProvider: BaseProvider{
			ProviderType: ProviderOpenAI,
			APIKey:       apiKey,
			BaseURL:      "",  // Set via config or WithOpenAIBaseURL option
			DefaultModel: "", // Set via config or WithOpenAIModel option
			Models:       make(map[string]*ModelInfo), // Loaded from config
		},
		Client: &http.Client{Timeout: 120 * time.Second},
	}

	for _, opt := range opts {
		opt(p)
	}

	// Apply defaults only if not set via options (backward compatibility)
	if p.BaseURL == "" {
		p.BaseURL = "https://api.openai.com/v1"
	}
	if p.DefaultModel == "" {
		p.DefaultModel = "gpt-4o"
	}

	return p
}

func WithOpenAIBaseURL(url string) OpenAIOption {
	return func(p *OpenAIMultimodalProvider) {
		p.BaseURL = url
	}
}

func WithOpenAIProject(project string) OpenAIOption {
	return func(p *OpenAIMultimodalProvider) {
		p.Project = project
	}
}

func WithOpenAIOrganization(org string) OpenAIOption {
	return func(p *OpenAIMultimodalProvider) {
		p.Organization = org
	}
}

// WithOpenAIModel sets the default model.
func WithOpenAIModel(model string) OpenAIOption {
	return func(p *OpenAIMultimodalProvider) {
		p.DefaultModel = model
	}
}

// WithOpenAIModels sets the available models from config.
func WithOpenAIModels(models map[string]*ModelInfo) OpenAIOption {
	return func(p *OpenAIMultimodalProvider) {
		p.Models = models
	}
}

// WithOpenAITimeout sets the HTTP client timeout.
func WithOpenAITimeout(timeout time.Duration) OpenAIOption {
	return func(p *OpenAIMultimodalProvider) {
		p.Client = &http.Client{Timeout: timeout}
	}
}

// ------------------------------
// Helper Methods
// ------------------------------

func (p *OpenAIMultimodalProvider) setHeaders(req *http.Request) {
	req.Header.Set("Authorization", "Bearer "+p.APIKey)
	req.Header.Set("Content-Type", "application/json")

	if p.Organization != "" {
		req.Header.Set("OpenAI-Organization", p.Organization)
	}
	if p.Project != "" {
		req.Header.Set("OpenAI-Project", p.Project)
	}
}

func (p *OpenAIMultimodalProvider) doRequest(ctx context.Context, method, path string, body any) ([]byte, error) {
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

// ------------------------------
// Capabilities
// ------------------------------

func (p *OpenAIMultimodalProvider) GetCapabilities() []Capability {
	return []Capability{
		CapText, CapVision, CapAudio, CapVideo,
		CapSpeechGen, CapTranscription, CapEmbedding,
		CapFunctionCall, CapStreaming, CapReasoning, CapAgents, CapCodeGeneration,
		CapImageGen,
	}
}

func (p *OpenAIMultimodalProvider) SupportsCapability(cap Capability) bool {
	for _, c := range p.GetCapabilities() {
		if c == cap {
			return true
		}
	}
	return false
}

// ------------------------------
// Models
// ------------------------------

func (p *OpenAIMultimodalProvider) ListModels(ctx context.Context) ([]ModelInfo, error) {
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
func (p *OpenAIMultimodalProvider) Generate(ctx context.Context, messages []*content.Message, config *GenerationConfig) (*GenerationResponse, error) {
	model := p.DefaultModel
	if config != nil && config.Model != "" {
		model = config.Model
	}

	reqBody := map[string]any{
		"model":    model,
		"messages": p.convertMessages(messages),
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
			reqBody["stop"] = config.StopSequences
		}
		if len(config.Tools) > 0 {
			reqBody["tools"] = config.Tools
		}
		if config.ResponseFormat != nil {
			reqBody["response_format"] = config.ResponseFormat
		}
	}

	body, err := p.doRequest(ctx, "POST", "/chat/completions", reqBody)
	if err != nil {
		return nil, err
	}

	var resp struct {
		ID      string `json:"id"`
		Model   string `json:"model"`
		Choices []struct {
			Message struct {
				Role       string `json:"role"`
				Content    string `json:"content"`
				ToolCalls  []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls,omitempty"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	choice := resp.Choices[0]

	// Convert tool calls
	var toolCalls []content.ToolCall
	for _, tc := range choice.Message.ToolCalls {
		toolCalls = append(toolCalls, content.ToolCall{
			ID:        tc.ID,
			Type:      tc.Type,
			Function:  content.FunctionCall{Name: tc.Function.Name, Arguments: tc.Function.Arguments},
		})
	}

	return &GenerationResponse{
		ID:           resp.ID,
		Model:        resp.Model,
		Message:      content.NewTextMessage(content.RoleAssistant, choice.Message.Content),
		FinishReason: choice.FinishReason,
		ToolCalls:    toolCalls,
		Usage: &Usage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
			TotalTokens:  resp.Usage.TotalTokens,
		},
	}, nil
}

// GenerateStream performs streaming text generation.
func (p *OpenAIMultimodalProvider) GenerateStream(ctx context.Context, messages []*content.Message, config *GenerationConfig) (<-chan StreamChunk, error) {
	ch := make(chan StreamChunk)

	go func() {
		defer close(ch)

		// For now, fall back to non-streaming
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

// convertMessages converts content.Message to OpenAI format.
func (p *OpenAIMultimodalProvider) convertMessages(messages []*content.Message) []map[string]any {
	result := make([]map[string]any, 0, len(messages))

	for _, msg := range messages {
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
					if c.URL != "" {
						parts = append(parts, map[string]any{
							"type":      "image_url",
							"image_url": map[string]string{"url": c.URL},
						})
					} else {
						parts = append(parts, map[string]any{
							"type":      "image_url",
							"image_url": map[string]string{"url": c.GetDataURI()},
						})
					}
				case content.TypeAudio:
					if c.URL != "" {
						parts = append(parts, map[string]any{
							"type": "input_audio",
							"input_audio": map[string]string{
								"data":   c.URL,
								"format": string(c.MediaType),
							},
						})
					}
				}
			}
			converted["content"] = parts
		} else {
			converted["content"] = msg.GetText()
		}

		result = append(result, converted)
	}

	return result
}

// ------------------------------
// Not Supported / Stubs
// ------------------------------

func (p *OpenAIMultimodalProvider) GenerateSpeech(ctx context.Context, req *SpeechRequest) (*SpeechResponse, error) {
	return nil, ErrCapabilityNotSupported
}

func (p *OpenAIMultimodalProvider) GenerateImage(ctx context.Context, req *ImageGenerationRequest) (*ImageGenerationResponse, error) {
	return nil, ErrCapabilityNotSupported
}

func (p *OpenAIMultimodalProvider) Transcribe(ctx context.Context, req *TranscriptionRequest) (*TranscriptionResponse, error) {
	return nil, ErrCapabilityNotSupported
}

func (p *OpenAIMultimodalProvider) Embed(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
	return nil, ErrCapabilityNotSupported
}
