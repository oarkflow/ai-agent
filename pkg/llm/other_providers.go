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

// DeepSeekProvider implements MultimodalProvider for DeepSeek.
type DeepSeekProvider struct {
	BaseProvider
	Client *http.Client
}

// NewDeepSeekProvider creates a new DeepSeek provider.
// When using JSON configuration, prefer using ProviderFactory.CreateProvider("deepseek") instead.
func NewDeepSeekProvider(apiKey string) *DeepSeekProvider {
	return &DeepSeekProvider{
		BaseProvider: BaseProvider{
			ProviderType: ProviderDeepSeek,
			APIKey:       apiKey,
			BaseURL:      "https://api.deepseek.com/v1", // Default, override via config
			DefaultModel: "deepseek-chat",               // Default, override via config
			Models:       make(map[string]*ModelInfo),   // Loaded from config
		},
		Client: &http.Client{Timeout: 120 * time.Second},
	}
}

// Note: Model definitions are now loaded from config/config.json

func (p *DeepSeekProvider) Generate(ctx context.Context, messages []*content.Message, config *GenerationConfig) (*GenerationResponse, error) {
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
		if len(config.StopSequences) > 0 {
			reqBody["stop"] = config.StopSequences
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
				Role             string `json:"role"`
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content,omitempty"`
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
	responseText := choice.Message.Content

	// Include reasoning for R1 model
	if choice.Message.ReasoningContent != "" {
		responseText = fmt.Sprintf("<reasoning>\n%s\n</reasoning>\n\n%s",
			choice.Message.ReasoningContent, choice.Message.Content)
	}

	return &GenerationResponse{
		ID:           resp.ID,
		Model:        resp.Model,
		Message:      content.NewTextMessage(content.RoleAssistant, responseText),
		FinishReason: choice.FinishReason,
		Usage: &Usage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
			TotalTokens:  resp.Usage.TotalTokens,
		},
	}, nil
}

func (p *DeepSeekProvider) GenerateStream(ctx context.Context, messages []*content.Message, config *GenerationConfig) (<-chan StreamChunk, error) {
	// Similar to OpenAI streaming implementation
	ch := make(chan StreamChunk)
	go func() {
		defer close(ch)
		ch <- StreamChunk{Error: fmt.Errorf("streaming not implemented for DeepSeek")}
	}()
	return ch, nil
}

func (p *DeepSeekProvider) Embed(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
	return nil, ErrCapabilityNotSupported
}

func (p *DeepSeekProvider) Transcribe(ctx context.Context, req *TranscriptionRequest) (*TranscriptionResponse, error) {
	return nil, ErrCapabilityNotSupported
}

func (p *DeepSeekProvider) GenerateImage(ctx context.Context, req *ImageGenerationRequest) (*ImageGenerationResponse, error) {
	return nil, ErrCapabilityNotSupported
}

func (p *DeepSeekProvider) GenerateSpeech(ctx context.Context, req *SpeechRequest) (*SpeechResponse, error) {
	return nil, ErrCapabilityNotSupported
}

func (p *DeepSeekProvider) ListModels(ctx context.Context) ([]ModelInfo, error) {
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

func (p *DeepSeekProvider) GetCapabilities() []Capability {
	return []Capability{CapText, CapCodeGeneration, CapReasoning, CapFunctionCall}
}

func (p *DeepSeekProvider) SupportsCapability(cap Capability) bool {
	for _, c := range p.GetCapabilities() {
		if c == cap {
			return true
		}
	}
	return false
}

func (p *DeepSeekProvider) convertMessages(messages []*content.Message) []map[string]string {
	result := make([]map[string]string, 0, len(messages))
	for _, msg := range messages {
		result = append(result, map[string]string{
			"role":    string(msg.Role),
			"content": msg.GetText(),
		})
	}
	return result
}

func (p *DeepSeekProvider) doRequest(ctx context.Context, method, path string, body any) ([]byte, error) {
	jsonData, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, method, p.BaseURL+path, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.APIKey)

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

// MistralProvider implements MultimodalProvider for Mistral AI.
type MistralProvider struct {
	BaseProvider
	Client *http.Client
}

// NewMistralProvider creates a new Mistral provider.
// When using JSON configuration, prefer using ProviderFactory.CreateProvider("mistral") instead.
func NewMistralProvider(apiKey string) *MistralProvider {
	return &MistralProvider{
		BaseProvider: BaseProvider{
			ProviderType: ProviderMistral,
			APIKey:       apiKey,
			BaseURL:      "https://api.mistral.ai/v1", // Default, override via config
			DefaultModel: "mistral-large-latest",      // Default, override via config
			Models:       make(map[string]*ModelInfo), // Loaded from config
		},
		Client: &http.Client{Timeout: 120 * time.Second},
	}
}

// Note: Model definitions are now loaded from config/config.json

func (p *MistralProvider) Generate(ctx context.Context, messages []*content.Message, config *GenerationConfig) (*GenerationResponse, error) {
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
				Role    string `json:"role"`
				Content string `json:"content"`
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
	return &GenerationResponse{
		ID:           resp.ID,
		Model:        resp.Model,
		Message:      content.NewTextMessage(content.RoleAssistant, choice.Message.Content),
		FinishReason: choice.FinishReason,
		Usage: &Usage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
			TotalTokens:  resp.Usage.TotalTokens,
		},
	}, nil
}

func (p *MistralProvider) GenerateStream(ctx context.Context, messages []*content.Message, config *GenerationConfig) (<-chan StreamChunk, error) {
	ch := make(chan StreamChunk)
	go func() {
		defer close(ch)
		ch <- StreamChunk{Error: fmt.Errorf("streaming not fully implemented for Mistral")}
	}()
	return ch, nil
}

func (p *MistralProvider) Embed(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
	model := "mistral-embed"
	if req.Model != "" {
		model = req.Model
	}

	reqBody := map[string]any{
		"model": model,
		"input": req.Input,
	}

	body, err := p.doRequest(ctx, "POST", "/embeddings", reqBody)
	if err != nil {
		return nil, err
	}

	var resp struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
		Model string `json:"model"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
			TotalTokens  int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse embedding response: %w", err)
	}

	embeddings := make([][]float64, len(resp.Data))
	for i, d := range resp.Data {
		embeddings[i] = d.Embedding
	}

	return &EmbeddingResponse{
		Embeddings: embeddings,
		Model:      resp.Model,
		Usage: &Usage{
			InputTokens: resp.Usage.PromptTokens,
			TotalTokens: resp.Usage.TotalTokens,
		},
	}, nil
}

func (p *MistralProvider) Transcribe(ctx context.Context, req *TranscriptionRequest) (*TranscriptionResponse, error) {
	return nil, ErrCapabilityNotSupported
}

func (p *MistralProvider) GenerateImage(ctx context.Context, req *ImageGenerationRequest) (*ImageGenerationResponse, error) {
	return nil, ErrCapabilityNotSupported
}

func (p *MistralProvider) GenerateSpeech(ctx context.Context, req *SpeechRequest) (*SpeechResponse, error) {
	return nil, ErrCapabilityNotSupported
}

func (p *MistralProvider) ListModels(ctx context.Context) ([]ModelInfo, error) {
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

func (p *MistralProvider) GetCapabilities() []Capability {
	return []Capability{CapText, CapVision, CapCodeGeneration, CapFunctionCall, CapStreaming, CapEmbedding}
}

func (p *MistralProvider) SupportsCapability(cap Capability) bool {
	for _, c := range p.GetCapabilities() {
		if c == cap {
			return true
		}
	}
	return false
}

func (p *MistralProvider) convertMessages(messages []*content.Message) []map[string]any {
	result := make([]map[string]any, 0, len(messages))
	for _, msg := range messages {
		converted := map[string]any{
			"role": string(msg.Role),
		}

		// Handle multimodal content for Pixtral
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

func (p *MistralProvider) doRequest(ctx context.Context, method, path string, body any) ([]byte, error) {
	jsonData, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, method, p.BaseURL+path, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.APIKey)

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

// XAIProvider implements MultimodalProvider for xAI (Grok).
type XAIProvider struct {
	BaseProvider
	Client *http.Client
}

// NewXAIProvider creates a new xAI provider.
// When using JSON configuration, prefer using ProviderFactory.CreateProvider("xai") instead.
func NewXAIProvider(apiKey string) *XAIProvider {
	return &XAIProvider{
		BaseProvider: BaseProvider{
			ProviderType: ProviderXAI,
			APIKey:       apiKey,
			BaseURL:      "https://api.x.ai/v1",       // Default, override via config
			DefaultModel: "grok-2",                    // Default, override via config
			Models:       make(map[string]*ModelInfo), // Loaded from config
		},
		Client: &http.Client{Timeout: 120 * time.Second},
	}
}

// Note: Model definitions are now loaded from config/config.json

func (p *XAIProvider) Generate(ctx context.Context, messages []*content.Message, config *GenerationConfig) (*GenerationResponse, error) {
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
				Role    string `json:"role"`
				Content string `json:"content"`
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
	return &GenerationResponse{
		ID:           resp.ID,
		Model:        resp.Model,
		Message:      content.NewTextMessage(content.RoleAssistant, choice.Message.Content),
		FinishReason: choice.FinishReason,
		Usage: &Usage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
			TotalTokens:  resp.Usage.TotalTokens,
		},
	}, nil
}

func (p *XAIProvider) GenerateStream(ctx context.Context, messages []*content.Message, config *GenerationConfig) (<-chan StreamChunk, error) {
	ch := make(chan StreamChunk)
	go func() {
		defer close(ch)
		ch <- StreamChunk{Error: fmt.Errorf("streaming not fully implemented for xAI")}
	}()
	return ch, nil
}

func (p *XAIProvider) Embed(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
	return nil, ErrCapabilityNotSupported
}

func (p *XAIProvider) Transcribe(ctx context.Context, req *TranscriptionRequest) (*TranscriptionResponse, error) {
	return nil, ErrCapabilityNotSupported
}

func (p *XAIProvider) GenerateImage(ctx context.Context, req *ImageGenerationRequest) (*ImageGenerationResponse, error) {
	return nil, ErrCapabilityNotSupported
}

func (p *XAIProvider) GenerateSpeech(ctx context.Context, req *SpeechRequest) (*SpeechResponse, error) {
	return nil, ErrCapabilityNotSupported
}

func (p *XAIProvider) ListModels(ctx context.Context) ([]ModelInfo, error) {
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

func (p *XAIProvider) GetCapabilities() []Capability {
	return []Capability{CapText, CapVision, CapAudio, CapCodeGeneration, CapFunctionCall, CapStreaming, CapReasoning, CapWebSearch}
}

func (p *XAIProvider) SupportsCapability(cap Capability) bool {
	for _, c := range p.GetCapabilities() {
		if c == cap {
			return true
		}
	}
	return false
}

func (p *XAIProvider) convertMessages(messages []*content.Message) []map[string]any {
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

func (p *XAIProvider) doRequest(ctx context.Context, method, path string, body any) ([]byte, error) {
	jsonData, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, method, p.BaseURL+path, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.APIKey)

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
