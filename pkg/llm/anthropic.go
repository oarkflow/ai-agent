package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// AnthropicProvider implements the Provider interface for Anthropic.
// NOTE: For multimodal support and config-based initialization, use AnthropicMultimodalProvider instead.
type AnthropicProvider struct {
	APIKey     string
	Model      string
	BaseURL    string
	APIVersion string
	Client     *http.Client
}

// NewAnthropicProvider creates a new instance of AnthropicProvider.
// When using JSON configuration, prefer using ProviderFactory.CreateProvider("anthropic") instead.
func NewAnthropicProvider(apiKey string, model string) *AnthropicProvider {
	if model == "" {
		model = "claude-sonnet-4-20250514" // Default, override via config
	}
	return &AnthropicProvider{
		APIKey:     apiKey,
		Model:      model,
		BaseURL:    "https://api.anthropic.com/v1", // Default, override via config
		APIVersion: "2023-06-01",                   // Default, override via config
		Client:     &http.Client{},
	}
}

type anthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type anthropicRequest struct {
	Model       string             `json:"model"`
	Messages    []anthropicMessage `json:"messages"`
	System      string             `json:"system,omitempty"`
	MaxTokens   int                `json:"max_tokens"`
	Temperature float64            `json:"temperature,omitempty"`
}

type anthropicResponse struct {
	Content []struct {
		Text string `json:"text"`
		Type string `json:"type"`
	} `json:"content"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

func (p *AnthropicProvider) Generate(ctx context.Context, prompt string, options *GenerateOptions) (string, error) {
	messages := []Message{
		{Role: RoleUser, Content: prompt},
	}
	return p.Chat(ctx, messages, options)
}

func (p *AnthropicProvider) Chat(ctx context.Context, messages []Message, options *GenerateOptions) (string, error) {
	reqData := anthropicRequest{
		Model:     p.Model,
		Messages:  []anthropicMessage{},
		MaxTokens: 1024,
	}

	// Anthropic handles system prompts separately
	for _, msg := range messages {
		if msg.Role == RoleSystem {
			reqData.System += msg.Content + "\n"
		} else {
			reqData.Messages = append(reqData.Messages, anthropicMessage{
				Role:    msg.Role,
				Content: msg.Content,
			})
		}
	}
	reqData.System = strings.TrimSpace(reqData.System)

	if options != nil {
		if options.Model != "" {
			reqData.Model = options.Model
		}
		if options.MaxTokens > 0 {
			reqData.MaxTokens = options.MaxTokens
		}
		reqData.Temperature = options.Temperature
	}

	jsonData, err := json.Marshal(reqData)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	baseURL := p.BaseURL
	if baseURL == "" {
		baseURL = "https://api.anthropic.com/v1"
	}

	req, err := http.NewRequestWithContext(ctx, "POST", baseURL+"/messages", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.APIKey)
	apiVersion := p.APIVersion
	if apiVersion == "" {
		apiVersion = "2023-06-01"
	}
	req.Header.Set("anthropic-version", apiVersion)

	resp, err := p.Client.Do(req)
	if err != nil {
		return "", fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("api error (status %d): %s", resp.StatusCode, string(body))
	}

	var apiResp anthropicResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return "", fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if apiResp.Error != nil {
		return "", fmt.Errorf("api returned error: %s", apiResp.Error.Message)
	}

	if len(apiResp.Content) == 0 {
		return "", fmt.Errorf("no content returned")
	}

	return apiResp.Content[0].Text, nil
}
