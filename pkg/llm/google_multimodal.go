package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/sujit/ai-agent/pkg/content"
)

// GoogleMultimodalProvider implements MultimodalProvider for Google Gemini.
type GoogleMultimodalProvider struct {
	BaseProvider
	Client      *http.Client
	ProjectID   string
	Location    string
	UseVertexAI bool // Use Vertex AI instead of AI Studio
}

// NewGoogleMultimodalProvider creates a new Google Gemini multimodal provider.
// When using JSON configuration, prefer using ProviderFactory.CreateProvider("google") instead.
func NewGoogleMultimodalProvider(apiKey string, opts ...GoogleOption) *GoogleMultimodalProvider {
	p := &GoogleMultimodalProvider{
		BaseProvider: BaseProvider{
			ProviderType: ProviderGoogle,
			APIKey:       apiKey,
			BaseURL:      "",                          // Set via config or WithGoogleBaseURL option
			DefaultModel: "",                          // Set via config or WithGoogleModel option
			Models:       make(map[string]*ModelInfo), // Loaded from config
		},
		Client: &http.Client{Timeout: 300 * time.Second}, // Longer timeout for video
	}

	for _, opt := range opts {
		opt(p)
	}

	// Apply defaults only if not set via options (backward compatibility)
	if p.BaseURL == "" {
		p.BaseURL = "https://generativelanguage.googleapis.com/v1beta"
	}
	if p.DefaultModel == "" {
		p.DefaultModel = "gemini-2.0-flash"
	}

	return p
}

// GoogleOption is a configuration option for Google provider.
type GoogleOption func(*GoogleMultimodalProvider)

// WithVertexAI configures the provider for Vertex AI.
func WithVertexAI(projectID, location string) GoogleOption {
	return func(p *GoogleMultimodalProvider) {
		p.UseVertexAI = true
		p.ProjectID = projectID
		p.Location = location
		p.BaseURL = fmt.Sprintf("https://%s-aiplatform.googleapis.com/v1/projects/%s/locations/%s/publishers/google/models",
			location, projectID, location)
	}
}

// WithGoogleBaseURL sets custom base URL.
func WithGoogleBaseURL(url string) GoogleOption {
	return func(p *GoogleMultimodalProvider) {
		p.BaseURL = url
	}
}

// WithGoogleModel sets the default model.
func WithGoogleModel(model string) GoogleOption {
	return func(p *GoogleMultimodalProvider) {
		p.DefaultModel = model
	}
}

// WithGoogleModels sets the available models from config.
func WithGoogleModels(models map[string]*ModelInfo) GoogleOption {
	return func(p *GoogleMultimodalProvider) {
		p.Models = models
	}
}

// WithGoogleTimeout sets the HTTP client timeout.
func WithGoogleTimeout(timeout time.Duration) GoogleOption {
	return func(p *GoogleMultimodalProvider) {
		p.Client = &http.Client{Timeout: timeout}
	}
}

// Generate implements multimodal text generation.
func (p *GoogleMultimodalProvider) Generate(ctx context.Context, messages []*content.Message, config *GenerationConfig) (*GenerationResponse, error) {
	model := p.DefaultModel
	if config != nil && config.Model != "" {
		model = config.Model
	}

	reqBody := p.buildRequest(messages, config)

	endpoint := p.getEndpoint(model, "generateContent")
	body, err := p.doRequest(ctx, "POST", endpoint, reqBody)
	if err != nil {
		return nil, err
	}

	var resp geminiResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if resp.Error != nil {
		return nil, fmt.Errorf("API error: %s", resp.Error.Message)
	}

	if len(resp.Candidates) == 0 {
		return nil, fmt.Errorf("no candidates in response")
	}

	// Extract text content
	var textContent string
	candidate := resp.Candidates[0]
	for _, part := range candidate.Content.Parts {
		if part.Text != "" {
			textContent += part.Text
		}
	}

	return &GenerationResponse{
		Model:        model,
		Message:      content.NewTextMessage(content.RoleAssistant, textContent),
		FinishReason: candidate.FinishReason,
		Usage: &Usage{
			InputTokens:  resp.UsageMetadata.PromptTokenCount,
			OutputTokens: resp.UsageMetadata.CandidatesTokenCount,
			TotalTokens:  resp.UsageMetadata.TotalTokenCount,
		},
	}, nil
}

// GenerateStream implements streaming generation.
func (p *GoogleMultimodalProvider) GenerateStream(ctx context.Context, messages []*content.Message, config *GenerationConfig) (<-chan StreamChunk, error) {
	model := p.DefaultModel
	if config != nil && config.Model != "" {
		model = config.Model
	}

	reqBody := p.buildRequest(messages, config)

	ch := make(chan StreamChunk, 100)

	go func() {
		defer close(ch)

		endpoint := p.getEndpoint(model, "streamGenerateContent") + "&alt=sse"

		jsonData, err := json.Marshal(reqBody)
		if err != nil {
			ch <- StreamChunk{Error: err}
			return
		}

		req, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewBuffer(jsonData))
		if err != nil {
			ch <- StreamChunk{Error: err}
			return
		}

		req.Header.Set("Content-Type", "application/json")

		resp, err := p.Client.Do(req)
		if err != nil {
			ch <- StreamChunk{Error: err}
			return
		}
		defer resp.Body.Close()

		decoder := json.NewDecoder(resp.Body)
		for {
			select {
			case <-ctx.Done():
				ch <- StreamChunk{Error: ctx.Err()}
				return
			default:
				var chunk geminiResponse
				if err := decoder.Decode(&chunk); err != nil {
					if err == io.EOF {
						return
					}
					continue
				}

				if len(chunk.Candidates) > 0 {
					for _, part := range chunk.Candidates[0].Content.Parts {
						if part.Text != "" {
							ch <- StreamChunk{Delta: part.Text}
						}
					}
					if chunk.Candidates[0].FinishReason != "" {
						ch <- StreamChunk{FinishReason: chunk.Candidates[0].FinishReason}
					}
				}
			}
		}
	}()

	return ch, nil
}

// Embed creates embeddings.
func (p *GoogleMultimodalProvider) Embed(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
	model := "text-embedding-004"
	if req.Model != "" {
		model = req.Model
	}

	embeddings := make([][]float64, len(req.Input))

	for i, text := range req.Input {
		reqBody := map[string]any{
			"model": "models/" + model,
			"content": map[string]any{
				"parts": []map[string]any{
					{"text": text},
				},
			},
		}

		endpoint := p.getEndpoint(model, "embedContent")
		body, err := p.doRequest(ctx, "POST", endpoint, reqBody)
		if err != nil {
			return nil, err
		}

		var resp struct {
			Embedding struct {
				Values []float64 `json:"values"`
			} `json:"embedding"`
		}
		if err := json.Unmarshal(body, &resp); err != nil {
			return nil, err
		}

		embeddings[i] = resp.Embedding.Values
	}

	return &EmbeddingResponse{
		Embeddings: embeddings,
		Model:      model,
	}, nil
}

// Transcribe - Google handles audio natively in generate.
func (p *GoogleMultimodalProvider) Transcribe(ctx context.Context, req *TranscriptionRequest) (*TranscriptionResponse, error) {
	// Use Gemini to transcribe audio
	msg := content.NewUserMessage("Please transcribe this audio accurately. Provide only the transcription text, no commentary.")
	msg.AddContent(req.Audio)

	resp, err := p.Generate(ctx, []*content.Message{msg}, &GenerationConfig{
		Model: "gemini-2.0-flash",
	})
	if err != nil {
		return nil, err
	}

	return &TranscriptionResponse{
		Text: resp.Message.GetText(),
	}, nil
}

// GenerateImage generates images using Imagen.
func (p *GoogleMultimodalProvider) GenerateImage(ctx context.Context, req *ImageGenerationRequest) (*ImageGenerationResponse, error) {
	model := "imagen-3.0-generate-001"
	if req.Model != "" {
		model = req.Model
	}

	n := 1
	if req.N > 0 {
		n = req.N
	}

	reqBody := map[string]any{
		"instances": []map[string]any{
			{"prompt": req.Prompt},
		},
		"parameters": map[string]any{
			"sampleCount": n,
		},
	}

	endpoint := p.getEndpoint(model, "predict")
	body, err := p.doRequest(ctx, "POST", endpoint, reqBody)
	if err != nil {
		return nil, err
	}

	var resp struct {
		Predictions []struct {
			BytesBase64Encoded string `json:"bytesBase64Encoded"`
		} `json:"predictions"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, err
	}

	images := make([]*content.Content, len(resp.Predictions))
	for i, pred := range resp.Predictions {
		images[i] = &content.Content{
			Type:       content.TypeImage,
			MediaType:  content.MediaPNG,
			Base64Data: pred.BytesBase64Encoded,
		}
	}

	return &ImageGenerationResponse{
		Images: images,
		Model:  model,
	}, nil
}

// GenerateSpeech - Not directly supported by Google AI Studio (use Cloud TTS).
func (p *GoogleMultimodalProvider) GenerateSpeech(ctx context.Context, req *SpeechRequest) (*SpeechResponse, error) {
	return nil, ErrCapabilityNotSupported
}

// ListModels lists available models.
func (p *GoogleMultimodalProvider) ListModels(ctx context.Context) ([]ModelInfo, error) {
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

// GetCapabilities returns provider capabilities.
func (p *GoogleMultimodalProvider) GetCapabilities() []Capability {
	return []Capability{
		CapText, CapVision, CapAudio, CapVideo, CapDocument,
		CapCodeGeneration, CapCodeExecution, CapReasoning,
		CapFunctionCall, CapStreaming, CapEmbedding,
		CapImageGen, CapWebSearch,
	}
}

// SupportsCapability checks if a capability is supported.
func (p *GoogleMultimodalProvider) SupportsCapability(cap Capability) bool {
	for _, c := range p.GetCapabilities() {
		if c == cap {
			return true
		}
	}
	return false
}

// Helper methods

func (p *GoogleMultimodalProvider) buildRequest(messages []*content.Message, config *GenerationConfig) map[string]any {
	req := map[string]any{
		"contents": p.convertMessages(messages),
	}

	genConfig := make(map[string]any)
	if config != nil {
		if config.Temperature != 0 {
			genConfig["temperature"] = config.Temperature
		}
		if config.TopP != 0 {
			genConfig["topP"] = config.TopP
		}
		if config.TopK > 0 {
			genConfig["topK"] = config.TopK
		}
		if config.MaxTokens > 0 {
			genConfig["maxOutputTokens"] = config.MaxTokens
		}
		if len(config.StopSequences) > 0 {
			genConfig["stopSequences"] = config.StopSequences
		}
		if config.ResponseFormat != nil {
			if config.ResponseFormat.Type == "json_object" {
				genConfig["responseMimeType"] = "application/json"
			}
			if config.ResponseFormat.JSONSchema != nil {
				genConfig["responseSchema"] = config.ResponseFormat.JSONSchema.Schema
			}
		}
	}

	if len(genConfig) > 0 {
		req["generationConfig"] = genConfig
	}

	// System instruction
	for _, m := range messages {
		if m.Role == content.RoleSystem {
			req["systemInstruction"] = map[string]any{
				"parts": []map[string]any{
					{"text": m.GetText()},
				},
			}
			break
		}
	}

	// Add tools if specified
	if config != nil && len(config.Tools) > 0 {
		req["tools"] = p.convertTools(config.Tools)
	}

	return req
}

func (p *GoogleMultimodalProvider) convertMessages(messages []*content.Message) []map[string]any {
	result := make([]map[string]any, 0, len(messages))

	for _, msg := range messages {
		// Skip system messages (handled separately)
		if msg.Role == content.RoleSystem {
			continue
		}

		role := "user"
		if msg.Role == content.RoleAssistant {
			role = "model"
		}

		parts := make([]map[string]any, 0, len(msg.Contents))
		for _, c := range msg.Contents {
			switch c.Type {
			case content.TypeText, content.TypeCode:
				parts = append(parts, map[string]any{
					"text": c.Text,
				})
			case content.TypeImage:
				if c.URL != "" {
					parts = append(parts, map[string]any{
						"fileData": map[string]any{
							"mimeType": string(c.MediaType),
							"fileUri":  c.URL,
						},
					})
				} else {
					parts = append(parts, map[string]any{
						"inlineData": map[string]any{
							"mimeType": string(c.MediaType),
							"data":     c.GetBase64(),
						},
					})
				}
			case content.TypeAudio:
				parts = append(parts, map[string]any{
					"inlineData": map[string]any{
						"mimeType": string(c.MediaType),
						"data":     c.GetBase64(),
					},
				})
			case content.TypeVideo:
				if c.URL != "" {
					parts = append(parts, map[string]any{
						"fileData": map[string]any{
							"mimeType": string(c.MediaType),
							"fileUri":  c.URL,
						},
					})
				} else {
					parts = append(parts, map[string]any{
						"inlineData": map[string]any{
							"mimeType": string(c.MediaType),
							"data":     c.GetBase64(),
						},
					})
				}
			case content.TypeDocument:
				parts = append(parts, map[string]any{
					"inlineData": map[string]any{
						"mimeType": string(c.MediaType),
						"data":     c.GetBase64(),
					},
				})
			}
		}

		result = append(result, map[string]any{
			"role":  role,
			"parts": parts,
		})
	}

	return result
}

func (p *GoogleMultimodalProvider) convertTools(tools []Tool) []map[string]any {
	functionDeclarations := make([]map[string]any, len(tools))
	for i, t := range tools {
		functionDeclarations[i] = map[string]any{
			"name":        t.Function.Name,
			"description": t.Function.Description,
			"parameters":  t.Function.Parameters,
		}
	}
	return []map[string]any{
		{"functionDeclarations": functionDeclarations},
	}
}

func (p *GoogleMultimodalProvider) getEndpoint(model, action string) string {
	if p.UseVertexAI {
		return fmt.Sprintf("%s/%s:%s", p.BaseURL, model, action)
	}
	return fmt.Sprintf("%s/models/%s:%s?key=%s", p.BaseURL, model, action, p.APIKey)
}

func (p *GoogleMultimodalProvider) doRequest(ctx context.Context, method, url string, body any) ([]byte, error) {
	jsonData, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if p.UseVertexAI {
		// For Vertex AI, use OAuth token
		req.Header.Set("Authorization", "Bearer "+p.APIKey)
	}

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

// Response types

type geminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text         string `json:"text,omitempty"`
				FunctionCall *struct {
					Name string         `json:"name"`
					Args map[string]any `json:"args"`
				} `json:"functionCall,omitempty"`
			} `json:"parts"`
			Role string `json:"role"`
		} `json:"content"`
		FinishReason  string `json:"finishReason"`
		SafetyRatings []struct {
			Category    string `json:"category"`
			Probability string `json:"probability"`
		} `json:"safetyRatings"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
	Error *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Status  string `json:"status"`
	} `json:"error,omitempty"`
}

// UploadFile uploads a file to Google's File API for processing large media.
func (p *GoogleMultimodalProvider) UploadFile(ctx context.Context, c *content.Content) (string, error) {
	if p.UseVertexAI {
		return "", fmt.Errorf("file upload not supported for Vertex AI in this implementation")
	}

	// Step 1: Start resumable upload
	initURL := fmt.Sprintf("https://generativelanguage.googleapis.com/upload/v1beta/files?key=%s", p.APIKey)

	metadata := map[string]any{
		"file": map[string]any{
			"display_name": c.FileName,
		},
	}
	metaJSON, _ := json.Marshal(metadata)

	req, err := http.NewRequestWithContext(ctx, "POST", initURL, bytes.NewBuffer(metaJSON))
	if err != nil {
		return "", err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Goog-Upload-Protocol", "resumable")
	req.Header.Set("X-Goog-Upload-Command", "start")
	req.Header.Set("X-Goog-Upload-Header-Content-Length", fmt.Sprintf("%d", len(c.Data)))
	req.Header.Set("X-Goog-Upload-Header-Content-Type", string(c.MediaType))

	resp, err := p.Client.Do(req)
	if err != nil {
		return "", err
	}
	resp.Body.Close()

	uploadURL := resp.Header.Get("X-Goog-Upload-URL")
	if uploadURL == "" {
		return "", fmt.Errorf("no upload URL received")
	}

	// Step 2: Upload the file data
	req2, err := http.NewRequestWithContext(ctx, "POST", uploadURL, bytes.NewBuffer(c.Data))
	if err != nil {
		return "", err
	}

	req2.Header.Set("Content-Length", fmt.Sprintf("%d", len(c.Data)))
	req2.Header.Set("X-Goog-Upload-Offset", "0")
	req2.Header.Set("X-Goog-Upload-Command", "upload, finalize")

	resp2, err := p.Client.Do(req2)
	if err != nil {
		return "", err
	}
	defer resp2.Body.Close()

	body, _ := io.ReadAll(resp2.Body)

	var result struct {
		File struct {
			Name  string `json:"name"`
			URI   string `json:"uri"`
			State string `json:"state"`
		} `json:"file"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return "", err
	}

	// Wait for processing
	for result.File.State == "PROCESSING" {
		time.Sleep(2 * time.Second)
		statusURL := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/%s?key=%s", result.File.Name, p.APIKey)
		statusResp, err := http.Get(statusURL)
		if err != nil {
			return "", err
		}
		statusBody, _ := io.ReadAll(statusResp.Body)
		statusResp.Body.Close()

		var status struct {
			State string `json:"state"`
			URI   string `json:"uri"`
		}
		json.Unmarshal(statusBody, &status)
		result.File.State = status.State
		result.File.URI = status.URI
	}

	if result.File.State != "ACTIVE" {
		return "", fmt.Errorf("file processing failed: %s", result.File.State)
	}

	return result.File.URI, nil
}

// DeleteFile deletes an uploaded file.
func (p *GoogleMultimodalProvider) DeleteFile(ctx context.Context, fileName string) error {
	if !strings.HasPrefix(fileName, "files/") {
		fileName = "files/" + fileName
	}

	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/%s?key=%s", fileName, p.APIKey)
	req, err := http.NewRequestWithContext(ctx, "DELETE", url, nil)
	if err != nil {
		return err
	}

	resp, err := p.Client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNoContent {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("failed to delete file: %s", string(body))
	}

	return nil
}
