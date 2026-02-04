package agent

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/sujit/ai-agent/pkg/content"
	"github.com/sujit/ai-agent/pkg/llm"
	"github.com/sujit/ai-agent/pkg/training"
)

// MultimodalAgent is an advanced agent that handles all content types.
type MultimodalAgent struct {
	mu             sync.RWMutex
	Name           string
	SystemPrompt   string
	Router         *llm.SmartRouter
	Registry       *llm.ProviderRegistry
	DomainTrainer  *training.DomainTrainer
	Conversation   *content.Conversation
	Config         *AgentConfig
	Preprocessors  []ContentPreprocessor
	Postprocessors []ResponsePostprocessor
	Tools          []llm.Tool
	EventHandlers  map[AgentEvent][]EventHandler
}

// AgentConfig configures the multimodal agent.
type AgentConfig struct {
	// Model selection
	DefaultModel    string
	PreferredModels map[content.ContentType]string

	// Generation settings
	Temperature float64
	MaxTokens   int
	TopP        float64

	// Behavior
	EnableRAG       bool
	EnableTools     bool
	EnableStreaming bool
	AutoPreprocess  bool // Auto-transcribe audio, extract PDF text, etc.

	// Domain
	DomainID string

	// Limits
	MaxConversationLength int
	MaxInputTokens        int

	// Timeouts
	Timeout time.Duration

	// Cost management
	MaxCostPerRequest float64
	TrackUsage        bool
}

// DefaultAgentConfig returns sensible defaults.
func DefaultAgentConfig() *AgentConfig {
	return &AgentConfig{
		Temperature:           0.7,
		MaxTokens:             4096,
		TopP:                  1.0,
		EnableRAG:             true,
		EnableTools:           true,
		EnableStreaming:       false,
		AutoPreprocess:        true,
		MaxConversationLength: 100,
		MaxInputTokens:        100000,
		Timeout:               120 * time.Second,
		TrackUsage:            true,
	}
}

// ContentPreprocessor preprocesses content before sending to LLM.
type ContentPreprocessor interface {
	CanProcess(c *content.Content) bool
	Process(ctx context.Context, c *content.Content) (*content.Content, error)
}

// ResponsePostprocessor postprocesses responses.
type ResponsePostprocessor interface {
	Process(ctx context.Context, response *llm.GenerationResponse) (*llm.GenerationResponse, error)
}

// AgentEvent represents events that can occur during agent operation.
type AgentEvent string

const (
	EventBeforeGenerate AgentEvent = "before_generate"
	EventAfterGenerate  AgentEvent = "after_generate"
	EventError          AgentEvent = "error"
	EventToolCall       AgentEvent = "tool_call"
	EventPreprocess     AgentEvent = "preprocess"
	EventRAGSearch      AgentEvent = "rag_search"
)

// EventHandler handles agent events.
type EventHandler func(ctx context.Context, data any) error

// NewMultimodalAgent creates a new multimodal agent.
func NewMultimodalAgent(name string, registry *llm.ProviderRegistry, opts ...AgentOption) *MultimodalAgent {
	config := DefaultAgentConfig()

	agent := &MultimodalAgent{
		Name:           name,
		Registry:       registry,
		Router:         llm.NewSmartRouter(registry),
		Conversation:   content.NewConversation(fmt.Sprintf("%s-%d", name, time.Now().UnixNano())),
		Config:         config,
		Preprocessors:  make([]ContentPreprocessor, 0),
		Postprocessors: make([]ResponsePostprocessor, 0),
		Tools:          make([]llm.Tool, 0),
		EventHandlers:  make(map[AgentEvent][]EventHandler),
	}

	for _, opt := range opts {
		opt(agent)
	}

	return agent
}

// AgentOption configures the agent.
type AgentOption func(*MultimodalAgent)

// WithSystemPrompt sets the system prompt.
func WithSystemPrompt(prompt string) AgentOption {
	return func(a *MultimodalAgent) {
		a.SystemPrompt = prompt
	}
}

// WithConfig sets the agent configuration.
func WithConfig(config *AgentConfig) AgentOption {
	return func(a *MultimodalAgent) {
		a.Config = config
	}
}

// WithDomainTrainer sets the domain trainer for RAG.
func WithDomainTrainer(trainer *training.DomainTrainer) AgentOption {
	return func(a *MultimodalAgent) {
		a.DomainTrainer = trainer
	}
}

// WithTools adds tools to the agent.
func WithTools(tools ...llm.Tool) AgentOption {
	return func(a *MultimodalAgent) {
		a.Tools = append(a.Tools, tools...)
	}
}

// WithPreprocessor adds a content preprocessor.
func WithPreprocessor(p ContentPreprocessor) AgentOption {
	return func(a *MultimodalAgent) {
		a.Preprocessors = append(a.Preprocessors, p)
	}
}

// WithPostprocessor adds a response postprocessor.
func WithPostprocessor(p ResponsePostprocessor) AgentOption {
	return func(a *MultimodalAgent) {
		a.Postprocessors = append(a.Postprocessors, p)
	}
}

// Chat sends a text message and returns a response.
func (a *MultimodalAgent) Chat(ctx context.Context, input string) (*llm.GenerationResponse, error) {
	msg := content.NewUserMessage(input)
	return a.Send(ctx, msg)
}

// ChatWithImage sends a text message with an image.
func (a *MultimodalAgent) ChatWithImage(ctx context.Context, text, imagePath string) (*llm.GenerationResponse, error) {
	msg := content.NewUserMessage(text)
	if _, err := msg.AddImageFromFile(imagePath); err != nil {
		return nil, fmt.Errorf("failed to add image: %w", err)
	}
	return a.Send(ctx, msg)
}

// ChatWithAudio sends a message with audio for analysis.
func (a *MultimodalAgent) ChatWithAudio(ctx context.Context, text, audioPath string) (*llm.GenerationResponse, error) {
	msg := content.NewUserMessage(text)
	if _, err := msg.AddAudioFromFile(audioPath); err != nil {
		return nil, fmt.Errorf("failed to add audio: %w", err)
	}
	return a.Send(ctx, msg)
}

// ChatWithVideo sends a message with video for analysis.
func (a *MultimodalAgent) ChatWithVideo(ctx context.Context, text, videoPath string) (*llm.GenerationResponse, error) {
	msg := content.NewUserMessage(text)
	if _, err := msg.AddVideoFromFile(videoPath); err != nil {
		return nil, fmt.Errorf("failed to add video: %w", err)
	}
	return a.Send(ctx, msg)
}

// ChatWithDocument sends a message with a document.
func (a *MultimodalAgent) ChatWithDocument(ctx context.Context, text, docPath string) (*llm.GenerationResponse, error) {
	msg := content.NewUserMessage(text)
	if _, err := msg.AddDocumentFromFile(docPath); err != nil {
		return nil, fmt.Errorf("failed to add document: %w", err)
	}
	return a.Send(ctx, msg)
}

// ChatMultimodal sends a message with multiple content types.
func (a *MultimodalAgent) ChatMultimodal(ctx context.Context, contents ...*content.Content) (*llm.GenerationResponse, error) {
	msg := content.NewMultimodalMessage(content.RoleUser, contents...)
	return a.Send(ctx, msg)
}

// Send sends a message and returns a response.
func (a *MultimodalAgent) Send(ctx context.Context, msg *content.Message) (*llm.GenerationResponse, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	// Apply timeout
	if a.Config.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, a.Config.Timeout)
		defer cancel()
	}

	// Preprocess content
	if a.Config.AutoPreprocess {
		if err := a.preprocessMessage(ctx, msg); err != nil {
			a.emit(ctx, EventError, err)
			return nil, fmt.Errorf("preprocessing failed: %w", err)
		}
	}

	// Build messages
	messages := a.buildMessages(ctx, msg)

	// DEBUG LOG
	fmt.Printf("\n[DEBUG] Agent '%s' sending %d messages to model '%s'\n", a.Name, len(messages), a.Config.DefaultModel)
	for i, m := range messages {
		fmt.Printf("  Message %d (%s): %s\n", i, m.Role, truncateText(m.GetText(), 200))
	}

	// Emit before generate event
	a.emit(ctx, EventBeforeGenerate, messages)

	// Build generation config
	genConfig := &llm.GenerationConfig{
		Model:       a.Config.DefaultModel,
		Temperature: a.Config.Temperature,
		MaxTokens:   a.Config.MaxTokens,
		TopP:        a.Config.TopP,
	}

	if a.Config.EnableTools && len(a.Tools) > 0 {
		genConfig.Tools = a.Tools
	}

	// Determine requirements based on content
	requirements := a.buildRequirements(msg)

	// Route and generate
	response, err := a.Router.Route(ctx, messages, genConfig, requirements)
	if err != nil {
		a.emit(ctx, EventError, err)
		return nil, fmt.Errorf("generation failed: %w", err)
	}

	// Apply postprocessors
	for _, pp := range a.Postprocessors {
		response, err = pp.Process(ctx, response)
		if err != nil {
			return nil, fmt.Errorf("postprocessing failed: %w", err)
		}
	}

	// Update conversation
	a.Conversation.AddMessage(msg)
	a.Conversation.AddMessage(response.Message)

	// Trim conversation if too long
	a.trimConversation()

	// Emit after generate event
	a.emit(ctx, EventAfterGenerate, response)

	return response, nil
}

// Stream sends a message and streams the response.
func (a *MultimodalAgent) Stream(ctx context.Context, msg *content.Message) (<-chan llm.StreamChunk, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	// Preprocess content
	if a.Config.AutoPreprocess {
		if err := a.preprocessMessage(ctx, msg); err != nil {
			return nil, fmt.Errorf("preprocessing failed: %w", err)
		}
	}

	// Build messages
	messages := a.buildMessages(ctx, msg)

	// Build generation config
	genConfig := &llm.GenerationConfig{
		Model:       a.Config.DefaultModel,
		Temperature: a.Config.Temperature,
		MaxTokens:   a.Config.MaxTokens,
		TopP:        a.Config.TopP,
		Stream:      true,
	}

	// Determine requirements
	requirements := a.buildRequirements(msg)

	// Route and stream
	ch, _, err := a.Router.RouteStream(ctx, messages, genConfig, requirements)
	if err != nil {
		return nil, fmt.Errorf("streaming failed: %w", err)
	}

	// Wrap channel to update conversation
	outCh := make(chan llm.StreamChunk, 100)
	go func() {
		defer close(outCh)
		var fullResponse string
		for chunk := range ch {
			outCh <- chunk
			fullResponse += chunk.Delta
			if chunk.FinishReason != "" {
				// Update conversation with complete response
				a.mu.Lock()
				a.Conversation.AddMessage(msg)
				a.Conversation.AddMessage(content.NewAssistantMessage(fullResponse))
				a.trimConversation()
				a.mu.Unlock()
			}
		}
	}()

	return outCh, nil
}

// TranscribeAudio transcribes audio content.
func (a *MultimodalAgent) TranscribeAudio(ctx context.Context, audioPath string) (*llm.TranscriptionResponse, error) {
	audioContent, err := content.NewAudioFromFile(audioPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load audio: %w", err)
	}

	return a.Router.RouteTranscribe(ctx, &llm.TranscriptionRequest{
		Audio:      audioContent,
		Timestamps: true,
	})
}

// GenerateImage generates an image from a prompt.
func (a *MultimodalAgent) GenerateImage(ctx context.Context, prompt string) (*llm.ImageGenerationResponse, error) {
	return a.Router.RouteImageGen(ctx, &llm.ImageGenerationRequest{
		Prompt: prompt,
	})
}

// GenerateSpeech generates speech from text.
func (a *MultimodalAgent) GenerateSpeech(ctx context.Context, text, voice string) (*llm.SpeechResponse, error) {
	return a.Router.RouteSpeech(ctx, &llm.SpeechRequest{
		Input: text,
		Voice: voice,
	})
}

// GetEmbedding generates embeddings for text.
func (a *MultimodalAgent) GetEmbedding(ctx context.Context, text string) ([]float64, error) {
	resp, err := a.Router.RouteEmbed(ctx, &llm.EmbeddingRequest{
		Input: []string{text},
	})
	if err != nil {
		return nil, err
	}
	if len(resp.Embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}
	return resp.Embeddings[0], nil
}

// SetDomain sets the domain for domain-specific responses.
func (a *MultimodalAgent) SetDomain(domainID string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.Config.DomainID = domainID
}

// ClearConversation clears the conversation history.
func (a *MultimodalAgent) ClearConversation() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.Conversation.Clear()
}

// OnEvent registers an event handler.
func (a *MultimodalAgent) OnEvent(event AgentEvent, handler EventHandler) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.EventHandlers[event] = append(a.EventHandlers[event], handler)
}

// Helper methods

func (a *MultimodalAgent) buildMessages(ctx context.Context, userMsg *content.Message) []*content.Message {
	var messages []*content.Message

	// Add system prompt
	systemPrompt := a.SystemPrompt

	// Add domain context if enabled
	if a.Config.EnableRAG && a.DomainTrainer != nil && a.Config.DomainID != "" {
		a.emit(ctx, EventRAGSearch, userMsg.GetText())
		domainPrompt, err := a.DomainTrainer.BuildSystemPrompt(ctx, a.Config.DomainID, userMsg.GetText())
		if err == nil && domainPrompt != "" {
			if systemPrompt != "" {
				systemPrompt = systemPrompt + "\n\n" + domainPrompt
			} else {
				systemPrompt = domainPrompt
			}
		}
	}

	if systemPrompt != "" {
		messages = append(messages, content.NewSystemMessage(systemPrompt))
	}

	// Add conversation history
	for _, msg := range a.Conversation.Messages {
		messages = append(messages, msg)
	}

	// Add current user message
	messages = append(messages, userMsg)

	return messages
}

func (a *MultimodalAgent) buildRequirements(msg *content.Message) *llm.ModelRequirements {
	req := &llm.ModelRequirements{
		Speed: llm.SpeedBalanced,
	}

	// Detect required capabilities
	for _, c := range msg.Contents {
		switch c.Type {
		case content.TypeImage:
			req.Capabilities = append(req.Capabilities, llm.CapVision)
		case content.TypeAudio:
			req.Capabilities = append(req.Capabilities, llm.CapAudio)
		case content.TypeVideo:
			req.Capabilities = append(req.Capabilities, llm.CapVideo)
		case content.TypeDocument:
			req.Capabilities = append(req.Capabilities, llm.CapDocument)
		case content.TypeCode:
			req.Capabilities = append(req.Capabilities, llm.CapCodeGeneration)
			req.TaskType = llm.TaskCoding
		}
	}

	// Check preferred model
	if a.Config.DefaultModel != "" {
		req.PreferredModel = a.Config.DefaultModel
	} else if len(msg.Contents) > 0 {
		ct := msg.Contents[0].Type
		if preferred, ok := a.Config.PreferredModels[ct]; ok {
			req.PreferredModel = preferred
		}
	}

	// Set domain if configured
	if a.Config.DomainID != "" {
		req.Domain = a.Config.DomainID
	}

	return req
}

func (a *MultimodalAgent) preprocessMessage(ctx context.Context, msg *content.Message) error {
	a.emit(ctx, EventPreprocess, msg)

	for i, c := range msg.Contents {
		for _, pp := range a.Preprocessors {
			if pp.CanProcess(c) {
				processed, err := pp.Process(ctx, c)
				if err != nil {
					return err
				}
				msg.Contents[i] = processed
			}
		}
	}
	return nil
}

func (a *MultimodalAgent) trimConversation() {
	if a.Config.MaxConversationLength > 0 && len(a.Conversation.Messages) > a.Config.MaxConversationLength {
		// Keep the most recent messages
		start := len(a.Conversation.Messages) - a.Config.MaxConversationLength
		a.Conversation.Messages = a.Conversation.Messages[start:]
	}
}

func (a *MultimodalAgent) emit(ctx context.Context, event AgentEvent, data any) {
	handlers, ok := a.EventHandlers[event]
	if !ok {
		return
	}
	for _, handler := range handlers {
		handler(ctx, data)
	}
}

// AudioTranscriptionPreprocessor automatically transcribes audio.
type AudioTranscriptionPreprocessor struct {
	Router *llm.SmartRouter
}

// CanProcess checks if this preprocessor can handle the content.
func (p *AudioTranscriptionPreprocessor) CanProcess(c *content.Content) bool {
	return c.Type == content.TypeAudio && c.RequiresTranscription()
}

// Process transcribes the audio and converts to text.
func (p *AudioTranscriptionPreprocessor) Process(ctx context.Context, c *content.Content) (*content.Content, error) {
	resp, err := p.Router.RouteTranscribe(ctx, &llm.TranscriptionRequest{
		Audio: c,
	})
	if err != nil {
		return nil, err
	}

	// Return text content with transcription
	return &content.Content{
		Type: content.TypeText,
		Text: fmt.Sprintf("[Transcribed Audio]\n%s", resp.Text),
		Metadata: map[string]any{
			"original_type": content.TypeAudio,
			"duration":      resp.Duration,
			"language":      resp.Language,
		},
	}, nil
}

// UsageTracker tracks usage and costs.
type UsageTracker struct {
	mu           sync.Mutex
	TotalInput   int
	TotalOutput  int
	TotalCost    float64
	RequestCount int
	ModelUsage   map[string]*ModelUsage
}

// ModelUsage tracks usage for a specific model.
type ModelUsage struct {
	InputTokens  int
	OutputTokens int
	Cost         float64
	Requests     int
}

// NewUsageTracker creates a new usage tracker.
func NewUsageTracker() *UsageTracker {
	return &UsageTracker{
		ModelUsage: make(map[string]*ModelUsage),
	}
}

// Track tracks a generation response.
func (t *UsageTracker) Track(response *llm.GenerationResponse) {
	if response == nil || response.Usage == nil {
		return
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	t.TotalInput += response.Usage.InputTokens
	t.TotalOutput += response.Usage.OutputTokens
	t.TotalCost += response.Usage.TotalCost
	t.RequestCount++

	model := response.Model
	if _, ok := t.ModelUsage[model]; !ok {
		t.ModelUsage[model] = &ModelUsage{}
	}
	t.ModelUsage[model].InputTokens += response.Usage.InputTokens
	t.ModelUsage[model].OutputTokens += response.Usage.OutputTokens
	t.ModelUsage[model].Cost += response.Usage.TotalCost
	t.ModelUsage[model].Requests++
}

// GetStats returns current usage statistics.
func (t *UsageTracker) GetStats() map[string]any {
	t.mu.Lock()
	defer t.mu.Unlock()

	return map[string]any{
		"total_input_tokens":  t.TotalInput,
		"total_output_tokens": t.TotalOutput,
		"total_cost":          t.TotalCost,
		"request_count":       t.RequestCount,
		"model_usage":         t.ModelUsage,
	}
}

func truncateText(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
