package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/sujit/ai-agent/pkg/content"
	"github.com/sujit/ai-agent/pkg/llm"
)

// ConversationMemory manages conversation history with various strategies.
type ConversationMemory struct {
	mu              sync.RWMutex
	messages        []*content.Message
	maxMessages     int
	maxTokens       int
	strategy        MemoryStrategy
	summaryProvider llm.MultimodalProvider
	// Extensible components
	tokenEstimator TokenEstimator
	scorer         MessageScorer
	// Use cache for LLM-based scorers
	useScorerCache bool

	metadata *ConversationMetadata
}

// ConversationMetadata stores metadata about the conversation.
type ConversationMetadata struct {
	ID           string            `json:"id"`
	StartedAt    time.Time         `json:"started_at"`
	UpdatedAt    time.Time         `json:"updated_at"`
	MessageCount int               `json:"message_count"`
	TotalTokens  int               `json:"total_tokens"`
	Topics       []string          `json:"topics"`
	Entities     map[string]string `json:"entities"`
	CustomData   map[string]any    `json:"custom_data"`
}

// MemoryStrategy defines how to handle memory limits.
type MemoryStrategy string

const (
	// StrategyWindow keeps only the last N messages.
	StrategyWindow MemoryStrategy = "window"
	// StrategySummary summarizes old messages.
	StrategySummary MemoryStrategy = "summary"
	// StrategyImportance keeps important messages.
	StrategyImportance MemoryStrategy = "importance"
	// StrategyHybrid combines window and summary.
	StrategyHybrid MemoryStrategy = "hybrid"
)

// MemoryConfig configures conversation memory.
type MemoryConfig struct {
	MaxMessages      int            `json:"max_messages"`
	MaxTokens        int            `json:"max_tokens"`
	Strategy         MemoryStrategy `json:"strategy"`
	SummaryInterval  int            `json:"summary_interval"`
	KeepSystemPrompt bool           `json:"keep_system_prompt"`
	// UseScorerCache enables caching for LLM-based message scorers
	UseScorerCache bool `json:"use_scorer_cache"`
}

// DefaultMemoryConfig returns a default configuration.
func DefaultMemoryConfig() *MemoryConfig {
	return &MemoryConfig{
		MaxMessages:      50,
		MaxTokens:        100000,
		Strategy:         StrategyHybrid,
		SummaryInterval:  20,
		KeepSystemPrompt: true,
		UseScorerCache:   true,
	}
}

// NewConversationMemory creates a new conversation memory.
func NewConversationMemory(config *MemoryConfig) *ConversationMemory {
	if config == nil {
		config = DefaultMemoryConfig()
	} else {
		// Fill defaults for missing values
		defaultCfg := DefaultMemoryConfig()
		if config.Strategy == "" {
			config.Strategy = defaultCfg.Strategy
		}
		if config.MaxMessages == 0 {
			config.MaxMessages = defaultCfg.MaxMessages
		}
		if config.MaxTokens == 0 {
			config.MaxTokens = defaultCfg.MaxTokens
		}
	}

	cm := &ConversationMemory{
		messages:       make([]*content.Message, 0),
		maxMessages:    config.MaxMessages,
		maxTokens:      config.MaxTokens,
		strategy:       config.Strategy,
		useScorerCache: config.UseScorerCache,
		metadata: &ConversationMetadata{
			ID:         generateID(),
			StartedAt:  time.Now(),
			UpdatedAt:  time.Now(),
			Entities:   make(map[string]string),
			CustomData: make(map[string]any),
		},
	}

	// Default token estimator: prefer tiktoken but fall back to heuristic
	if enc, err := NewTiktokenEstimator("cl100k_base"); err == nil {
		cm.tokenEstimator = enc
	} else {
		cm.tokenEstimator = DefaultTokenEstimator{}
	}

	return cm
}

// SetSummaryProvider sets the provider used for summarization.
// If an importance scorer is not set, the summary provider will be used to
// create an LLM-based importance scorer (and cache it when configured).
func (cm *ConversationMemory) SetSummaryProvider(provider llm.MultimodalProvider) {
	cm.summaryProvider = provider
	// If no scorer configured, wire an LLM scorer using this provider
	if cm.scorer == nil && provider != nil {
		cm.SetImportanceProvider(provider, "")
	}
}

// SetTokenEstimator sets a custom token estimator.
func (cm *ConversationMemory) SetTokenEstimator(est TokenEstimator) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.tokenEstimator = est
}

// SetMessageScorer sets the scorer used by importance strategy.
func (cm *ConversationMemory) SetMessageScorer(scorer MessageScorer) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.scorer = scorer
}

// SetImportanceProvider configures the importance scorer from an LLM provider.
// If useCache is not provided, it uses the memory config default.
func (cm *ConversationMemory) SetImportanceProvider(provider llm.MultimodalProvider, model string) {
	if provider == nil {
		return
	}
	// Create LLM scorer
	llmScorer := NewLLMScorer(provider, model)
	if cm.useScorerCache {
		cm.SetMessageScorer(NewLLMScorerCache(llmScorer))
	} else {
		cm.SetMessageScorer(llmScorer)
	}
}

// estimateTokens is the instance-aware token estimator wrapper.
func (cm *ConversationMemory) estimateTokens(msg *content.Message) int {
	if cm.tokenEstimator != nil {
		return cm.tokenEstimator.EstimateMessage(msg)
	}
	var d DefaultTokenEstimator
	return d.EstimateMessage(msg)
}

// Add adds a message to the conversation.
func (cm *ConversationMemory) Add(msg *content.Message) {
	cm.mu.Lock()
	cm.messages = append(cm.messages, msg)
	cm.metadata.MessageCount++
	cm.metadata.UpdatedAt = time.Now()
	needTrim := cm.shouldTrim()
	cm.mu.Unlock()

	// Apply memory strategy if needed (outside lock to avoid deadlocks with strategies that lock)
	if needTrim {
		cm.applyStrategy()
	}
}

// AddAll adds multiple messages.
func (cm *ConversationMemory) AddAll(msgs []*content.Message) {
	for _, msg := range msgs {
		cm.Add(msg)
	}
}

// Get returns all messages.
func (cm *ConversationMemory) Get() []*content.Message {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	result := make([]*content.Message, len(cm.messages))
	copy(result, cm.messages)
	return result
}

// GetLast returns the last N messages.
func (cm *ConversationMemory) GetLast(n int) []*content.Message {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	if n >= len(cm.messages) {
		result := make([]*content.Message, len(cm.messages))
		copy(result, cm.messages)
		return result
	}

	result := make([]*content.Message, n)
	copy(result, cm.messages[len(cm.messages)-n:])
	return result
}

// GetContext returns messages optimized for LLM context.
func (cm *ConversationMemory) GetContext(maxTokens int) []*content.Message {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	if len(cm.messages) == 0 {
		return nil
	}

	// Start from the most recent and work backwards
	result := make([]*content.Message, 0)
	estimatedTokens := 0

	for i := len(cm.messages) - 1; i >= 0; i-- {
		msg := cm.messages[i]
		msgTokens := cm.estimateTokens(msg)

		if estimatedTokens+msgTokens > maxTokens && len(result) > 0 {
			break
		}

		// Prepend
		result = append([]*content.Message{msg}, result...)
		estimatedTokens += msgTokens
	}

	return result
}

// Clear removes all messages.
func (cm *ConversationMemory) Clear() {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	cm.messages = make([]*content.Message, 0)
	cm.metadata.MessageCount = 0
}

// GetMetadata returns conversation metadata.
func (cm *ConversationMemory) GetMetadata() *ConversationMetadata {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return cm.metadata
}

// SetEntity stores a named entity.
func (cm *ConversationMemory) SetEntity(name, value string) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.metadata.Entities[name] = value
}

// GetEntity retrieves a named entity.
func (cm *ConversationMemory) GetEntity(name string) (string, bool) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	val, ok := cm.metadata.Entities[name]
	return val, ok
}

// SetCustomData stores custom data.
func (cm *ConversationMemory) SetCustomData(key string, value any) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.metadata.CustomData[key] = value
}

// Summarize creates a summary of the conversation.
func (cm *ConversationMemory) Summarize(ctx context.Context) (string, error) {
	if cm.summaryProvider == nil {
		return "", fmt.Errorf("summary provider not set")
	}

	cm.mu.RLock()
	messages := make([]*content.Message, len(cm.messages))
	copy(messages, cm.messages)
	cm.mu.RUnlock()

	if len(messages) == 0 {
		return "", nil
	}

	// Build conversation text
	var conversationText string
	for _, msg := range messages {
		conversationText += fmt.Sprintf("%s: %s\n", msg.Role, msg.GetText())
	}

	summaryPrompt := fmt.Sprintf(`Summarize the following conversation concisely, capturing:
1. Main topics discussed
2. Key decisions or conclusions
3. Important facts or entities mentioned
4. Any pending questions or action items

Conversation:
%s

Provide a clear, structured summary.`, conversationText)

	resp, err := cm.summaryProvider.Generate(ctx, []*content.Message{
		content.NewSystemMessage("You are a conversation summarizer. Create clear, concise summaries."),
		content.NewUserMessage(summaryPrompt),
	}, &llm.GenerationConfig{Temperature: 0.2, MaxTokens: 1000})

	if err != nil {
		return "", fmt.Errorf("failed to generate summary: %w", err)
	}

	return resp.Message.GetText(), nil
}

// Export exports the conversation to JSON.
func (cm *ConversationMemory) Export() ([]byte, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	export := struct {
		Metadata *ConversationMetadata `json:"metadata"`
		Messages []*content.Message    `json:"messages"`
	}{
		Metadata: cm.metadata,
		Messages: cm.messages,
	}

	return json.MarshalIndent(export, "", "  ")
}

// Import imports a conversation from JSON.
func (cm *ConversationMemory) Import(data []byte) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	var imported struct {
		Metadata *ConversationMetadata `json:"metadata"`
		Messages []*content.Message    `json:"messages"`
	}

	if err := json.Unmarshal(data, &imported); err != nil {
		return fmt.Errorf("failed to parse import data: %w", err)
	}

	cm.metadata = imported.Metadata
	cm.messages = imported.Messages
	return nil
}

// shouldTrim checks if memory needs trimming.
func (cm *ConversationMemory) shouldTrim() bool {
	if len(cm.messages) > cm.maxMessages {
		return true
	}
	// Could also check token count
	return false
}

// applyStrategy applies the configured memory strategy.
func (cm *ConversationMemory) applyStrategy() {
	switch cm.strategy {
	case StrategyWindow:
		cm.applyWindowStrategy()
	case StrategySummary:
		cm.applySummaryStrategy()
	case StrategyImportance:
		cm.applyImportanceStrategy()
	case StrategyHybrid:
		cm.applyHybridStrategy()
	}
}

func (cm *ConversationMemory) applyWindowStrategy() {
	if len(cm.messages) > cm.maxMessages {
		// Keep system prompt if present
		var systemPrompt *content.Message
		start := 0
		if len(cm.messages) > 0 && cm.messages[0].Role == content.RoleSystem {
			systemPrompt = cm.messages[0]
			start = 1
		}

		// Calculate how many to keep
		keepCount := cm.maxMessages
		if systemPrompt != nil {
			keepCount--
		}

		// Trim from the beginning (after system prompt)
		if len(cm.messages)-start > keepCount {
			cm.messages = cm.messages[len(cm.messages)-keepCount:]
		}

		// Add system prompt back
		if systemPrompt != nil {
			cm.messages = append([]*content.Message{systemPrompt}, cm.messages...)
		}
	}
}

func (cm *ConversationMemory) applySummaryStrategy() {
	// If no summary provider is configured, fall back to window strategy
	if cm.summaryProvider == nil {
		cm.applyWindowStrategy()
		return
	}

	// If we're within limits, nothing to do
	if len(cm.messages) <= cm.maxMessages {
		return
	}

	// Identify system prompt (if any) and calculate how many recent messages to keep
	var systemPrompt *content.Message
	start := 0
	if len(cm.messages) > 0 && cm.messages[0].Role == content.RoleSystem {
		systemPrompt = cm.messages[0]
		start = 1
	}

	keepCount := cm.maxMessages
	if systemPrompt != nil {
		keepCount--
	}
	if keepCount < 1 {
		keepCount = 1
	}

	if len(cm.messages)-start <= keepCount {
		return
	}

	// Messages to summarize: from start up to the oldest messages that will be kept out
	summarizeEnd := len(cm.messages) - keepCount
	oldMessages := make([]*content.Message, summarizeEnd-start)
	copy(oldMessages, cm.messages[start:summarizeEnd])

	// Insert a placeholder summary message synchronously so context remains compact
	placeholderText := "Summarizing previous conversation..."
	placeholder := content.NewSystemMessage(placeholderText)

	newMsgs := make([]*content.Message, 0, 1+1+keepCount)
	if systemPrompt != nil {
		newMsgs = append(newMsgs, systemPrompt)
	}
	newMsgs = append(newMsgs, placeholder)
	newMsgs = append(newMsgs, cm.messages[summarizeEnd:]...)
	cm.messages = newMsgs

	// Generate the summary asynchronously and replace the placeholder
	go func(old []*content.Message) {
		// Build the text to summarize
		var convText strings.Builder
		for _, m := range old {
			convText.WriteString(fmt.Sprintf("%s: %s\n", m.Role, m.GetText()))
		}

		prompt := fmt.Sprintf(`Summarize the following conversation concisely, capturing:
1. Main topics discussed
2. Key decisions or conclusions
3. Important facts or entities mentioned
4. Any pending questions or action items

Conversation:
%s

Provide a clear, structured summary.`, convText.String())

		resp, err := cm.summaryProvider.Generate(context.Background(), []*content.Message{
			content.NewSystemMessage("You are a conversation summarizer. Create clear, concise summaries."),
			content.NewUserMessage(prompt),
		}, &llm.GenerationConfig{Temperature: 0.2, MaxTokens: 500})

		summaryText := ""
		if err != nil {
			summaryText = fmt.Sprintf("[Summary generation failed: %v]", err)
		} else if resp != nil && resp.Message != nil {
			summaryText = resp.Message.GetText()
		} else {
			summaryText = "[No summary generated]"
		}

		summaryMessage := content.NewSystemMessage("Summary of earlier conversation:\n" + summaryText)

		// Replace placeholder with the final summary
		cm.mu.Lock()
		defer cm.mu.Unlock()
		for i, m := range cm.messages {
			if m.Role == content.RoleSystem && strings.HasPrefix(m.GetText(), placeholderText) {
				cm.messages[i] = summaryMessage
				break
			}
		}
	}(oldMessages)
}

func (cm *ConversationMemory) applyImportanceStrategy() {
	cm.mu.Lock()
	if len(cm.messages) <= cm.maxMessages {
		cm.mu.Unlock()
		return
	}
	// we'll build new messages while locked, but scoring may use network calls, so copy candidates while locked
	var systemPrompt *content.Message
	start := 0
	if len(cm.messages) > 0 && cm.messages[0].Role == content.RoleSystem {
		systemPrompt = cm.messages[0]
		start = 1
	}
	candidates := make([]*content.Message, len(cm.messages[start:]))
	copy(candidates, cm.messages[start:])
	keepCount := cm.maxMessages
	if systemPrompt != nil {
		keepCount--
	}
	if keepCount < 1 {
		keepCount = 1
	}
	cm.mu.Unlock()

	// Score each candidate; allow scorer to use context
	type scored struct {
		idx   int
		score float64
	}

	scores := make([]scored, 0, len(candidates))
	for i, m := range candidates {
		var s float64
		if cm.scorer != nil {
			s = cm.scorer.Score(context.Background(), m)
		} else {
			var h HeuristicScorer
			s = h.Score(context.Background(), m)
		}
		scores = append(scores, scored{idx: i, score: s})
	}

	// Sort by score desc
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	// Pick top keepCount indices
	if keepCount > len(scores) {
		keepCount = len(scores)
	}
	sel := scores[:keepCount]

	// Restore original chronological order among selected
	sort.Slice(sel, func(i, j int) bool {
		return sel[i].idx < sel[j].idx
	})

	// Build new messages: system prompt (if any) + selected
	newMsgs := make([]*content.Message, 0, 1+keepCount)
	if systemPrompt != nil {
		newMsgs = append(newMsgs, systemPrompt)
	}
	for _, s := range sel {
		newMsgs = append(newMsgs, candidates[s.idx])
	}

	cm.mu.Lock()
	cm.messages = newMsgs
	cm.mu.Unlock()
}

func (cm *ConversationMemory) applyHybridStrategy() {
	// Combine window with summarization of older messages
	// Prefer summarization when provider is available, otherwise fall back to window.
	if cm.summaryProvider == nil {
		cm.applyWindowStrategy()
		return
	}
	cm.applySummaryStrategy()
}

// TokenEstimator estimates tokens for messages or text.
type TokenEstimator interface {
	EstimateMessage(msg *content.Message) int
	EstimateText(text string) int
}

// DefaultTokenEstimator uses a simple heuristic (~4 chars per token).
type DefaultTokenEstimator struct{}

func (d DefaultTokenEstimator) EstimateMessage(msg *content.Message) int {
	return d.EstimateText(msg.GetText())
}

func (d DefaultTokenEstimator) EstimateText(text string) int {
	if text == "" {
		return 0
	}
	return len(text) / 4
}

// MessageScorer assigns importance scores to messages (0..1 range, higher is more important).
// The scorer accepts a context to allow LLM-based scorers to make network calls.
type MessageScorer interface {
	Score(ctx context.Context, msg *content.Message) float64
}

// MessageScorerFunc helper to allow using functions as scorers.
type MessageScorerFunc func(ctx context.Context, msg *content.Message) float64

func (f MessageScorerFunc) Score(ctx context.Context, msg *content.Message) float64 {
	return f(ctx, msg)
}

// HeuristicScorer is a simple fallback scorer that prioritizes assistant messages and longer content.
type HeuristicScorer struct{}

func (h HeuristicScorer) Score(ctx context.Context, msg *content.Message) float64 {
	_ = ctx
	text := msg.GetText()
	if text == "" {
		return 0.0
	}
	// base score by length
	score := float64(len(text)) / 100.0
	// boost assistant messages
	if msg.Role == content.RoleAssistant {
		score += 1.0
	}
	// normalize roughly
	if score > 1.0 {
		score = 1.0
	}
	return score
}

func generateID() string {
	return fmt.Sprintf("conv_%d", time.Now().UnixNano())
}

// SemanticMemory provides long-term memory with semantic search.
type SemanticMemory struct {
	mu          sync.RWMutex
	memories    []*MemoryItem
	provider    llm.MultimodalProvider
	vectorStore VectorStore
}

// MemoryItem represents a single memory item.
type MemoryItem struct {
	ID          string         `json:"id"`
	Content     string         `json:"content"`
	Type        MemoryType     `json:"type"`
	Embedding   []float64      `json:"embedding,omitempty"`
	Metadata    map[string]any `json:"metadata"`
	CreatedAt   time.Time      `json:"created_at"`
	AccessCount int            `json:"access_count"`
	LastAccess  time.Time      `json:"last_access"`
	Importance  float64        `json:"importance"`
}

// MemoryType categorizes memories.
type MemoryType string

const (
	MemoryFact       MemoryType = "fact"
	MemoryEvent      MemoryType = "event"
	MemoryPreference MemoryType = "preference"
	MemoryEntity     MemoryType = "entity"
	MemoryProcedure  MemoryType = "procedure"
)

// VectorStore interface for embedding storage.
type VectorStore interface {
	Store(id string, embedding []float64, metadata map[string]any) error
	Search(embedding []float64, limit int) ([]SearchResult, error)
	Delete(id string) error
}

// SearchResult from vector store.
type SearchResult struct {
	ID       string
	Score    float64
	Metadata map[string]any
}

// NewSemanticMemory creates a new semantic memory.
func NewSemanticMemory(provider llm.MultimodalProvider, store VectorStore) *SemanticMemory {
	return &SemanticMemory{
		memories:    make([]*MemoryItem, 0),
		provider:    provider,
		vectorStore: store,
	}
}

// Store stores a memory with embedding.
func (sm *SemanticMemory) Store(ctx context.Context, content string, memType MemoryType, metadata map[string]any) (*MemoryItem, error) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	// Generate embedding
	embResp, err := sm.provider.Embed(ctx, &llm.EmbeddingRequest{
		Input: []string{content},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	if len(embResp.Embeddings) == 0 {
		return nil, fmt.Errorf("no embedding returned")
	}

	memory := &MemoryItem{
		ID:         generateID(),
		Content:    content,
		Type:       memType,
		Embedding:  embResp.Embeddings[0],
		Metadata:   metadata,
		CreatedAt:  time.Now(),
		Importance: 0.5,
	}

	sm.memories = append(sm.memories, memory)

	// Store in vector store
	if sm.vectorStore != nil {
		meta := make(map[string]any)
		for k, v := range metadata {
			meta[k] = v
		}
		meta["type"] = string(memType)
		meta["content"] = content

		if err := sm.vectorStore.Store(memory.ID, memory.Embedding, meta); err != nil {
			return nil, fmt.Errorf("failed to store in vector store: %w", err)
		}
	}

	return memory, nil
}

// Recall retrieves relevant memories.
func (sm *SemanticMemory) Recall(ctx context.Context, query string, limit int) ([]*MemoryItem, error) {
	// Generate query embedding
	embResp, err := sm.provider.Embed(ctx, &llm.EmbeddingRequest{
		Input: []string{query},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to generate query embedding: %w", err)
	}

	if len(embResp.Embeddings) == 0 {
		return nil, fmt.Errorf("no embedding returned")
	}

	// Search vector store
	if sm.vectorStore != nil {
		results, err := sm.vectorStore.Search(embResp.Embeddings[0], limit)
		if err != nil {
			return nil, fmt.Errorf("search failed: %w", err)
		}

		memories := make([]*MemoryItem, 0, len(results))
		sm.mu.RLock()
		for _, r := range results {
			for _, m := range sm.memories {
				if m.ID == r.ID {
					m.AccessCount++
					m.LastAccess = time.Now()
					memories = append(memories, m)
					break
				}
			}
		}
		sm.mu.RUnlock()
		return memories, nil
	}

	return nil, nil
}

// Forget removes a memory.
func (sm *SemanticMemory) Forget(id string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	for i, m := range sm.memories {
		if m.ID == id {
			sm.memories = append(sm.memories[:i], sm.memories[i+1:]...)
			break
		}
	}

	if sm.vectorStore != nil {
		return sm.vectorStore.Delete(id)
	}
	return nil
}

// GetAll returns all memories.
func (sm *SemanticMemory) GetAll() []*MemoryItem {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	result := make([]*MemoryItem, len(sm.memories))
	copy(result, sm.memories)
	return result
}
