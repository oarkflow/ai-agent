package memory

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/oarkflow/ai-agent/pkg/content"
	"github.com/oarkflow/ai-agent/pkg/llm"
	"github.com/oarkflow/ai-agent/pkg/training"
)

// mockProvider implements a minimal MultimodalProvider for tests.
type mockProvider struct {
	resp      string
	genCalled chan struct{}
}

func (m *mockProvider) Generate(ctx context.Context, messages []*content.Message, config *llm.GenerationConfig) (*llm.GenerationResponse, error) {
	if m.genCalled != nil {
		select {
		case m.genCalled <- struct{}{}:
		default:
		}
	}
	return &llm.GenerationResponse{Message: content.NewAssistantMessage(m.resp)}, nil
}

// mockProviderWithEmbed wraps mockProvider and allows custom Embed behavior.
type mockProviderWithEmbed struct {
	*mockProvider
	embedFn func(ctx context.Context, req *llm.EmbeddingRequest) (*llm.EmbeddingResponse, error)
}

func (m *mockProviderWithEmbed) Embed(ctx context.Context, req *llm.EmbeddingRequest) (*llm.EmbeddingResponse, error) {
	if m.embedFn != nil {
		return m.embedFn(ctx, req)
	}
	return &llm.EmbeddingResponse{Embeddings: [][]float64{{0.0}}}, nil
}

func (m *mockProvider) GenerateStream(ctx context.Context, messages []*content.Message, config *llm.GenerationConfig) (<-chan llm.StreamChunk, error) {
	return nil, fmt.Errorf("not implemented")
}
func (m *mockProvider) Embed(ctx context.Context, req *llm.EmbeddingRequest) (*llm.EmbeddingResponse, error) {
	return &llm.EmbeddingResponse{Embeddings: [][]float64{{0.0}}}, nil
}
func (m *mockProvider) Transcribe(ctx context.Context, req *llm.TranscriptionRequest) (*llm.TranscriptionResponse, error) {
	return nil, fmt.Errorf("not implemented")
}
func (m *mockProvider) GenerateImage(ctx context.Context, req *llm.ImageGenerationRequest) (*llm.ImageGenerationResponse, error) {
	return nil, fmt.Errorf("not implemented")
}
func (m *mockProvider) GenerateSpeech(ctx context.Context, req *llm.SpeechRequest) (*llm.SpeechResponse, error) {
	return nil, fmt.Errorf("not implemented")
}
func (m *mockProvider) GetProviderType() llm.ProviderType { return "mock" }
func (m *mockProvider) GetModelInfo(model string) (*llm.ModelInfo, error) {
	return nil, fmt.Errorf("not implemented")
}
func (m *mockProvider) ListModels(ctx context.Context) ([]llm.ModelInfo, error) {
	return nil, fmt.Errorf("not implemented")
}
func (m *mockProvider) GetCapabilities() []llm.Capability          { return nil }
func (m *mockProvider) SupportsCapability(cap llm.Capability) bool { return false }

func TestApplySummaryStrategy_WithProvider(t *testing.T) {
	cfg := &MemoryConfig{MaxMessages: 3}
	cm := NewConversationMemory(cfg)
	mock := &mockProvider{resp: "This is the summary", genCalled: make(chan struct{}, 1)}
	cm.SetSummaryProvider(mock)

	cm.Add(content.NewSystemMessage("system prompt"))
	// Add messages to exceed max
	for i := 0; i < 6; i++ {
		cm.Add(content.NewUserMessage(fmt.Sprintf("msg %d", i)))
	}

	// Wait for Generate to be called
	select {
	case <-mock.genCalled:
	case <-time.After(2 * time.Second):
		t.Fatalf("summary provider not called")
	}

	// Wait for summary replacement
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()
	timeout := time.After(2 * time.Second)
	for {
		select {
		case <-timeout:
			t.Fatalf("summary not inserted in time")
		case <-ticker.C:
			msgs := cm.Get()
			for _, m := range msgs {
				if m.Role == content.RoleSystem && strings.HasPrefix(m.GetText(), "Summary of earlier conversation") {
					if !strings.Contains(m.GetText(), "This is the summary") {
						t.Fatalf("unexpected summary text: %s", m.GetText())
					}
					return
				}
			}
		}
	}
}

func TestApplySummaryStrategy_NoProviderFallsBack(t *testing.T) {
	cfg := &MemoryConfig{MaxMessages: 3}
	cm := NewConversationMemory(cfg)
	cm.SetSummaryProvider(nil)
	cm.Add(content.NewSystemMessage("system prompt"))
	for i := 0; i < 5; i++ {
		cm.Add(content.NewUserMessage(fmt.Sprintf("msg %d", i)))
	}
	msgs := cm.Get()
	if len(msgs) > cfg.MaxMessages {
		t.Fatalf("expected messages length <= %d, got %d", cfg.MaxMessages, len(msgs))
	}
}

func TestApplyImportanceStrategy_WithScorer(t *testing.T) {
	cfg := &MemoryConfig{MaxMessages: 3, Strategy: StrategyImportance}
	cm := NewConversationMemory(cfg)

	// Provide a scorer that marks messages containing 'keep' as important
	cm.SetMessageScorer(MessageScorerFunc(func(ctx context.Context, msg *content.Message) float64 {
		_ = ctx
		if strings.Contains(msg.GetText(), "keep") {
			return 1.0
		}
		return 0.1
	}))
	cm.Add(content.NewSystemMessage("system"))
	cm.Add(content.NewUserMessage("keep this 1"))
	cm.Add(content.NewUserMessage("remove this"))
	cm.Add(content.NewUserMessage("keep this 2"))
	cm.Add(content.NewUserMessage("another"))
	// After additions trimming should have happened
	msgs := cm.Get()
	if len(msgs) > cfg.MaxMessages {
		t.Fatalf("expected messages length <= %d, got %d", cfg.MaxMessages, len(msgs))
	}
	// Ensure both 'keep' messages are present
	foundKeep1, foundKeep2 := false, false
	for _, m := range msgs {
		if strings.Contains(m.GetText(), "keep this 1") {
			foundKeep1 = true
		}
		if strings.Contains(m.GetText(), "keep this 2") {
			foundKeep2 = true
		}
	}
	if !foundKeep1 || !foundKeep2 {
		t.Fatalf("important messages were not preserved: %v %v", foundKeep1, foundKeep2)
	}
}

func TestSemanticMemory_StoreAndRecall(t *testing.T) {
	// Use training in-memory vector store and adapter
	vs := training.NewInMemoryVectorStore()
	adapter := NewTrainingVectorStoreAdapter(vs)
	mock := &mockProvider{resp: "", genCalled: nil}
	// Make mock embed return deterministic embeddings based on content length
	mockEmbed := func(ctx context.Context, req *llm.EmbeddingRequest) (*llm.EmbeddingResponse, error) {
		emb := make([]float64, 3)
		for _, s := range req.Input {
			emb[0] = float64(len(s))
			emb[1] = float64(len(s)) / 2
			emb[2] = 1.0
			// return same embedding for single-item input
			return &llm.EmbeddingResponse{Embeddings: [][]float64{emb}}, nil
		}
		return nil, fmt.Errorf("no input")
	}
	mockEmbedProvider := &mockProviderWithEmbed{mockProvider: mock, embedFn: mockEmbed}
	sm := NewSemanticMemory(mockEmbedProvider, adapter)
	// Store a memory
	mem, err := sm.Store(context.Background(), "apple pie is tasty", MemoryFact, map[string]any{"tag": "food"})
	if err != nil {
		t.Fatalf("store failed: %v", err)
	}
	if mem == nil {
		t.Fatalf("expected memory item")
	}
	// Recall with related query
	results, err := sm.Recall(context.Background(), "apple", 5)
	if err != nil {
		t.Fatalf("recall failed: %v", err)
	}
	if len(results) == 0 {
		t.Fatalf("expected recall to return at least one memory")
	}
}
