package memory

import (
	"context"
	"strings"
	"testing"

	"github.com/oarkflow/ai-agent/pkg/content"
	"github.com/oarkflow/ai-agent/pkg/llm"
)

// mockForLLMScorer returns simple numeric strings based on message content.
type mockScorerProvider struct{}

func (m *mockScorerProvider) Generate(ctx context.Context, messages []*content.Message, config *llm.GenerationConfig) (*llm.GenerationResponse, error) {
	text := messages[len(messages)-1].GetText()
	if strings.Contains(text, "keep") {
		return &llm.GenerationResponse{Message: content.NewAssistantMessage("0.95")}, nil
	}
	return &llm.GenerationResponse{Message: content.NewAssistantMessage("0.05")}, nil
}

func (m *mockScorerProvider) GenerateStream(ctx context.Context, messages []*content.Message, config *llm.GenerationConfig) (<-chan llm.StreamChunk, error) {
	return nil, nil
}
func (m *mockScorerProvider) Embed(ctx context.Context, req *llm.EmbeddingRequest) (*llm.EmbeddingResponse, error) {
	return &llm.EmbeddingResponse{Embeddings: [][]float64{{0}}}, nil
}
func (m *mockScorerProvider) Transcribe(ctx context.Context, req *llm.TranscriptionRequest) (*llm.TranscriptionResponse, error) {
	return nil, nil
}
func (m *mockScorerProvider) GenerateImage(ctx context.Context, req *llm.ImageGenerationRequest) (*llm.ImageGenerationResponse, error) {
	return nil, nil
}
func (m *mockScorerProvider) GenerateSpeech(ctx context.Context, req *llm.SpeechRequest) (*llm.SpeechResponse, error) {
	return nil, nil
}
func (m *mockScorerProvider) GetProviderType() llm.ProviderType                 { return "mock" }
func (m *mockScorerProvider) GetModelInfo(model string) (*llm.ModelInfo, error) { return nil, nil }
func (m *mockScorerProvider) ListModels(ctx context.Context) ([]llm.ModelInfo, error) {
	return nil, nil
}
func (m *mockScorerProvider) GetCapabilities() []llm.Capability          { return nil }
func (m *mockScorerProvider) SupportsCapability(cap llm.Capability) bool { return false }

func TestLLMScorer_Scores(t *testing.T) {
	prov := &mockScorerProvider{}
	s := NewLLMScorer(prov, "test-model")
	if v := s.Score(context.Background(), content.NewUserMessage("keep")); v < 0.9 {
		t.Fatalf("expected high score, got %v", v)
	}
	if v := s.Score(context.Background(), content.NewUserMessage("ignore")); v > 0.2 {
		t.Fatalf("expected low score, got %v", v)
	}
}
