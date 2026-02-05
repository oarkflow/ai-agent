package memory

import (
	"context"
	"testing"

	"github.com/oarkflow/ai-agent/pkg/content"
	"github.com/oarkflow/ai-agent/pkg/llm"
)

type countingProvider struct {
	calls int
}

func (p *countingProvider) Generate(ctx context.Context, messages []*content.Message, config *llm.GenerationConfig) (*llm.GenerationResponse, error) {
	p.calls++
	return &llm.GenerationResponse{Message: content.NewAssistantMessage("0.5")}, nil
}
func (p *countingProvider) GenerateStream(ctx context.Context, messages []*content.Message, config *llm.GenerationConfig) (<-chan llm.StreamChunk, error) {
	return nil, nil
}
func (p *countingProvider) Embed(ctx context.Context, req *llm.EmbeddingRequest) (*llm.EmbeddingResponse, error) {
	return &llm.EmbeddingResponse{Embeddings: [][]float64{{0}}}, nil
}
func (p *countingProvider) Transcribe(ctx context.Context, req *llm.TranscriptionRequest) (*llm.TranscriptionResponse, error) {
	return nil, nil
}
func (p *countingProvider) GenerateImage(ctx context.Context, req *llm.ImageGenerationRequest) (*llm.ImageGenerationResponse, error) {
	return nil, nil
}
func (p *countingProvider) GenerateSpeech(ctx context.Context, req *llm.SpeechRequest) (*llm.SpeechResponse, error) {
	return nil, nil
}
func (p *countingProvider) GetProviderType() llm.ProviderType                       { return "mock" }
func (p *countingProvider) GetModelInfo(model string) (*llm.ModelInfo, error)       { return nil, nil }
func (p *countingProvider) ListModels(ctx context.Context) ([]llm.ModelInfo, error) { return nil, nil }
func (p *countingProvider) GetCapabilities() []llm.Capability                       { return nil }
func (p *countingProvider) SupportsCapability(cap llm.Capability) bool              { return false }

func TestLLMScorerCache(t *testing.T) {
	p := &countingProvider{}
	scorer := NewLLMScorer(p, "test-model")
	cache := NewLLMScorerCache(scorer)

	msg := content.NewUserMessage("important note")
	v1 := cache.Score(context.Background(), msg)
	v2 := cache.Score(context.Background(), msg)
	if v1 != v2 {
		t.Fatalf("expected cached values to match: %v != %v", v1, v2)
	}
	if p.calls != 1 {
		t.Fatalf("expected provider to be called once, called %d times", p.calls)
	}

	// different message should trigger new call
	msg2 := content.NewUserMessage("another message")
	cache.Score(context.Background(), msg2)
	if p.calls != 2 {
		t.Fatalf("expected provider to be called twice, called %d", p.calls)
	}
}
