package memory

import (
	"context"
	"fmt"

	"github.com/sujit/ai-agent/pkg/training"
)

// TrainingVectorStoreAdapter adapts a training.VectorStore to the memory.VectorStore interface.
type TrainingVectorStoreAdapter struct {
	store training.VectorStore
}

func NewTrainingVectorStoreAdapter(store training.VectorStore) *TrainingVectorStoreAdapter {
	return &TrainingVectorStoreAdapter{store: store}
}

func (a *TrainingVectorStoreAdapter) Store(id string, embedding []float64, metadata map[string]any) error {
	if a.store == nil {
		return fmt.Errorf("underlying store is nil")
	}
	return a.store.Store(context.Background(), id, embedding, metadata)
}

func (a *TrainingVectorStoreAdapter) Search(embedding []float64, limit int) ([]SearchResult, error) {
	if a.store == nil {
		return nil, fmt.Errorf("underlying store is nil")
	}
	results, err := a.store.Search(context.Background(), embedding, limit, nil)
	if err != nil {
		return nil, err
	}
	out := make([]SearchResult, 0, len(results))
	for _, r := range results {
		out = append(out, SearchResult{
			ID:       r.ID,
			Score:    r.Score,
			Metadata: r.Metadata,
		})
	}
	return out, nil
}

func (a *TrainingVectorStoreAdapter) Delete(id string) error {
	if a.store == nil {
		return fmt.Errorf("underlying store is nil")
	}
	return a.store.Delete(context.Background(), id)
}
