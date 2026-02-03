package memory

import (
	"context"
	"fmt"

	"github.com/sujit/ai-agent/pkg/llm"
	"github.com/sujit/ai-agent/pkg/storage"
)

// StorageVectorStoreAdapter adapts storage.PersistentVectorStore to memory.VectorStore.
type StorageVectorStoreAdapter struct {
	store *storage.PersistentVectorStore
}

func NewStorageVectorStoreAdapter(st *storage.Storage, namespace string) (*StorageVectorStoreAdapter, error) {
	vs, err := storage.NewPersistentVectorStore(st, namespace)
	if err != nil {
		return nil, err
	}
	return &StorageVectorStoreAdapter{store: vs}, nil
}

// NewSemanticMemoryWithPersistentStore creates a SemanticMemory backed by persistent storage. (helper kept for convenience)
func NewSemanticMemoryWithPersistentStore(st *storage.Storage, namespace string, provider llm.MultimodalProvider) (*SemanticMemory, error) {
	adapter, err := NewStorageVectorStoreAdapter(st, namespace)
	if err != nil {
		return nil, fmt.Errorf("failed to create storage adapter: %w", err)
	}
	return NewSemanticMemory(provider, adapter), nil
}
func (a *StorageVectorStoreAdapter) Store(id string, embedding []float64, metadata map[string]any) error {
	return a.store.Store(context.Background(), id, embedding, metadata)
}

func (a *StorageVectorStoreAdapter) Search(embedding []float64, limit int) ([]SearchResult, error) {
	results, err := a.store.Search(context.Background(), embedding, limit, nil)
	if err != nil {
		return nil, err
	}
	out := make([]SearchResult, 0, len(results))
	for _, r := range results {
		out = append(out, SearchResult{ID: r.ID, Score: r.Score, Metadata: r.Metadata})
	}
	return out, nil
}

func (a *StorageVectorStoreAdapter) Delete(id string) error {
	return a.store.Delete(context.Background(), id)
}
