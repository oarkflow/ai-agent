package storage

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
)

// PersistentVectorStore implements VectorStore with file-based persistence.
type PersistentVectorStore struct {
	storage   *Storage
	namespace string
	vectors   map[string]*StoredVector
	mu        sync.RWMutex
	dirty     bool
}

// StoredVector represents a stored vector with metadata.
type StoredVector struct {
	ID        string         `json:"id"`
	Embedding []float64      `json:"embedding"`
	Metadata  map[string]any `json:"metadata"`
}

// VectorSearchResult represents a search result.
type VectorSearchResult struct {
	ID       string         `json:"id"`
	Score    float64        `json:"score"`
	Content  string         `json:"content,omitempty"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// NewPersistentVectorStore creates a new persistent vector store.
func NewPersistentVectorStore(storage *Storage, namespace string) (*PersistentVectorStore, error) {
	vs := &PersistentVectorStore{
		storage:   storage,
		namespace: namespace,
		vectors:   make(map[string]*StoredVector),
	}

	// Load existing vectors
	if err := vs.load(); err != nil {
		// Ignore if file doesn't exist
		if !os.IsNotExist(err) {
			return nil, err
		}
	}

	return vs, nil
}

// getFilePath returns the storage path for this namespace.
func (vs *PersistentVectorStore) getFilePath() string {
	return filepath.Join("embeddings", vs.namespace+".json")
}

// load loads vectors from disk.
func (vs *PersistentVectorStore) load() error {
	var vectors []*StoredVector
	if err := vs.storage.LoadJSON(vs.getFilePath(), &vectors); err != nil {
		return err
	}

	vs.mu.Lock()
	defer vs.mu.Unlock()

	for _, v := range vectors {
		vs.vectors[v.ID] = v
	}
	return nil
}

// save persists vectors to disk.
func (vs *PersistentVectorStore) save() error {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	vectors := make([]*StoredVector, 0, len(vs.vectors))
	for _, v := range vs.vectors {
		vectors = append(vectors, v)
	}

	return vs.storage.SaveJSON(vs.getFilePath(), vectors)
}

// Store stores an embedding with metadata.
func (vs *PersistentVectorStore) Store(ctx context.Context, id string, embedding []float64, metadata map[string]any) error {
	vs.mu.Lock()
	defer vs.mu.Unlock()

	vs.vectors[id] = &StoredVector{
		ID:        id,
		Embedding: normalize(embedding),
		Metadata:  metadata,
	}
	vs.dirty = true

	// Auto-save after each store (can be optimized with batching)
	go vs.save()
	return nil
}

// Search performs similarity search.
func (vs *PersistentVectorStore) Search(ctx context.Context, query []float64, topK int, filter map[string]any) ([]VectorSearchResult, error) {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	query = normalize(query)

	type scored struct {
		id       string
		score    float64
		metadata map[string]any
	}

	var results []scored

	for _, v := range vs.vectors {
		// Apply metadata filter
		if filter != nil {
			match := true
			for key, value := range filter {
				if mv, ok := v.Metadata[key]; !ok || mv != value {
					match = false
					break
				}
			}
			if !match {
				continue
			}
		}

		similarity := cosineSimilarity(query, v.Embedding)
		results = append(results, scored{
			id:       v.ID,
			score:    similarity,
			metadata: v.Metadata,
		})
	}

	// Sort by similarity DESC
	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	if topK > 0 && len(results) > topK {
		results = results[:topK]
	}

	// Convert to VectorSearchResult
	vectorResults := make([]VectorSearchResult, len(results))
	for i, r := range results {
		vectorResults[i] = VectorSearchResult{
			ID:       r.id,
			Score:    r.score,
			Metadata: r.metadata,
		}
		if contentVal, ok := r.metadata["content"].(string); ok {
			vectorResults[i].Content = contentVal
		}
	}

	return vectorResults, nil
}

// Delete removes an embedding by ID.
func (vs *PersistentVectorStore) Delete(ctx context.Context, id string) error {
	vs.mu.Lock()
	defer vs.mu.Unlock()

	delete(vs.vectors, id)
	vs.dirty = true
	go vs.save()
	return nil
}

// Clear removes all embeddings.
func (vs *PersistentVectorStore) Clear(ctx context.Context) error {
	vs.mu.Lock()
	defer vs.mu.Unlock()

	vs.vectors = make(map[string]*StoredVector)
	vs.dirty = true
	return vs.save()
}

// Size returns the number of stored vectors.
func (vs *PersistentVectorStore) Size() int {
	vs.mu.RLock()
	defer vs.mu.RUnlock()
	return len(vs.vectors)
}

// Flush forces a save to disk.
func (vs *PersistentVectorStore) Flush() error {
	if vs.dirty {
		vs.dirty = false
		return vs.save()
	}
	return nil
}

// GetAll returns all stored vectors.
func (vs *PersistentVectorStore) GetAll() []*StoredVector {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	result := make([]*StoredVector, 0, len(vs.vectors))
	for _, v := range vs.vectors {
		result = append(result, v)
	}
	return result
}

// Export exports vectors to a JSON file.
func (vs *PersistentVectorStore) Export(path string) error {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	vectors := make([]*StoredVector, 0, len(vs.vectors))
	for _, v := range vs.vectors {
		vectors = append(vectors, v)
	}

	data, err := json.MarshalIndent(vectors, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}

// Import imports vectors from a JSON file.
func (vs *PersistentVectorStore) Import(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	var vectors []*StoredVector
	if err := json.Unmarshal(data, &vectors); err != nil {
		return err
	}

	vs.mu.Lock()
	defer vs.mu.Unlock()

	for _, v := range vectors {
		vs.vectors[v.ID] = v
	}
	vs.dirty = true
	return vs.save()
}

// Helper functions

func normalize(v []float64) []float64 {
	var sum float64
	for _, val := range v {
		sum += val * val
	}
	norm := math.Sqrt(sum)
	if norm == 0 {
		return v
	}
	result := make([]float64, len(v))
	for i, val := range v {
		result[i] = val / norm
	}
	return result
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot float64
	for i := range a {
		dot += a[i] * b[i]
	}
	return dot
}

// VectorStoreManager manages multiple vector stores by namespace.
type VectorStoreManager struct {
	storage *Storage
	stores  map[string]*PersistentVectorStore
	mu      sync.RWMutex
}

// NewVectorStoreManager creates a new vector store manager.
func NewVectorStoreManager(storage *Storage) *VectorStoreManager {
	return &VectorStoreManager{
		storage: storage,
		stores:  make(map[string]*PersistentVectorStore),
	}
}

// GetStore gets or creates a vector store for a namespace.
func (m *VectorStoreManager) GetStore(namespace string) (*PersistentVectorStore, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if store, ok := m.stores[namespace]; ok {
		return store, nil
	}

	store, err := NewPersistentVectorStore(m.storage, namespace)
	if err != nil {
		return nil, err
	}

	m.stores[namespace] = store
	return store, nil
}

// FlushAll saves all dirty stores.
func (m *VectorStoreManager) FlushAll() error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	for _, store := range m.stores {
		if err := store.Flush(); err != nil {
			return err
		}
	}
	return nil
}

// GetStats returns stats for all stores.
func (m *VectorStoreManager) GetStats() map[string]int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	stats := make(map[string]int)
	for name, store := range m.stores {
		stats[name] = store.Size()
	}
	return stats
}
