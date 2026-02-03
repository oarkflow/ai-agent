package training

import (
	"context"
	"math"
	"sort"
	"sync"
)

// ------------------------------
// Vector Store
// ------------------------------

// InMemoryVectorStore is a simple in-memory vector store for development.
type InMemoryVectorStore struct {
	mu      sync.RWMutex
	vectors map[string]*StoredVector
}

// StoredVector represents a stored vector with metadata.
type StoredVector struct {
	ID        string
	Embedding []float64
	Metadata  map[string]any
}

// ------------------------------
// Constructor
// ------------------------------

func NewInMemoryVectorStore() *InMemoryVectorStore {
	return &InMemoryVectorStore{
		vectors: make(map[string]*StoredVector),
	}
}

// ------------------------------
// CRUD Operations
// ------------------------------

// Store stores an embedding with metadata.
func (s *InMemoryVectorStore) Store(
	ctx context.Context,
	id string,
	embedding []float64,
	metadata map[string]any,
) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.vectors[id] = &StoredVector{
		ID:        id,
		Embedding: normalize(embedding),
		Metadata:  metadata,
	}
	return nil
}

// Delete removes an embedding by ID.
func (s *InMemoryVectorStore) Delete(ctx context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	delete(s.vectors, id)
	return nil
}

// Clear removes all embeddings.
func (s *InMemoryVectorStore) Clear(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.vectors = make(map[string]*StoredVector)
	return nil
}

// Size returns the number of stored vectors.
func (s *InMemoryVectorStore) Size() int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return len(s.vectors)
}

// ------------------------------
// Search
// ------------------------------

func (s *InMemoryVectorStore) Search(
	ctx context.Context,
	query []float64,
	topK int,
	filter map[string]any,
) ([]VectorResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	query = normalize(query)

	type scored struct {
		id       string
		score    float64
		metadata map[string]any
	}

	var results []scored

	for _, v := range s.vectors {
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

	// Convert to VectorResult
	vectorResults := make([]VectorResult, len(results))
	for i, r := range results {
		vectorResults[i] = VectorResult{
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

// ------------------------------
// Math Helpers
// ------------------------------

// normalize normalizes a vector to unit length.
func normalize(v []float64) []float64 {
	var norm float64
	for _, val := range v {
		norm += val * val
	}

	norm = math.Sqrt(norm)
	if norm == 0 {
		return v
	}

	result := make([]float64, len(v))
	for i, val := range v {
		result[i] = val / norm
	}
	return result
}

// dotProduct calculates the dot product of two vectors.
func dotProduct(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// euclideanDistance calculates the Euclidean distance between two vectors.
func euclideanDistance(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.MaxFloat64
	}

	var sum float64
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// cosineSimilarity calculates the cosine similarity between two vectors.
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
