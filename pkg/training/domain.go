package training

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/sujit/ai-agent/pkg/content"
	"github.com/sujit/ai-agent/pkg/llm"
)

// RAGConfig defines parameters for RAG optimization experiments.
type RAGConfig struct {
	ChunkSize    int         `json:"chunk_size"`
	ChunkOverlap int         `json:"chunk_overlap"`
	TopK         int         `json:"top_k"`
	VectorStore  VectorStore `json:"-"`
}

//
// ==============================
// Core Domain Models
// ==============================
//

type DomainKnowledge struct {
	ID           string            `json:"id"`
	Name         string            `json:"name"`
	Description  string            `json:"description"`
	SystemPrompt string            `json:"system_prompt"`
	Documents    []Document        `json:"documents"`
	Examples     []TrainingExample `json:"examples"`
	Guidelines   []string          `json:"guidelines"`
	Terminology  map[string]string `json:"terminology"`
	Metadata     map[string]any    `json:"metadata"`
	CreatedAt    time.Time         `json:"created_at"`
	UpdatedAt    time.Time         `json:"updated_at"`
}

type Document struct {
	ID        string              `json:"id"`
	Title     string              `json:"title"`
	Content   string              `json:"content"`
	Source    string              `json:"source"`
	Type      content.ContentType `json:"type"`
	Chunks    []DocumentChunk     `json:"chunks,omitempty"`
	Metadata  map[string]any      `json:"metadata,omitempty"`
	CreatedAt time.Time           `json:"created_at"`
}

type DocumentChunk struct {
	ID        string    `json:"id"`
	DocID     string    `json:"doc_id"`
	Content   string    `json:"content"`
	StartIdx  int       `json:"start_idx"`
	EndIdx    int       `json:"end_idx"`
	Embedding []float64 `json:"embedding,omitempty"`
}

type TrainingExample struct {
	ID        string             `json:"id"`
	Messages  []*content.Message `json:"messages"`
	Validated bool               `json:"validated"`
	Quality   float64            `json:"quality,omitempty"`
	Category  string             `json:"category,omitempty"`
	CreatedAt time.Time          `json:"created_at"`
}

//
// ==============================
// Vector Store Abstraction
// ==============================
//

type VectorResult struct {
	ID       string         `json:"id"`
	Score    float64        `json:"score"`
	Content  string         `json:"content,omitempty"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

type VectorStore interface {
	Store(ctx context.Context, id string, embedding []float64, metadata map[string]any) error
	Search(ctx context.Context, query []float64, topK int, filter map[string]any) ([]VectorResult, error)
	Delete(ctx context.Context, id string) error
	Clear(ctx context.Context) error
}

//
// ==============================
// Trainer
// ==============================
//

type DomainTrainer struct {
	provider     llm.MultimodalProvider
	vectorStore  VectorStore
	domains      map[string]*DomainKnowledge
	storagePath  string
	chunkSize    int
	chunkOverlap int
	mu           sync.RWMutex
}

type TrainerOption func(*DomainTrainer)

func NewDomainTrainer(
	provider llm.MultimodalProvider,
	vectorStore VectorStore,
	opts ...TrainerOption,
) *DomainTrainer {
	t := &DomainTrainer{
		provider:     provider,
		vectorStore:  vectorStore,
		domains:      make(map[string]*DomainKnowledge),
		chunkSize:    1000,
		chunkOverlap: 200,
	}

	for _, opt := range opts {
		opt(t)
	}

	return t
}

func WithStoragePath(path string) TrainerOption {
	return func(t *DomainTrainer) {
		t.storagePath = path
	}
}

func WithChunkSize(size int) TrainerOption {
	return func(t *DomainTrainer) {
		t.chunkSize = size
	}
}

func WithChunkOverlap(overlap int) TrainerOption {
	return func(t *DomainTrainer) {
		t.chunkOverlap = overlap
	}
}

//
// ==============================
// Domain Lifecycle
// ==============================
//

func (t *DomainTrainer) CreateDomain(name, description string) (*DomainKnowledge, error) {
	return t.CreateDomainWithID(generateID(name), name, description)
}

func (t *DomainTrainer) CreateDomainWithID(id, name, description string) (*DomainKnowledge, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if _, exists := t.domains[id]; exists {
		return nil, fmt.Errorf("domain already exists")
	}

	domain := &DomainKnowledge{
		ID:          id,
		Name:        name,
		Description: description,
		Documents:   []Document{},
		Examples:    []TrainingExample{},
		Guidelines:  []string{},
		Terminology: map[string]string{},
		Metadata:    map[string]any{},
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}

	t.domains[id] = domain
	return domain, nil
}

func (t *DomainTrainer) GetDomain(id string) (*DomainKnowledge, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	d, ok := t.domains[id]
	if !ok {
		return nil, fmt.Errorf("domain not found")
	}
	return d, nil
}

// AddTerminology adds a term and its definition to a domain.
func (t *DomainTrainer) AddTerminology(domainID, term, definition string) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	domain, ok := t.domains[domainID]
	if !ok {
		return fmt.Errorf("domain not found")
	}

	if domain.Terminology == nil {
		domain.Terminology = make(map[string]string)
	}
	domain.Terminology[term] = definition
	domain.UpdatedAt = time.Now()
	return nil
}

// AddGuideline adds a guideline to a domain.
func (t *DomainTrainer) AddGuideline(domainID, guideline string) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	domain, ok := t.domains[domainID]
	if !ok {
		return fmt.Errorf("domain not found")
	}

	domain.Guidelines = append(domain.Guidelines, guideline)
	domain.UpdatedAt = time.Now()
	return nil
}

// SetSystemPrompt sets the system prompt for a domain.
func (t *DomainTrainer) SetSystemPrompt(domainID, prompt string) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	domain, ok := t.domains[domainID]
	if !ok {
		return fmt.Errorf("domain not found")
	}

	domain.SystemPrompt = prompt
	domain.UpdatedAt = time.Now()
	return nil
}

// AddTrainingExample adds a training example to a domain.
func (t *DomainTrainer) AddTrainingExample(domainID string, example *TrainingExample) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	domain, ok := t.domains[domainID]
	if !ok {
		return fmt.Errorf("domain not found")
	}

	if example.ID == "" {
		example.ID = generateID(fmt.Sprintf("%d", len(domain.Examples)))
	}
	example.CreatedAt = time.Now()
	domain.Examples = append(domain.Examples, *example)
	domain.UpdatedAt = time.Now()
	return nil
}

//
// ==============================
// Documents & RAG
// ==============================
//

func (t *DomainTrainer) AddDocument(ctx context.Context, domainID string, doc *Document) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	domain, ok := t.domains[domainID]
	if !ok {
		return fmt.Errorf("domain not found")
	}

	if doc.ID == "" {
		doc.ID = generateID(doc.Title + doc.Content)
	}
	doc.CreatedAt = time.Now()

	doc.Chunks = t.chunkDocument(doc)

	if err := t.generateChunkEmbeddings(ctx, domainID, doc.Chunks); err != nil {
		return err
	}

	domain.Documents = append(domain.Documents, *doc)
	domain.UpdatedAt = time.Now()
	return nil
}

// Reindex re-processes all documents in a domain with the current trainer settings.
// This is useful for testing different chunking strategies.
func (t *DomainTrainer) Reindex(ctx context.Context, domainID string) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	domain, ok := t.domains[domainID]
	if !ok {
		return fmt.Errorf("domain not found")
	}

	// 1. Clear existing vectors for this domain from the store
	// Note: Simple InMemoryVectorStore doesn't support selective clearing easily without a scan,
	// but we can just clear and re-store everything for now.
	// In a real DB we'd use a filter.
	if t.vectorStore != nil {
		// For simplicity in this local implementation, we'll assume Search can be used to find IDs
		// but VectorStore interface doesn't have ListIDsByFilter.
		// Let's just Clear the whole store if it's an experiment store.
		// For now, we'll assume the user provides a fresh VectorStore for experiments.
	}

	// 2. Re-chunk and re-embed all documents
	for i := range domain.Documents {
		doc := &domain.Documents[i]
		doc.Chunks = t.chunkDocument(doc)
		if err := t.generateChunkEmbeddings(ctx, domainID, doc.Chunks); err != nil {
			return err
		}
	}

	domain.UpdatedAt = time.Now()
	return nil
}

// UseConfig temporarily applies a RAGConfig for the duration of the provided function.
func (t *DomainTrainer) UseConfig(cfg *RAGConfig, fn func() error) error {
	t.mu.Lock()

	// Backup original settings
	origSize := t.chunkSize
	origOverlap := t.chunkOverlap
	origStore := t.vectorStore

	// Apply new settings
	if cfg.ChunkSize > 0 {
		t.chunkSize = cfg.ChunkSize
	}
	if cfg.ChunkOverlap >= 0 {
		t.chunkOverlap = cfg.ChunkOverlap
	}
	if cfg.VectorStore != nil {
		t.vectorStore = cfg.VectorStore
	}

	t.mu.Unlock()

	// Execute the test function
	err := fn()

	// Restore original settings
	t.mu.Lock()
	t.chunkSize = origSize
	t.chunkOverlap = origOverlap
	t.vectorStore = origStore
	t.mu.Unlock()

	return err
}

func (t *DomainTrainer) chunkDocument(doc *Document) []DocumentChunk {
	var chunks []DocumentChunk
	text := doc.Content

	for i := 0; i < len(text); i += t.chunkSize - t.chunkOverlap {
		end := min(i+t.chunkSize, len(text))
		chunks = append(chunks, DocumentChunk{
			ID:       generateID(fmt.Sprintf("%s-%d", doc.ID, i)),
			DocID:    doc.ID,
			Content:  text[i:end],
			StartIdx: i,
			EndIdx:   end,
		})
		if end >= len(text) {
			break
		}
	}
	return chunks
}

func (t *DomainTrainer) generateChunkEmbeddings(
	ctx context.Context,
	domainID string,
	chunks []DocumentChunk,
) error {
	for i := range chunks {
		resp, err := t.provider.Embed(ctx, &llm.EmbeddingRequest{
			Input: []string{chunks[i].Content},
		})
		if err != nil {
			return err
		}
		if len(resp.Embeddings) == 0 {
			continue
		}

		chunks[i].Embedding = resp.Embeddings[0]

		if t.vectorStore != nil {
			err = t.vectorStore.Store(
				ctx,
				chunks[i].ID,
				chunks[i].Embedding,
				map[string]any{
					"domain_id": domainID,
					"doc_id":    chunks[i].DocID,
					"content":   chunks[i].Content,
				},
			)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func (t *DomainTrainer) Search(
	ctx context.Context,
	domainID string,
	query string,
	topK int,
) ([]VectorResult, error) {
	resp, err := t.provider.Embed(ctx, &llm.EmbeddingRequest{
		Input: []string{query},
	})
	if err != nil || len(resp.Embeddings) == 0 {
		return nil, fmt.Errorf("embedding failed")
	}

	return t.vectorStore.Search(ctx, resp.Embeddings[0], topK, map[string]any{
		"domain_id": domainID,
	})
}

//
// ==============================
// System Prompt (RAG)
// ==============================
//

func (t *DomainTrainer) BuildSystemPrompt(
	ctx context.Context,
	domainID string,
	userQuery string,
) (string, error) {
	domain, err := t.GetDomain(domainID)
	if err != nil {
		return "", err
	}

	// If explicit system prompt is set, use it.
	// We can still append RAG context if userQuery is provided.
	var sb strings.Builder
	if domain.SystemPrompt != "" {
		sb.WriteString(domain.SystemPrompt)
		sb.WriteString("\n\n")
	} else {
		sb.WriteString(fmt.Sprintf("You are an AI assistant specialized in %s.\n\n", domain.Name))
		sb.WriteString(fmt.Sprintf("Domain Description: %s\n\n", domain.Description))

		if len(domain.Guidelines) > 0 {
			sb.WriteString("Guidelines:\n")
			for _, g := range domain.Guidelines {
				sb.WriteString("- " + g + "\n")
			}
			sb.WriteString("\n")
		}
	}

	if userQuery != "" && t.vectorStore != nil {
		results, err := t.Search(ctx, domainID, userQuery, 5)
		if err == nil && len(results) > 0 {
			sb.WriteString("Relevant Context:\n```\n")
			for _, r := range results {
				sb.WriteString(r.Content + "\n---\n")
			}
			sb.WriteString("```\n\n")
		}
	}

	if domain.SystemPrompt == "" {
		sb.WriteString("Provide accurate, domain-specific responses.")
	}
	return sb.String(), nil
}

//
// ==============================
// Helpers
// ==============================
//

func generateID(input string) string {
	hash := sha256.Sum256([]byte(input + time.Now().String()))
	return hex.EncodeToString(hash[:])[:16]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
