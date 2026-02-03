package domains

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/sujit/ai-agent/pkg/llm"
	"github.com/sujit/ai-agent/pkg/storage"
)

// DomainRegistry manages all domain trainers.
type DomainRegistry struct {
	domains  map[string]Domain
	storage  *storage.Storage
	provider llm.MultimodalProvider
	mu       sync.RWMutex
}

// Domain is the interface all domain trainers must implement.
type Domain interface {
	GetID() string
	GetName() string
	GetSystemPrompt() string
	GetFewShotExamples(category string, count int) ([]*FewShotExample, error)
	AddTrainingExample(input, output, category string, quality float64) error
	Generate(ctx context.Context, prompt string, category string) (string, error)
}

// DomainInfo provides metadata about a registered domain.
type DomainInfo struct {
	ID          string   `json:"id"`
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Categories  []string `json:"categories"`
	ExampleCount int     `json:"example_count"`
}

// NewDomainRegistry creates a new domain registry.
func NewDomainRegistry(provider llm.MultimodalProvider, storage *storage.Storage) *DomainRegistry {
	return &DomainRegistry{
		domains:  make(map[string]Domain),
		storage:  storage,
		provider: provider,
	}
}

// RegisterDomain registers a domain trainer.
func (r *DomainRegistry) RegisterDomain(domain Domain) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.domains[domain.GetID()] = domain
}

// GetDomain retrieves a domain by ID.
func (r *DomainRegistry) GetDomain(id string) (Domain, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	domain, ok := r.domains[id]
	return domain, ok
}

// ListDomains returns info about all registered domains.
func (r *DomainRegistry) ListDomains() []*DomainInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var infos []*DomainInfo
	for _, domain := range r.domains {
		info := &DomainInfo{
			ID:   domain.GetID(),
			Name: domain.GetName(),
		}

		infos = append(infos, info)
	}
	return infos
}

// RegisterBuiltinDomains registers all built-in domain trainers.
func (r *DomainRegistry) RegisterBuiltinDomains() {
	// Register workflow domain
	workflow := NewWorkflowDomain(r.provider, r.storage)
	r.RegisterDomain(workflow)

	// Register ReactFlow domain
	reactflow := NewReactFlowDomain(r.provider, r.storage)
	r.RegisterDomain(reactflow)

	// Register Healthcare domain
	healthcare := NewHealthcareDomain(r.provider, r.storage)
	r.RegisterDomain(healthcare)

	// Register Multimodal domain
	multimodal := NewMultimodalDomain(r.provider, r.storage)
	r.RegisterDomain(multimodal)
}

// LoadExamplesFromFiles loads training examples from JSON files.
func (r *DomainRegistry) LoadExamplesFromFiles(dataDir string) error {
	examplesDir := filepath.Join(dataDir, "examples")

	entries, err := os.ReadDir(examplesDir)
	if err != nil {
		return fmt.Errorf("failed to read examples directory: %w", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		domainID := entry.Name()
		domain, ok := r.GetDomain(domainID)
		if !ok {
			continue
		}

		examplesFile := filepath.Join(examplesDir, domainID, "examples.json")
		data, err := os.ReadFile(examplesFile)
		if err != nil {
			continue
		}

		var examples []*storage.TrainingExample
		if err := json.Unmarshal(data, &examples); err != nil {
			continue
		}

		for _, ex := range examples {
			domain.AddTrainingExample(ex.Input, ex.Output, ex.Category, ex.Quality)
		}
	}

	return nil
}

// Generate generates output using the specified domain.
func (r *DomainRegistry) Generate(ctx context.Context, domainID, prompt, category string) (string, error) {
	domain, ok := r.GetDomain(domainID)
	if !ok {
		return "", fmt.Errorf("domain not found: %s", domainID)
	}
	return domain.Generate(ctx, prompt, category)
}

// TrainFromFeedback adds a training example from user feedback.
func (r *DomainRegistry) TrainFromFeedback(domainID, input, output, category string, rating float64) error {
	domain, ok := r.GetDomain(domainID)
	if !ok {
		return fmt.Errorf("domain not found: %s", domainID)
	}
	return domain.AddTrainingExample(input, output, category, rating/5.0)
}

// ExportAllExamples exports all training examples to a directory.
func (r *DomainRegistry) ExportAllExamples(outputDir string) error {
	r.mu.RLock()
	defer r.mu.RUnlock()

	for id, domain := range r.domains {
		// Try to get examples from domain if it has ExportExamples method
		if exporter, ok := domain.(interface{ ExportExamples(string) error }); ok {
			outputPath := filepath.Join(outputDir, id, "examples.json")
			if err := exporter.ExportExamples(outputPath); err != nil {
				return fmt.Errorf("failed to export %s: %w", id, err)
			}
		}
	}
	return nil
}

// GetStats returns statistics about all domains.
func (r *DomainRegistry) GetStats() map[string]map[string]int {
	r.mu.RLock()
	defer r.mu.RUnlock()

	stats := make(map[string]map[string]int)
	for id := range r.domains {
		stats[id] = map[string]int{
			"examples": 0,
		}
	}
	return stats
}
