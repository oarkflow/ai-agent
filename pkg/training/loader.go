package training

import (
	"fmt"

	"github.com/sujit/ai-agent/pkg/config"
	"github.com/sujit/ai-agent/pkg/llm"
)

// DomainLoader loads domain configurations.
type DomainLoader struct {
	config *config.Config
}

// NewDomainLoader creates a new domain loader.
func NewDomainLoader(cfg *config.Config) *DomainLoader {
	return &DomainLoader{config: cfg}
}

// LoadDomains loads all domains from configuration into the trainer.
func (dl *DomainLoader) LoadDomains(trainer *DomainTrainer) error {
	if dl.config.Domains == nil {
		return nil
	}

	for _, domainCfg := range dl.config.Domains.Domains {
		if !domainCfg.Enabled {
			continue
		}

		// Create the domain
		domain, err := trainer.CreateDomainWithID(domainCfg.ID, domainCfg.Name, domainCfg.Description)
		if err != nil {
			return fmt.Errorf("failed to create domain %s: %w", domainCfg.ID, err)
		}
		domain.SystemPrompt = domainCfg.SystemPrompt

		// Add terminology
		for term, definition := range domainCfg.Terminology {
			if err := trainer.AddTerminology(domainCfg.ID, term, definition); err != nil {
				// Log warning but continue
				continue
			}
		}

		// Add guidelines
		for _, guideline := range domainCfg.Guidelines {
			if err := trainer.AddGuideline(domainCfg.ID, guideline); err != nil {
				// Log warning but continue
				continue
			}
		}
	}

	return nil
}

// LoadDomain loads a single domain by ID.
func (dl *DomainLoader) LoadDomain(trainer *DomainTrainer, domainID string) error {
	domainCfg, ok := dl.config.GetDomain(domainID)
	if !ok {
		return fmt.Errorf("domain not found in configuration: %s", domainID)
	}

	// Create the domain
	domain, err := trainer.CreateDomainWithID(domainCfg.ID, domainCfg.Name, domainCfg.Description)
	if err != nil {
		return fmt.Errorf("failed to create domain %s: %w", domainCfg.ID, err)
	}
	domain.SystemPrompt = domainCfg.SystemPrompt

	// Add terminology
	for term, definition := range domainCfg.Terminology {
		_ = trainer.AddTerminology(domainCfg.ID, term, definition)
	}

	// Add guidelines
	for _, guideline := range domainCfg.Guidelines {
		_ = trainer.AddGuideline(domainCfg.ID, guideline)
	}

	return nil
}

// GetDomainSettings returns the domain settings from configuration.
func (dl *DomainLoader) GetDomainSettings() *DomainSettings {
	if dl.config.Domains == nil {
		return defaultDomainSettings()
	}

	settings := dl.config.Domains.DomainSettings

	return &DomainSettings{
		StoragePath:         settings.StoragePath,
		ChunkSize:           settings.ChunkSize,
		ChunkOverlap:        settings.ChunkOverlap,
		EmbeddingModel:      settings.EmbeddingModel,
		SimilarityThreshold: settings.SimilarityThreshold,
		MaxContextDocuments: settings.MaxContextDocuments,
		AutoDetectDomain:    settings.AutoDetectDomain,
	}
}

// DomainSettings contains domain configuration settings.
type DomainSettings struct {
	StoragePath         string
	ChunkSize           int
	ChunkOverlap        int
	EmbeddingModel      string
	SimilarityThreshold float64
	MaxContextDocuments int
	AutoDetectDomain    bool
}

func defaultDomainSettings() *DomainSettings {
	return &DomainSettings{
		StoragePath:         "./data/domains",
		ChunkSize:           1000,
		ChunkOverlap:        200,
		EmbeddingModel:      "text-embedding-3-small",
		SimilarityThreshold: 0.7,
		MaxContextDocuments: 5,
		AutoDetectDomain:    true,
	}
}

// NewDomainTrainerFromConfig creates a domain trainer from configuration.
func NewDomainTrainerFromConfig(cfg *config.Config, provider llm.MultimodalProvider, vectorStore VectorStore) (*DomainTrainer, error) {
	loader := NewDomainLoader(cfg)
	settings := loader.GetDomainSettings()

	trainer := NewDomainTrainer(
		provider,
		vectorStore,
		WithStoragePath(settings.StoragePath),
		WithChunkSize(settings.ChunkSize),
		WithChunkOverlap(settings.ChunkOverlap),
	)

	// Load domains from config
	if err := loader.LoadDomains(trainer); err != nil {
		return nil, err
	}

	return trainer, nil
}
