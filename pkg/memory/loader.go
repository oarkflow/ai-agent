package memory

import (
	"github.com/sujit/ai-agent/pkg/config"
)

// MemoryLoader loads memory configuration.
type MemoryLoader struct {
	config *config.Config
}

// NewMemoryLoader creates a new memory loader.
func NewMemoryLoader(cfg *config.Config) *MemoryLoader {
	return &MemoryLoader{config: cfg}
}

// LoadMemoryConfig creates a MemoryConfig from configuration.
func (ml *MemoryLoader) LoadMemoryConfig() *MemoryConfig {
	memoryCfg := ml.config.Memory

	strategy := StrategyWindow
	switch memoryCfg.Strategy {
	case "window":
		strategy = StrategyWindow
	case "summary":
		strategy = StrategySummary
	case "importance":
		strategy = StrategyImportance
	case "hybrid":
		strategy = StrategyHybrid
	}

	return &MemoryConfig{
		MaxMessages:      memoryCfg.MaxMessages,
		MaxTokens:        memoryCfg.MaxTokens,
		Strategy:         strategy,
		SummaryInterval:  memoryCfg.SummaryInterval,
		KeepSystemPrompt: memoryCfg.KeepSystemPrompt,
	}
}

// GetSemanticMemoryConfig returns semantic memory settings.
func (ml *MemoryLoader) GetSemanticMemoryConfig() *SemanticMemoryConfig {
	semCfg := ml.config.Memory.SemanticMemory

	return &SemanticMemoryConfig{
		Enabled:             semCfg.Enabled,
		VectorStoreType:     semCfg.VectorStoreType,
		SimilarityThreshold: semCfg.SimilarityThreshold,
		MaxResults:          semCfg.MaxResults,
	}
}

// IsMemoryEnabled returns whether memory is enabled.
func (ml *MemoryLoader) IsMemoryEnabled() bool {
	return ml.config.Memory.Enabled
}

// GetPersistPath returns the memory persistence path.
func (ml *MemoryLoader) GetPersistPath() string {
	return ml.config.Memory.PersistPath
}

// SemanticMemoryConfig holds semantic memory configuration.
type SemanticMemoryConfig struct {
	Enabled             bool
	VectorStoreType     string
	SimilarityThreshold float64
	MaxResults          int
}

// NewConversationMemoryFromConfig creates a conversation memory from configuration.
func NewConversationMemoryFromConfig(cfg *config.Config) *ConversationMemory {
	loader := NewMemoryLoader(cfg)

	if !loader.IsMemoryEnabled() {
		// Return a minimal memory instance
		return NewConversationMemory(&MemoryConfig{
			MaxMessages: 10,
			Strategy:    StrategyWindow,
		})
	}

	memoryConfig := loader.LoadMemoryConfig()
	return NewConversationMemory(memoryConfig)
}
