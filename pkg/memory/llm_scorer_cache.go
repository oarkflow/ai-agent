package memory

import (
	"context"
	"strings"
	"sync"

	"github.com/oarkflow/ai-agent/pkg/content"
)

// LLMScorerCache caches results from an underlying MessageScorer to avoid repeated LLM calls.
// It's safe for concurrent use.
type LLMScorerCache struct {
	scorer MessageScorer
	mu     sync.RWMutex
	cache  map[string]float64
}

func NewLLMScorerCache(scorer MessageScorer) *LLMScorerCache {
	return &LLMScorerCache{scorer: scorer, cache: make(map[string]float64)}
}

func (c *LLMScorerCache) Score(ctx context.Context, msg *content.Message) float64 {
	if msg == nil {
		return 0.0
	}
	key := strings.TrimSpace(msg.GetText())
	if key == "" {
		return 0.0
	}

	c.mu.RLock()
	v, ok := c.cache[key]
	c.mu.RUnlock()
	if ok {
		return v
	}

	// Compute and store
	v = c.scorer.Score(ctx, msg)
	c.mu.Lock()
	c.cache[key] = v
	c.mu.Unlock()
	return v
}
