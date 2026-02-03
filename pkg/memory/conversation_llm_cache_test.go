package memory

import (
	"testing"

	"github.com/sujit/ai-agent/pkg/content"
)

func TestConversationMemory_DefaultsToCachedLLMScorer(t *testing.T) {
	cfg := &MemoryConfig{MaxMessages: 3, Strategy: StrategyImportance, UseScorerCache: true}
	cm := NewConversationMemory(cfg)
	p := &countingProvider{}
	// Setting summary provider should auto-wire cached LLM scorer
	cm.SetSummaryProvider(p)

	cm.Add(content.NewSystemMessage("system"))
	// Add many messages, some duplicates to exercise cache
	cm.Add(content.NewUserMessage("dup"))
	cm.Add(content.NewUserMessage("dup"))
	cm.Add(content.NewUserMessage("unique1"))
	cm.Add(content.NewUserMessage("dup"))
	cm.Add(content.NewUserMessage("unique2"))

	// After trimming, scorer should have been called at most number of unique texts
	if p.calls > 3 {
		t.Fatalf("expected provider to be called at most 3 times, called %d", p.calls)
	}
}
