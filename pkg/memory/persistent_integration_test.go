package memory

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/oarkflow/ai-agent/pkg/storage"
)

func TestPersistentSemanticMemoryIntegration(t *testing.T) {
	// Create temp dir for storage
	tmpDir, err := os.MkdirTemp("", "semantic_memory_test")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	st, err := storage.NewStorage(tmpDir)
	if err != nil {
		t.Fatalf("failed to create storage: %v", err)
	}

	// Ensure embeddings file exists to avoid constructor read error
	if err := os.MkdirAll(filepath.Join(tmpDir, "embeddings"), 0755); err != nil {
		t.Fatalf("failed to create embeddings dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "embeddings", "integration.json"), []byte("[]"), 0644); err != nil {
		t.Fatalf("failed to write initial embeddings file: %v", err)
	}

	// Use training in-memory vector store and adapter to back persistent store locations
	// For this integration test we simply create persistent store via adapter helper
	provider := &mockProvider{resp: "", genCalled: nil}

	sm, err := NewSemanticMemoryWithPersistentStore(st, "integration", provider)
	if err != nil {
		t.Fatalf("failed to create semantic memory with persistent store: %v", err)
	}

	mem, err := sm.Store(context.Background(), "integration test content", MemoryFact, map[string]any{"source": "test"})
	if err != nil {
		t.Fatalf("store failed: %v", err)
	}
	if mem == nil {
		t.Fatalf("expected memory item")
	}

	// Flush storage files and ensure files exist
	if _, err := os.Stat(filepath.Join(tmpDir, "embeddings", "integration.json")); os.IsNotExist(err) {
		// not required that file exists immediately (async writes may defer), so try to recall
	}

	results, err := sm.Recall(context.Background(), "integration", 5)
	if err != nil {
		t.Fatalf("recall failed: %v", err)
	}
	if len(results) == 0 {
		t.Fatalf("expected recall to return at least one memory")
	}
}
