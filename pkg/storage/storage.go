package storage

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// Storage provides persistent file-based storage for training data.
type Storage struct {
	basePath string
	mu       sync.RWMutex
}

// StorageConfig holds storage configuration.
type StorageConfig struct {
	BasePath string `json:"base_path"`
}

// NewStorage creates a new storage instance.
func NewStorage(basePath string) (*Storage, error) {
	s := &Storage{basePath: basePath}

	// Create required directories
	dirs := []string{
		"domains",
		"examples",
		"memory",
		"embeddings",
		"documents",
		"prompts",
		"cache",
	}

	for _, dir := range dirs {
		path := filepath.Join(basePath, dir)
		if err := os.MkdirAll(path, 0755); err != nil {
			return nil, fmt.Errorf("failed to create directory %s: %w", path, err)
		}
	}

	return s, nil
}

// GetPath returns the full path for a storage location.
func (s *Storage) GetPath(parts ...string) string {
	return filepath.Join(append([]string{s.basePath}, parts...)...)
}

// SaveJSON saves data as JSON to a file.
func (s *Storage) SaveJSON(path string, data any) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	fullPath := filepath.Join(s.basePath, path)

	// Ensure directory exists
	if err := os.MkdirAll(filepath.Dir(fullPath), 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	content, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal data: %w", err)
	}

	return os.WriteFile(fullPath, content, 0644)
}

// LoadJSON loads data from a JSON file.
func (s *Storage) LoadJSON(path string, data any) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	fullPath := filepath.Join(s.basePath, path)

	content, err := os.ReadFile(fullPath)
	if err != nil {
		return fmt.Errorf("failed to read file: %w", err)
	}

	return json.Unmarshal(content, data)
}

// Exists checks if a file exists.
func (s *Storage) Exists(path string) bool {
	fullPath := filepath.Join(s.basePath, path)
	_, err := os.Stat(fullPath)
	return err == nil
}

// Delete removes a file.
func (s *Storage) Delete(path string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	fullPath := filepath.Join(s.basePath, path)
	return os.Remove(fullPath)
}

// List returns all files in a directory.
func (s *Storage) List(dir string) ([]string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	fullPath := filepath.Join(s.basePath, dir)
	entries, err := os.ReadDir(fullPath)
	if err != nil {
		if os.IsNotExist(err) {
			return []string{}, nil
		}
		return nil, err
	}

	var files []string
	for _, entry := range entries {
		if !entry.IsDir() {
			files = append(files, entry.Name())
		}
	}
	return files, nil
}

// AppendJSONL appends a JSON line to a JSONL file (for logs, examples).
func (s *Storage) AppendJSONL(path string, data any) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	fullPath := filepath.Join(s.basePath, path)

	// Ensure directory exists
	if err := os.MkdirAll(filepath.Dir(fullPath), 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	f, err := os.OpenFile(fullPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	content, err := json.Marshal(data)
	if err != nil {
		return err
	}

	_, err = f.WriteString(string(content) + "\n")
	return err
}

// ReadJSONL reads all lines from a JSONL file.
func (s *Storage) ReadJSONL(path string, factory func() any) ([]any, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	fullPath := filepath.Join(s.basePath, path)

	content, err := os.ReadFile(fullPath)
	if err != nil {
		if os.IsNotExist(err) {
			return []any{}, nil
		}
		return nil, err
	}

	var results []any
	lines := splitLines(string(content))

	for _, line := range lines {
		if line == "" {
			continue
		}
		item := factory()
		if err := json.Unmarshal([]byte(line), item); err != nil {
			continue // Skip malformed lines
		}
		results = append(results, item)
	}

	return results, nil
}

func splitLines(s string) []string {
	var lines []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			lines = append(lines, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		lines = append(lines, s[start:])
	}
	return lines
}

// StorageMetadata holds metadata about stored items.
type StorageMetadata struct {
	ID        string         `json:"id"`
	Type      string         `json:"type"`
	Domain    string         `json:"domain,omitempty"`
	CreatedAt time.Time      `json:"created_at"`
	UpdatedAt time.Time      `json:"updated_at"`
	Tags      []string       `json:"tags,omitempty"`
	Extra     map[string]any `json:"extra,omitempty"`
}

// Examples returns a TrainingExampleStore for a domain.
func (s *Storage) Examples(domain string) *TrainingExampleStore {
	return &TrainingExampleStore{
		storage: s,
		domain:  domain,
	}
}
