package storage

import (
	"fmt"
	"time"
)

// TrainingExample represents a training example for domain-specific learning.
type TrainingExample struct {
	ID          string         `json:"id"`
	Domain      string         `json:"domain"`
	Category    string         `json:"category"`
	Input       string         `json:"input"`
	Output      string         `json:"output"`
	InputType   string         `json:"input_type,omitempty"`  // text, image, document, multimodal
	OutputType  string         `json:"output_type,omitempty"` // json, text, code, markdown
	Quality     float64        `json:"quality,omitempty"`     // 0.0 to 1.0
	Validated   bool           `json:"validated"`
	Tags        []string       `json:"tags,omitempty"`
	Metadata    map[string]any `json:"metadata,omitempty"`
	CreatedAt   time.Time      `json:"created_at"`
	UpdatedAt   time.Time      `json:"updated_at"`
}

// TrainingExampleStore manages training examples.
type TrainingExampleStore struct {
	storage *Storage
	domain  string
}

// NewTrainingExampleStore creates a new training example store.
func NewTrainingExampleStore(storage *Storage, domain string) *TrainingExampleStore {
	return &TrainingExampleStore{storage: storage, domain: domain}
}

// Add adds a training example.
func (s *TrainingExampleStore) Add(example *TrainingExample) error {
	if example.ID == "" {
		example.ID = generateID()
	}
	if example.Domain == "" {
		example.Domain = s.domain
	}
	if example.CreatedAt.IsZero() {
		example.CreatedAt = time.Now()
	}
	example.UpdatedAt = time.Now()

	path := fmt.Sprintf("examples/%s/%s.json", example.Domain, example.ID)
	return s.storage.SaveJSON(path, example)
}

// Get retrieves a training example by ID.
func (s *TrainingExampleStore) Get(id string) (*TrainingExample, error) {
	path := fmt.Sprintf("examples/%s/%s.json", s.domain, id)

	var example TrainingExample
	if err := s.storage.LoadJSON(path, &example); err != nil {
		return nil, err
	}
	return &example, nil
}

// List lists all training examples for the domain.
func (s *TrainingExampleStore) List() ([]*TrainingExample, error) {
	dir := fmt.Sprintf("examples/%s", s.domain)
	files, err := s.storage.List(dir)
	if err != nil {
		return nil, err
	}

	var examples []*TrainingExample
	for _, file := range files {
		var example TrainingExample
		path := fmt.Sprintf("examples/%s/%s", s.domain, file)
		if err := s.storage.LoadJSON(path, &example); err != nil {
			continue
		}
		examples = append(examples, &example)
	}
	return examples, nil
}

// ListByCategory lists training examples by category.
func (s *TrainingExampleStore) ListByCategory(category string) ([]*TrainingExample, error) {
	examples, err := s.List()
	if err != nil {
		return nil, err
	}

	var filtered []*TrainingExample
	for _, e := range examples {
		if e.Category == category {
			filtered = append(filtered, e)
		}
	}
	return filtered, nil
}

// Delete removes a training example.
func (s *TrainingExampleStore) Delete(id string) error {
	path := fmt.Sprintf("examples/%s/%s.json", s.domain, id)
	return s.storage.Delete(path)
}

// Export exports all examples for the domain to a file.
func (s *TrainingExampleStore) Export(outputPath string) error {
	examples, err := s.List()
	if err != nil {
		return err
	}

	return s.storage.SaveJSON(outputPath, examples)
}

// GetFewShotExamples gets validated examples for few-shot prompting.
func (s *TrainingExampleStore) GetFewShotExamples(category string, limit int) ([]*TrainingExample, error) {
	examples, err := s.List()
	if err != nil {
		return nil, err
	}

	var validated []*TrainingExample
	for _, e := range examples {
		if e.Validated && (category == "" || e.Category == category) {
			validated = append(validated, e)
			if len(validated) >= limit {
				break
			}
		}
	}
	return validated, nil
}

// generateID generates a unique ID.
func generateID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano())
}
