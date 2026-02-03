package domains

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/sujit/ai-agent/pkg/content"
	"github.com/sujit/ai-agent/pkg/llm"
	"github.com/sujit/ai-agent/pkg/storage"
)

// BaseDomain provides common functionality for all domain trainers.
type BaseDomain struct {
	ID       string
	Name     string
	provider llm.MultimodalProvider
	storage  *storage.Storage
	examples *storage.TrainingExampleStore
	config   *DomainConfig
	mu       sync.RWMutex
}

// DomainConfig holds configuration for a domain.
type DomainConfig struct {
	ID              string          `json:"id"`
	Name            string          `json:"name"`
	Description     string          `json:"description"`
	SystemPrompt    string          `json:"system_prompt"`
	Categories      []string        `json:"categories"`
	DefaultModel    string          `json:"default_model"`
	Temperature     float64         `json:"temperature"`
	MaxTokens       int             `json:"max_tokens"`
	FewShotCount    int             `json:"few_shot_count"`
	EnabledFeatures map[string]bool `json:"enabled_features"`
	CustomSettings  map[string]any  `json:"custom_settings"`
}

// FewShotExample represents a training example for few-shot learning.
type FewShotExample struct {
	Input    string            `json:"input"`
	Output   string            `json:"output"`
	Category string            `json:"category,omitempty"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// NewBaseDomain creates a new base domain.
func NewBaseDomain(id, name string, provider llm.MultimodalProvider, storage *storage.Storage) *BaseDomain {
	bd := &BaseDomain{
		ID:       id,
		Name:     name,
		provider: provider,
		storage:  storage,
	}

	// Initialize examples store
	if storage != nil {
		bd.examples = storage.Examples(id)
	}

	// Load config if exists
	bd.loadConfig()

	return bd
}

// loadConfig loads domain configuration from storage.
func (d *BaseDomain) loadConfig() {
	if d.storage == nil {
		return
	}

	configPath := filepath.Join("domains", d.ID, "config.json")
	var config DomainConfig
	if err := d.storage.LoadJSON(configPath, &config); err == nil {
		d.config = &config
	} else {
		// Create default config
		d.config = &DomainConfig{
			ID:           d.ID,
			Name:         d.Name,
			Temperature:  0.7,
			MaxTokens:    2048,
			FewShotCount: 3,
		}
	}
}

// SaveConfig saves domain configuration to storage.
func (d *BaseDomain) SaveConfig() error {
	if d.storage == nil || d.config == nil {
		return nil
	}

	configPath := filepath.Join("domains", d.ID, "config.json")
	return d.storage.SaveJSON(configPath, d.config)
}

// GetID returns the domain ID.
func (d *BaseDomain) GetID() string {
	return d.ID
}

// GetName returns the domain name.
func (d *BaseDomain) GetName() string {
	return d.Name
}

// GetConfig returns the domain configuration.
func (d *BaseDomain) GetConfig() *DomainConfig {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.config
}

// UpdateConfig updates domain configuration.
func (d *BaseDomain) UpdateConfig(updates map[string]any) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.config == nil {
		d.config = &DomainConfig{ID: d.ID, Name: d.Name}
	}

	// Apply updates
	for key, value := range updates {
		switch key {
		case "description":
			d.config.Description = value.(string)
		case "system_prompt":
			d.config.SystemPrompt = value.(string)
		case "temperature":
			d.config.Temperature = value.(float64)
		case "max_tokens":
			d.config.MaxTokens = value.(int)
		case "few_shot_count":
			d.config.FewShotCount = value.(int)
		case "default_model":
			d.config.DefaultModel = value.(string)
		}
	}

	return d.SaveConfig()
}

// GetFewShotExamples retrieves few-shot examples for a category.
func (d *BaseDomain) GetFewShotExamples(category string, count int) ([]*FewShotExample, error) {
	if d.examples == nil {
		return nil, nil
	}

	if count <= 0 {
		if d.config != nil && d.config.FewShotCount > 0 {
			count = d.config.FewShotCount
		} else {
			count = 3
		}
	}

	trainingExamples, err := d.examples.GetFewShotExamples(category, count)
	if err != nil {
		return nil, err
	}

	var examples []*FewShotExample
	for _, te := range trainingExamples {
		examples = append(examples, &FewShotExample{
			Input:    te.Input,
			Output:   te.Output,
			Category: te.Category,
		})
	}

	return examples, nil
}

// AddTrainingExample adds a new training example.
func (d *BaseDomain) AddTrainingExample(input, output, category string, quality float64) error {
	if d.examples == nil {
		return fmt.Errorf("storage not initialized")
	}

	example := &storage.TrainingExample{
		Domain:    d.ID,
		Category:  category,
		Input:     input,
		Output:    output,
		Quality:   quality,
		Validated: quality >= 0.8,
		Metadata:  make(map[string]any),
	}

	return d.examples.Add(example)
}

// GetTrainingExamples lists training examples for the domain.
func (d *BaseDomain) GetTrainingExamples(category string) ([]*storage.TrainingExample, error) {
	if d.examples == nil {
		return nil, fmt.Errorf("storage not initialized")
	}

	if category != "" {
		return d.examples.ListByCategory(category)
	}
	return d.examples.List()
}

// ExportExamples exports all training examples for the domain.
func (d *BaseDomain) ExportExamples(outputPath string) error {
	if d.examples == nil {
		return fmt.Errorf("storage not initialized")
	}

	return d.examples.Export(outputPath)
}

// ImportExamples imports training examples from a file.
func (d *BaseDomain) ImportExamples(inputPath string) error {
	if d.examples == nil {
		return fmt.Errorf("storage not initialized")
	}

	// Read the file
	data, err := os.ReadFile(inputPath)
	if err != nil {
		return err
	}

	var examples []*storage.TrainingExample
	if err := json.Unmarshal(data, &examples); err != nil {
		return err
	}

	// Add each example
	for _, ex := range examples {
		ex.Domain = d.ID
		if err := d.examples.Add(ex); err != nil {
			return err
		}
	}

	return nil
}

// GetProvider returns the LLM provider.
func (d *BaseDomain) GetProvider() llm.MultimodalProvider {
	return d.provider
}

// GetStorage returns the storage instance.
func (d *BaseDomain) GetStorage() *storage.Storage {
	return d.storage
}

// Generate performs a basic generation with the domain context.
func (d *BaseDomain) Generate(ctx context.Context, prompt string, category string) (string, error) {
	if d.provider == nil {
		return "", fmt.Errorf("provider not initialized")
	}

	// Get few-shot examples
	examples, _ := d.GetFewShotExamples(category, 0)

	// Build messages
	messages := d.buildMessages(prompt, examples)

	// Generate config
	config := &llm.GenerationConfig{
		Temperature: d.config.Temperature,
		MaxTokens:   d.config.MaxTokens,
	}

	// Call provider
	resp, err := d.provider.Generate(ctx, messages, config)
	if err != nil {
		return "", err
	}

	return resp.Message.GetText(), nil
}

// buildMessages constructs message list with system prompt and examples.
func (d *BaseDomain) buildMessages(prompt string, examples []*FewShotExample) []*content.Message {
	var messages []*content.Message

	// Add system prompt
	systemPrompt := d.getSystemPrompt()
	if systemPrompt != "" {
		messages = append(messages, content.NewTextMessage(content.RoleSystem, systemPrompt))
	}

	// Add few-shot examples
	for _, ex := range examples {
		messages = append(messages,
			content.NewTextMessage(content.RoleUser, ex.Input),
			content.NewTextMessage(content.RoleAssistant, ex.Output),
		)
	}

	// Add user prompt
	messages = append(messages, content.NewTextMessage(content.RoleUser, prompt))

	return messages
}

// getSystemPrompt returns the system prompt for the domain.
func (d *BaseDomain) getSystemPrompt() string {
	if d.config != nil && d.config.SystemPrompt != "" {
		return d.config.SystemPrompt
	}
	return fmt.Sprintf("You are an expert assistant for the %s domain.", d.Name)
}

// GetSystemPrompt returns the system prompt (implements Domain interface).
func (d *BaseDomain) GetSystemPrompt() string {
	return d.getSystemPrompt()
}

// TrainingSession represents an active training/feedback session.
type TrainingSession struct {
	ID        string            `json:"id"`
	DomainID  string            `json:"domain_id"`
	StartedAt time.Time         `json:"started_at"`
	Examples  []*FewShotExample `json:"examples"`
	Feedback  []*Feedback       `json:"feedback"`
}

// Feedback represents user feedback on a generation.
type Feedback struct {
	Input      string    `json:"input"`
	Output     string    `json:"output"`
	Rating     int       `json:"rating"` // 1-5
	Correction string    `json:"correction,omitempty"`
	Comments   string    `json:"comments,omitempty"`
	CreatedAt  time.Time `json:"created_at"`
}

// StartTrainingSession begins a new training session.
func (d *BaseDomain) StartTrainingSession() *TrainingSession {
	return &TrainingSession{
		ID:        fmt.Sprintf("%s_%d", d.ID, time.Now().UnixNano()),
		DomainID:  d.ID,
		StartedAt: time.Now(),
		Examples:  []*FewShotExample{},
		Feedback:  []*Feedback{},
	}
}

// RecordFeedback records feedback for a generation.
func (d *BaseDomain) RecordFeedback(session *TrainingSession, input, output string, rating int, correction, comments string) {
	fb := &Feedback{
		Input:      input,
		Output:     output,
		Rating:     rating,
		Correction: correction,
		Comments:   comments,
		CreatedAt:  time.Now(),
	}
	session.Feedback = append(session.Feedback, fb)

	// If feedback is positive or has a correction, save as training example
	if rating >= 4 || correction != "" {
		finalOutput := output
		if correction != "" {
			finalOutput = correction
		}
		quality := float64(rating) / 5.0
		d.AddTrainingExample(input, finalOutput, "", quality)
	}
}

// SaveSession persists a training session.
func (d *BaseDomain) SaveSession(session *TrainingSession) error {
	if d.storage == nil {
		return nil
	}

	sessionPath := filepath.Join("domains", d.ID, "sessions", session.ID+".json")
	return d.storage.SaveJSON(sessionPath, session)
}
