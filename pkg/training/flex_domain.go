package training

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// FlexibleDomain represents a dynamic, configurable domain definition.
// This can be used to define ANY domain without hardcoding.
type FlexibleDomain struct {
	// Core identification
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	Version     string `json:"version"`

	// Domain type hints
	Category DomainCategory `json:"category"`
	Tags     []string       `json:"tags"`

	// Prompt configuration
	SystemPrompt       string           `json:"system_prompt"`
	UserPromptTemplate string           `json:"user_prompt_template"`
	Variables          []PromptVariable `json:"variables,omitempty"`

	// Input/Output schema
	InputSchema  *SchemaDefinition `json:"input_schema,omitempty"`
	OutputSchema *SchemaDefinition `json:"output_schema,omitempty"`

	// Training configuration
	Training FlexibleTrainingConfig `json:"training"`

	// Few-shot examples
	Examples []DomainExample `json:"examples,omitempty"`

	// Validation rules
	Validators []ValidatorRule `json:"validators,omitempty"`

	// Pre/Post processors
	Preprocessors  []ProcessorRule `json:"preprocessors,omitempty"`
	Postprocessors []ProcessorRule `json:"postprocessors,omitempty"`

	// Metadata
	Metadata map[string]any `json:"metadata,omitempty"`
}

// DomainCategory represents the type of domain.
type DomainCategory string

const (
	CategoryClassification DomainCategory = "classification"
	CategoryGeneration     DomainCategory = "generation"
	CategoryExtraction     DomainCategory = "extraction"
	CategoryQA             DomainCategory = "qa"
	CategoryConversation   DomainCategory = "conversation"
	CategorySummarization  DomainCategory = "summarization"
	CategoryTranslation    DomainCategory = "translation"
	CategoryCode           DomainCategory = "code"
	CategoryCustom         DomainCategory = "custom"
)

// PromptVariable defines a variable in prompts.
type PromptVariable struct {
	Name        string `json:"name"`
	Type        string `json:"type"` // string, number, boolean, array, object
	Required    bool   `json:"required"`
	Default     any    `json:"default,omitempty"`
	Description string `json:"description,omitempty"`
	EnumValues  []any  `json:"enum_values,omitempty"`
}

// SchemaDefinition defines input/output structure.
type SchemaDefinition struct {
	Type        string                        `json:"type"`
	Properties  map[string]*SchemaDefinition  `json:"properties,omitempty"`
	Items       *SchemaDefinition             `json:"items,omitempty"`
	Required    []string                      `json:"required,omitempty"`
	EnumValues  []any                         `json:"enum_values,omitempty"`
	Format      string                        `json:"format,omitempty"`
	MinLength   int                           `json:"min_length,omitempty"`
	MaxLength   int                           `json:"max_length,omitempty"`
	Minimum     float64                       `json:"minimum,omitempty"`
	Maximum     float64                       `json:"maximum,omitempty"`
	Pattern     string                        `json:"pattern,omitempty"`
	Description string                        `json:"description,omitempty"`
}

// FlexibleTrainingConfig configures domain-specific training.
type FlexibleTrainingConfig struct {
	// Dataset settings
	MinExamples     int     `json:"min_examples"`
	MaxExamples     int     `json:"max_examples"`
	TrainRatio      float64 `json:"train_ratio"`
	ValidationRatio float64 `json:"validation_ratio"`
	TestRatio       float64 `json:"test_ratio"`

	// Few-shot settings
	DefaultFewShot   int  `json:"default_few_shot"`
	MaxFewShot       int  `json:"max_few_shot"`
	DynamicSelection bool `json:"dynamic_selection"` // Select based on similarity

	// Quality thresholds
	MinQualityScore     float64 `json:"min_quality_score"`
	MinAccuracy         float64 `json:"min_accuracy"`
	ConfidenceThreshold float64 `json:"confidence_threshold"`

	// Optimization
	OptimizationGoal string   `json:"optimization_goal"` // accuracy, speed, balanced
	MetricsToTrack   []string `json:"metrics_to_track"`

	// Suggested hyperparameters
	SuggestedHyperparams *Hyperparameters `json:"suggested_hyperparams,omitempty"`
}

// DomainExample represents a training example.
type DomainExample struct {
	ID          string         `json:"id"`
	Input       string         `json:"input"`
	Output      string         `json:"output"`
	Label       string         `json:"label,omitempty"`
	Quality     float64        `json:"quality"`
	Category    string         `json:"category,omitempty"`
	Explanation string         `json:"explanation,omitempty"`
	Variables   map[string]any `json:"variables,omitempty"`
	Metadata    map[string]any `json:"metadata,omitempty"`
}

// ValidatorRule configures output validation.
type ValidatorRule struct {
	Type    string         `json:"type"` // json_schema, regex, length, contains, format, custom
	Params  map[string]any `json:"params"`
	Message string         `json:"message,omitempty"`
}

// ProcessorRule configures pre/post processing.
type ProcessorRule struct {
	Type     string         `json:"type"` // lowercase, trim, normalize, template, extract, format
	Params   map[string]any `json:"params,omitempty"`
	Priority int            `json:"priority,omitempty"`
}

// DomainRegistry manages multiple domain definitions.
type DomainRegistry struct {
	domains   map[string]*FlexibleDomain
	dataDir   string
	mu        sync.RWMutex
}

// NewDomainRegistry creates a new registry.
func NewDomainRegistry(dataDir string) *DomainRegistry {
	return &DomainRegistry{
		domains: make(map[string]*FlexibleDomain),
		dataDir: dataDir,
	}
}

// Register registers a domain definition.
func (r *DomainRegistry) Register(domain *FlexibleDomain) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if domain.ID == "" {
		return fmt.Errorf("domain ID is required")
	}
	if domain.Name == "" {
		domain.Name = domain.ID
	}

	// Set defaults
	r.applyDefaults(domain)

	r.domains[domain.ID] = domain
	return nil
}

func (r *DomainRegistry) applyDefaults(domain *FlexibleDomain) {
	if domain.Training.TrainRatio == 0 {
		domain.Training.TrainRatio = 0.8
	}
	if domain.Training.ValidationRatio == 0 {
		domain.Training.ValidationRatio = 0.1
	}
	if domain.Training.TestRatio == 0 {
		domain.Training.TestRatio = 0.1
	}
	if domain.Training.DefaultFewShot == 0 {
		domain.Training.DefaultFewShot = 3
	}
	if domain.Training.MaxFewShot == 0 {
		domain.Training.MaxFewShot = 10
	}
	if domain.Training.MinQualityScore == 0 {
		domain.Training.MinQualityScore = 0.5
	}
	if domain.Version == "" {
		domain.Version = "1.0.0"
	}
}

// Get retrieves a domain definition.
func (r *DomainRegistry) Get(id string) (*FlexibleDomain, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	domain, ok := r.domains[id]
	return domain, ok
}

// List returns all registered domain IDs.
func (r *DomainRegistry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	ids := make([]string, 0, len(r.domains))
	for id := range r.domains {
		ids = append(ids, id)
	}
	return ids
}

// ListByCategory returns domains matching a category.
func (r *DomainRegistry) ListByCategory(category DomainCategory) []*FlexibleDomain {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var result []*FlexibleDomain
	for _, domain := range r.domains {
		if domain.Category == category {
			result = append(result, domain)
		}
	}
	return result
}

// LoadFromFile loads a domain definition from JSON file.
func (r *DomainRegistry) LoadFromFile(path string) (*FlexibleDomain, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var domain FlexibleDomain
	if err := json.Unmarshal(data, &domain); err != nil {
		return nil, fmt.Errorf("failed to parse domain definition: %w", err)
	}

	if err := r.Register(&domain); err != nil {
		return nil, err
	}

	return &domain, nil
}

// LoadFromDirectory loads all domain definitions from a directory.
func (r *DomainRegistry) LoadFromDirectory(dir string) error {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}

	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}

		path := filepath.Join(dir, entry.Name())
		if _, err := r.LoadFromFile(path); err != nil {
			return fmt.Errorf("failed to load %s: %w", path, err)
		}
	}

	return nil
}

// SaveToFile saves a domain definition to JSON file.
func (r *DomainRegistry) SaveToFile(domain *FlexibleDomain, path string) error {
	data, err := json.MarshalIndent(domain, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// Clone creates a copy of a domain with a new ID.
func (r *DomainRegistry) Clone(sourceID, newID string) (*FlexibleDomain, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	source, ok := r.domains[sourceID]
	if !ok {
		return nil, fmt.Errorf("source domain %s not found", sourceID)
	}

	// Deep copy via JSON
	data, err := json.Marshal(source)
	if err != nil {
		return nil, err
	}

	var clone FlexibleDomain
	if err := json.Unmarshal(data, &clone); err != nil {
		return nil, err
	}

	clone.ID = newID
	r.domains[newID] = &clone

	return &clone, nil
}

// Delete removes a domain from the registry.
func (r *DomainRegistry) Delete(id string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.domains[id]; ok {
		delete(r.domains, id)
		return true
	}
	return false
}

// DomainRunner executes domain operations.
type DomainRunner struct {
	domain   *FlexibleDomain
	examples []DomainExample
}

// NewDomainRunner creates a runner for a domain.
func NewDomainRunner(domain *FlexibleDomain) *DomainRunner {
	examples := make([]DomainExample, len(domain.Examples))
	copy(examples, domain.Examples)

	return &DomainRunner{
		domain:   domain,
		examples: examples,
	}
}

// BuildPrompt constructs a complete prompt.
func (r *DomainRunner) BuildPrompt(input string, variables map[string]any, numFewShot int) string {
	var prompt strings.Builder

	// System prompt
	prompt.WriteString(r.domain.SystemPrompt)
	prompt.WriteString("\n\n")

	// Few-shot examples
	if numFewShot > 0 && len(r.examples) > 0 {
		examples := r.selectExamples(numFewShot, input)
		for _, ex := range examples {
			prompt.WriteString(fmt.Sprintf("Input: %s\nOutput: %s\n\n", ex.Input, ex.Output))
		}
	}

	// User prompt with variable substitution
	userPrompt := r.domain.UserPromptTemplate
	if userPrompt == "" {
		userPrompt = "{{input}}"
	}

	// Substitute variables
	userPrompt = strings.ReplaceAll(userPrompt, "{{input}}", input)
	for key, value := range variables {
		placeholder := fmt.Sprintf("{{%s}}", key)
		userPrompt = strings.ReplaceAll(userPrompt, placeholder, fmt.Sprintf("%v", value))
	}

	prompt.WriteString(userPrompt)

	return prompt.String()
}

func (r *DomainRunner) selectExamples(n int, input string) []DomainExample {
	if len(r.examples) <= n {
		return r.examples
	}

	// Sort by quality descending
	selected := make([]DomainExample, len(r.examples))
	copy(selected, r.examples)

	for i := 0; i < len(selected)-1; i++ {
		for j := i + 1; j < len(selected); j++ {
			if selected[j].Quality > selected[i].Quality {
				selected[i], selected[j] = selected[j], selected[i]
			}
		}
	}

	return selected[:n]
}

// Preprocess applies preprocessing rules.
func (r *DomainRunner) Preprocess(input string) string {
	result := input

	for _, pp := range r.domain.Preprocessors {
		switch pp.Type {
		case "lowercase":
			result = strings.ToLower(result)
		case "uppercase":
			result = strings.ToUpper(result)
		case "trim":
			result = strings.TrimSpace(result)
		case "normalize":
			result = strings.Join(strings.Fields(result), " ")
		case "replace":
			if old, ok := pp.Params["old"].(string); ok {
				if new, ok := pp.Params["new"].(string); ok {
					result = strings.ReplaceAll(result, old, new)
				}
			}
		case "prefix":
			if prefix, ok := pp.Params["text"].(string); ok {
				result = prefix + result
			}
		case "suffix":
			if suffix, ok := pp.Params["text"].(string); ok {
				result = result + suffix
			}
		}
	}

	return result
}

// Postprocess applies postprocessing rules.
func (r *DomainRunner) Postprocess(output string) string {
	result := output

	for _, pp := range r.domain.Postprocessors {
		switch pp.Type {
		case "trim":
			result = strings.TrimSpace(result)
		case "lowercase":
			result = strings.ToLower(result)
		case "uppercase":
			result = strings.ToUpper(result)
		case "extract_between":
			start, _ := pp.Params["start"].(string)
			end, _ := pp.Params["end"].(string)
			if start != "" {
				if idx := strings.Index(result, start); idx >= 0 {
					result = result[idx+len(start):]
				}
			}
			if end != "" {
				if idx := strings.Index(result, end); idx >= 0 {
					result = result[:idx]
				}
			}
		case "extract_json":
			// Find JSON in output
			startIdx := strings.Index(result, "{")
			endIdx := strings.LastIndex(result, "}")
			if startIdx >= 0 && endIdx > startIdx {
				result = result[startIdx : endIdx+1]
			}
		case "remove_prefix":
			if prefix, ok := pp.Params["text"].(string); ok {
				result = strings.TrimPrefix(result, prefix)
			}
		case "remove_suffix":
			if suffix, ok := pp.Params["text"].(string); ok {
				result = strings.TrimSuffix(result, suffix)
			}
		}
	}

	return result
}

// Validate validates output against rules.
func (r *DomainRunner) Validate(output string) []ValidationResult {
	var results []ValidationResult

	for _, v := range r.domain.Validators {
		result := ValidationResult{Rule: v.Type, Valid: true}

		switch v.Type {
		case "json_valid":
			if !json.Valid([]byte(output)) {
				result.Valid = false
				result.Message = "output is not valid JSON"
			}

		case "length":
			minLen, _ := v.Params["min"].(float64)
			maxLen, _ := v.Params["max"].(float64)

			if minLen > 0 && len(output) < int(minLen) {
				result.Valid = false
				result.Message = fmt.Sprintf("output too short (min: %d, got: %d)", int(minLen), len(output))
			}
			if maxLen > 0 && len(output) > int(maxLen) {
				result.Valid = false
				result.Message = fmt.Sprintf("output too long (max: %d, got: %d)", int(maxLen), len(output))
			}

		case "contains":
			if text, ok := v.Params["text"].(string); ok {
				if !strings.Contains(output, text) {
					result.Valid = false
					result.Message = fmt.Sprintf("output must contain: %s", text)
				}
			}

		case "not_empty":
			if strings.TrimSpace(output) == "" {
				result.Valid = false
				result.Message = "output cannot be empty"
			}

		case "starts_with":
			if prefix, ok := v.Params["text"].(string); ok {
				if !strings.HasPrefix(output, prefix) {
					result.Valid = false
					result.Message = fmt.Sprintf("output must start with: %s", prefix)
				}
			}

		case "ends_with":
			if suffix, ok := v.Params["text"].(string); ok {
				if !strings.HasSuffix(output, suffix) {
					result.Valid = false
					result.Message = fmt.Sprintf("output must end with: %s", suffix)
				}
			}

		case "enum":
			if allowed, ok := v.Params["values"].([]any); ok {
				valid := false
				for _, a := range allowed {
					if fmt.Sprintf("%v", a) == strings.TrimSpace(output) {
						valid = true
						break
					}
				}
				if !valid {
					result.Valid = false
					result.Message = fmt.Sprintf("output must be one of: %v", allowed)
				}
			}
		}

		if v.Message != "" && !result.Valid {
			result.Message = v.Message
		}

		results = append(results, result)
	}

	return results
}

// ValidationResult holds validation result.
type ValidationResult struct {
	Rule    string `json:"rule"`
	Valid   bool   `json:"valid"`
	Message string `json:"message,omitempty"`
}

// IsValid returns true if all validations passed.
func (r *DomainRunner) IsValid(output string) bool {
	results := r.Validate(output)
	for _, res := range results {
		if !res.Valid {
			return false
		}
	}
	return true
}

// AddExample adds a training example.
func (r *DomainRunner) AddExample(example DomainExample) {
	r.examples = append(r.examples, example)
}

// GetExamples returns all examples.
func (r *DomainRunner) GetExamples() []DomainExample {
	return r.examples
}

// ToDataPoints converts examples to DataPoints for training.
func (r *DomainRunner) ToDataPoints() []DataPoint {
	points := make([]DataPoint, len(r.examples))

	for i, ex := range r.examples {
		points[i] = DataPoint{
			ID:       ex.ID,
			Input:    ex.Input,
			Output:   ex.Output,
			Category: ex.Category,
			Quality:  ex.Quality,
			Metadata: ex.Metadata,
		}
	}

	return points
}

// GetSuggestedHyperparams returns suggested hyperparameters.
func (r *DomainRunner) GetSuggestedHyperparams() *Hyperparameters {
	if r.domain.Training.SuggestedHyperparams != nil {
		return r.domain.Training.SuggestedHyperparams
	}

	// Return category-specific defaults
	switch r.domain.Category {
	case CategoryClassification:
		return &Hyperparameters{
			LearningRate: 2e-5,
			BatchSize:    16,
			Epochs:       5,
			WarmupSteps:  100,
		}
	case CategoryGeneration:
		return &Hyperparameters{
			LearningRate: 1e-5,
			BatchSize:    8,
			Epochs:       10,
			WarmupSteps:  200,
		}
	case CategoryExtraction:
		return &Hyperparameters{
			LearningRate: 2e-5,
			BatchSize:    16,
			Epochs:       5,
			WarmupSteps:  100,
		}
	default:
		return DefaultHyperparameters()
	}
}

// ============================================
// Domain Templates for Quick Start
// ============================================

// NewClassificationDomain creates a classification domain.
func NewClassificationDomain(id, name string, classes []string) *FlexibleDomain {
	classesStr := strings.Join(classes, ", ")
	enumValues := make([]any, len(classes))
	for i, c := range classes {
		enumValues[i] = c
	}

	return &FlexibleDomain{
		ID:       id,
		Name:     name,
		Category: CategoryClassification,
		Description: fmt.Sprintf("Classify input into one of: %s", classesStr),
		SystemPrompt: fmt.Sprintf(`You are a classification assistant.
Classify the given input into exactly one of these categories: %s

Respond with ONLY the category name, nothing else.`, classesStr),
		UserPromptTemplate: "Classify the following:\n\n{{input}}",
		OutputSchema: &SchemaDefinition{
			Type:       "string",
			EnumValues: enumValues,
		},
		Validators: []ValidatorRule{
			{Type: "not_empty"},
			{Type: "enum", Params: map[string]any{"values": enumValues}},
		},
		Postprocessors: []ProcessorRule{
			{Type: "trim"},
		},
		Training: FlexibleTrainingConfig{
			MinExamples:     10,
			TrainRatio:      0.8,
			ValidationRatio: 0.1,
			TestRatio:       0.1,
			DefaultFewShot:  3,
			MaxFewShot:      5,
			OptimizationGoal: "accuracy",
			MetricsToTrack:  []string{"accuracy", "f1_score", "precision", "recall"},
		},
	}
}

// NewGenerationDomain creates a text generation domain.
func NewGenerationDomain(id, name, description, systemPrompt string) *FlexibleDomain {
	return &FlexibleDomain{
		ID:          id,
		Name:        name,
		Category:    CategoryGeneration,
		Description: description,
		SystemPrompt: systemPrompt,
		UserPromptTemplate: "{{input}}",
		Validators: []ValidatorRule{
			{Type: "not_empty"},
			{Type: "length", Params: map[string]any{"min": 10.0}},
		},
		Postprocessors: []ProcessorRule{
			{Type: "trim"},
		},
		Training: FlexibleTrainingConfig{
			MinExamples:     20,
			TrainRatio:      0.8,
			ValidationRatio: 0.1,
			TestRatio:       0.1,
			DefaultFewShot:  2,
			MaxFewShot:      5,
			OptimizationGoal: "balanced",
			MetricsToTrack:  []string{"bleu", "coherence"},
		},
	}
}

// NewExtractionDomain creates a data extraction domain.
func NewExtractionDomain(id, name string, fields []string) *FlexibleDomain {
	props := make(map[string]*SchemaDefinition)
	for _, field := range fields {
		props[field] = &SchemaDefinition{Type: "string"}
	}

	return &FlexibleDomain{
		ID:          id,
		Name:        name,
		Category:    CategoryExtraction,
		Description: fmt.Sprintf("Extract %s from input", strings.Join(fields, ", ")),
		SystemPrompt: fmt.Sprintf(`You are a data extraction assistant.
Extract the following fields from the input: %s

Respond in valid JSON format with these exact field names.`, strings.Join(fields, ", ")),
		UserPromptTemplate: "Extract information from:\n\n{{input}}",
		OutputSchema: &SchemaDefinition{
			Type:       "object",
			Properties: props,
			Required:   fields,
		},
		Validators: []ValidatorRule{
			{Type: "not_empty"},
			{Type: "json_valid"},
		},
		Postprocessors: []ProcessorRule{
			{Type: "trim"},
			{Type: "extract_json"},
		},
		Training: FlexibleTrainingConfig{
			MinExamples:     15,
			TrainRatio:      0.8,
			ValidationRatio: 0.1,
			TestRatio:       0.1,
			DefaultFewShot:  3,
			MaxFewShot:      5,
			OptimizationGoal: "accuracy",
			MetricsToTrack:  []string{"accuracy", "exact_match"},
		},
	}
}

// NewQADomain creates a question-answering domain.
func NewQADomain(id, name, contextDescription string) *FlexibleDomain {
	return &FlexibleDomain{
		ID:          id,
		Name:        name,
		Category:    CategoryQA,
		Description: fmt.Sprintf("Answer questions about %s", contextDescription),
		SystemPrompt: fmt.Sprintf(`You are a knowledgeable assistant specialized in %s.
Answer questions accurately and concisely based on the provided context.
If you don't know the answer, say "I don't know" rather than making something up.`, contextDescription),
		UserPromptTemplate: `Context: {{context}}

Question: {{input}}`,
		Variables: []PromptVariable{
			{Name: "context", Type: "string", Required: true, Description: "Background context for the question"},
		},
		Validators: []ValidatorRule{
			{Type: "not_empty"},
		},
		Postprocessors: []ProcessorRule{
			{Type: "trim"},
		},
		Training: FlexibleTrainingConfig{
			MinExamples:      30,
			TrainRatio:       0.8,
			ValidationRatio:  0.1,
			TestRatio:        0.1,
			DefaultFewShot:   5,
			MaxFewShot:       10,
			DynamicSelection: true,
			OptimizationGoal: "accuracy",
			MetricsToTrack:   []string{"accuracy", "f1_score"},
		},
	}
}

// NewSummarizationDomain creates a summarization domain.
func NewSummarizationDomain(id, name string, maxLength int) *FlexibleDomain {
	return &FlexibleDomain{
		ID:          id,
		Name:        name,
		Category:    CategorySummarization,
		Description: "Summarize text concisely",
		SystemPrompt: fmt.Sprintf(`You are a summarization assistant.
Provide concise, accurate summaries of the given text.
Keep summaries under %d characters while retaining key information.`, maxLength),
		UserPromptTemplate: "Summarize the following:\n\n{{input}}",
		Validators: []ValidatorRule{
			{Type: "not_empty"},
			{Type: "length", Params: map[string]any{"max": float64(maxLength)}},
		},
		Postprocessors: []ProcessorRule{
			{Type: "trim"},
		},
		Training: FlexibleTrainingConfig{
			MinExamples:     20,
			TrainRatio:      0.8,
			ValidationRatio: 0.1,
			TestRatio:       0.1,
			DefaultFewShot:  2,
			MaxFewShot:      3,
			OptimizationGoal: "balanced",
			MetricsToTrack:  []string{"rouge_1", "rouge_2", "rouge_l"},
		},
	}
}

// NewCodeDomain creates a code generation domain.
func NewCodeDomain(id, name, language string) *FlexibleDomain {
	return &FlexibleDomain{
		ID:          id,
		Name:        name,
		Category:    CategoryCode,
		Description: fmt.Sprintf("Generate %s code", language),
		Tags:        []string{"code", language},
		SystemPrompt: fmt.Sprintf(`You are an expert %s programmer.
Generate clean, well-documented, and efficient code.
Follow best practices and coding standards for %s.
Include comments explaining complex logic.`, language, language),
		UserPromptTemplate: "{{input}}",
		Validators: []ValidatorRule{
			{Type: "not_empty"},
		},
		Postprocessors: []ProcessorRule{
			{Type: "trim"},
		},
		Training: FlexibleTrainingConfig{
			MinExamples:     30,
			TrainRatio:      0.8,
			ValidationRatio: 0.1,
			TestRatio:       0.1,
			DefaultFewShot:  2,
			MaxFewShot:      5,
			OptimizationGoal: "balanced",
			MetricsToTrack:  []string{"bleu", "exact_match"},
		},
	}
}

// NewCustomDomain creates a fully custom domain.
func NewCustomDomain(id, name, description, systemPrompt, userTemplate string) *FlexibleDomain {
	return &FlexibleDomain{
		ID:                 id,
		Name:               name,
		Category:           CategoryCustom,
		Description:        description,
		SystemPrompt:       systemPrompt,
		UserPromptTemplate: userTemplate,
		Training: FlexibleTrainingConfig{
			MinExamples:     10,
			TrainRatio:      0.8,
			ValidationRatio: 0.1,
			TestRatio:       0.1,
			DefaultFewShot:  3,
			MaxFewShot:      10,
			OptimizationGoal: "balanced",
			MetricsToTrack:  []string{"accuracy"},
		},
	}
}
