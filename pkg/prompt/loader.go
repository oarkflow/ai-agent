package prompt

import (
	"bytes"
	"fmt"
	"text/template"
	"time"

	"github.com/sujit/ai-agent/pkg/config"
	"github.com/sujit/ai-agent/pkg/llm"
)

// PromptLoader loads prompt templates from configuration.
type PromptLoader struct {
	config *config.Config
}

// NewPromptLoader creates a new prompt loader.
func NewPromptLoader(cfg *config.Config) *PromptLoader {
	return &PromptLoader{config: cfg}
}

// LoadTemplates loads all prompt templates from configuration.
func (pl *PromptLoader) LoadTemplates() map[string]*PromptTemplate {
	templates := make(map[string]*PromptTemplate)

	if pl.config.Prompts == nil {
		return templates
	}

	for id, promptCfg := range pl.config.Prompts.Prompts {
		templates[id] = pl.convertPromptCfg(promptCfg)
	}

	return templates
}

// LoadTemplate loads a single prompt template by ID.
func (pl *PromptLoader) LoadTemplate(id string) (*PromptTemplate, bool) {
	promptCfg, ok := pl.config.GetPrompt(id)
	if !ok {
		return nil, false
	}

	return pl.convertPromptCfg(promptCfg), true
}

// convertPromptCfg converts a config.PromptCfg to PromptTemplate.
func (pl *PromptLoader) convertPromptCfg(cfg *config.PromptCfg) *PromptTemplate {
	template := &PromptTemplate{
		ID:          cfg.ID,
		Name:        cfg.Name,
		Description: cfg.Description,
		Version:     cfg.Version,
		Category:    PromptCategory(cfg.Category),
		Template:    cfg.Template,
		Variables:   make([]TemplateVar, 0, len(cfg.Variables)),
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}

	// Convert variables
	for _, v := range cfg.Variables {
		template.Variables = append(template.Variables, TemplateVar{
			Name:        v.Name,
			Type:        v.Type,
			Required:    v.Required,
			Default:     v.Default,
			Description: v.Description,
			Enum:        v.Enum,
		})
	}

	// Convert config if present
	if cfg.Config != nil {
		template.Config = &PromptConfig{
			PreferredProvider: llm.ProviderType(cfg.Config.PreferredProvider),
			PreferredModel:    cfg.Config.PreferredModel,
			Temperature:       cfg.Config.Temperature,
			MaxTokens:         cfg.Config.MaxTokens,
			OutputFormat:      OutputFormat(cfg.Config.OutputFormat),
		}
	}

	return template
}

// NewPromptLibraryFromConfig creates a prompt library from configuration.
func NewPromptLibraryFromConfig(cfg *config.Config) *PromptLibrary {
	lib := &PromptLibrary{
		templates: make(map[string]*PromptTemplate),
	}

	loader := NewPromptLoader(cfg)
	lib.templates = loader.LoadTemplates()

	return lib
}

// PromptRenderer renders prompts with variable substitution.
type PromptRenderer struct {
	cache map[string]*template.Template
}

// NewPromptRenderer creates a new prompt renderer.
func NewPromptRenderer() *PromptRenderer {
	return &PromptRenderer{
		cache: make(map[string]*template.Template),
	}
}

// Render renders a prompt template with the given variables.
func (pr *PromptRenderer) Render(pt *PromptTemplate, vars map[string]any) (string, error) {
	// Check cache
	tmpl, ok := pr.cache[pt.ID]
	if !ok {
		// Parse and cache template
		var err error
		tmpl, err = template.New(pt.ID).Parse(pt.Template)
		if err != nil {
			return "", fmt.Errorf("failed to parse template: %w", err)
		}
		pr.cache[pt.ID] = tmpl
	}

	// Apply defaults for missing variables
	mergedVars := make(map[string]any)
	for _, v := range pt.Variables {
		if v.Default != "" {
			mergedVars[v.Name] = v.Default
		}
	}
	for k, v := range vars {
		mergedVars[k] = v
	}

	// Validate required variables
	for _, v := range pt.Variables {
		if v.Required {
			if _, ok := mergedVars[v.Name]; !ok {
				return "", fmt.Errorf("missing required variable: %s", v.Name)
			}
		}
	}

	// Render template
	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, mergedVars); err != nil {
		return "", fmt.Errorf("failed to execute template: %w", err)
	}

	return buf.String(), nil
}

// ClearCache clears the template cache.
func (pr *PromptRenderer) ClearCache() {
	pr.cache = make(map[string]*template.Template)
}
