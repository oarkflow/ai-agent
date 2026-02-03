package prompt

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"text/template"
	"time"

	"github.com/sujit/ai-agent/pkg/content"
	"github.com/sujit/ai-agent/pkg/llm"
)

// PromptTemplate represents a reusable prompt template with metadata.
type PromptTemplate struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Version     string            `json:"version"`
	Category    PromptCategory    `json:"category"`
	Template    string            `json:"template"`
	Variables   []TemplateVar     `json:"variables"`
	Examples    []PromptExample   `json:"examples"`
	Config      *PromptConfig     `json:"config,omitempty"`
	Metadata    map[string]string `json:"metadata,omitempty"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
}

// PromptCategory defines the type of prompt.
type PromptCategory string

const (
	CategoryChat          PromptCategory = "chat"
	CategoryCodeGen       PromptCategory = "code_generation"
	CategoryCodeReview    PromptCategory = "code_review"
	CategorySummarization PromptCategory = "summarization"
	CategoryAnalysis      PromptCategory = "analysis"
	CategoryCreative      PromptCategory = "creative"
	CategoryExtraction    PromptCategory = "extraction"
	CategoryTranslation   PromptCategory = "translation"
	CategoryQA            PromptCategory = "question_answering"
	CategoryReasoning     PromptCategory = "reasoning"
	CategoryInstruction   PromptCategory = "instruction"
	CategoryMultimodal    PromptCategory = "multimodal"
)

// TemplateVar describes a variable in the template.
type TemplateVar struct {
	Name        string   `json:"name"`
	Type        string   `json:"type"`
	Required    bool     `json:"required"`
	Default     string   `json:"default,omitempty"`
	Description string   `json:"description"`
	Enum        []string `json:"enum,omitempty"`
}

// PromptExample shows example usage.
type PromptExample struct {
	Input    map[string]any `json:"input"`
	Output   string         `json:"output"`
	Quality  float64        `json:"quality"`
	Notes    string         `json:"notes,omitempty"`
}

// PromptConfig holds prompt-specific configuration.
type PromptConfig struct {
	PreferredModel   string             `json:"preferred_model,omitempty"`
	PreferredProvider llm.ProviderType  `json:"preferred_provider,omitempty"`
	Temperature       float64           `json:"temperature,omitempty"`
	MaxTokens        int                `json:"max_tokens,omitempty"`
	TopP             float64            `json:"top_p,omitempty"`
	StopSequences    []string           `json:"stop_sequences,omitempty"`
	SystemPrompt     string             `json:"system_prompt,omitempty"`
	OutputFormat     OutputFormat       `json:"output_format,omitempty"`
}

// OutputFormat specifies the expected output structure.
type OutputFormat string

const (
	FormatText     OutputFormat = "text"
	FormatJSON     OutputFormat = "json"
	FormatMarkdown OutputFormat = "markdown"
	FormatCode     OutputFormat = "code"
	FormatList     OutputFormat = "list"
	FormatTable    OutputFormat = "table"
)

// Render renders the template with given variables.
func (pt *PromptTemplate) Render(vars map[string]any) (string, error) {
	// Apply defaults for missing variables
	for _, v := range pt.Variables {
		if _, ok := vars[v.Name]; !ok && v.Default != "" {
			vars[v.Name] = v.Default
		}
	}

	// Validate required variables
	for _, v := range pt.Variables {
		if v.Required {
			if _, ok := vars[v.Name]; !ok {
				return "", fmt.Errorf("missing required variable: %s", v.Name)
			}
		}
	}

	// Parse and execute template
	tmpl, err := template.New(pt.ID).Parse(pt.Template)
	if err != nil {
		return "", fmt.Errorf("failed to parse template: %w", err)
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, vars); err != nil {
		return "", fmt.Errorf("failed to execute template: %w", err)
	}

	return buf.String(), nil
}

// ToMessage creates a content.Message from the rendered template.
func (pt *PromptTemplate) ToMessage(role content.Role, vars map[string]any) (*content.Message, error) {
	text, err := pt.Render(vars)
	if err != nil {
		return nil, err
	}
	return content.NewTextMessage(role, text), nil
}

// PromptBuilder provides a fluent API for building prompts.
type PromptBuilder struct {
	systemPrompt   string
	instructions   []string
	context        []string
	examples       []FewShotExample
	constraints    []string
	outputFormat   OutputFormat
	outputSchema   string
	chainOfThought bool
	rolePlay       string
	persona        string
}

// FewShotExample represents an input-output example for few-shot learning.
type FewShotExample struct {
	Input  string
	Output string
	Label  string
}

// NewPromptBuilder creates a new prompt builder.
func NewPromptBuilder() *PromptBuilder {
	return &PromptBuilder{
		instructions: make([]string, 0),
		context:      make([]string, 0),
		examples:     make([]FewShotExample, 0),
		constraints:  make([]string, 0),
		outputFormat: FormatText,
	}
}

// WithSystem sets the system prompt.
func (pb *PromptBuilder) WithSystem(prompt string) *PromptBuilder {
	pb.systemPrompt = prompt
	return pb
}

// WithPersona sets a persona for the AI.
func (pb *PromptBuilder) WithPersona(persona string) *PromptBuilder {
	pb.persona = persona
	return pb
}

// WithRolePlay sets a role-playing context.
func (pb *PromptBuilder) WithRolePlay(role string) *PromptBuilder {
	pb.rolePlay = role
	return pb
}

// AddInstruction adds an instruction.
func (pb *PromptBuilder) AddInstruction(instruction string) *PromptBuilder {
	pb.instructions = append(pb.instructions, instruction)
	return pb
}

// AddContext adds contextual information.
func (pb *PromptBuilder) AddContext(ctx string) *PromptBuilder {
	pb.context = append(pb.context, ctx)
	return pb
}

// AddExample adds a few-shot example.
func (pb *PromptBuilder) AddExample(input, output string) *PromptBuilder {
	pb.examples = append(pb.examples, FewShotExample{Input: input, Output: output})
	return pb
}

// AddLabeledExample adds a labeled few-shot example.
func (pb *PromptBuilder) AddLabeledExample(input, output, label string) *PromptBuilder {
	pb.examples = append(pb.examples, FewShotExample{Input: input, Output: output, Label: label})
	return pb
}

// AddConstraint adds an output constraint.
func (pb *PromptBuilder) AddConstraint(constraint string) *PromptBuilder {
	pb.constraints = append(pb.constraints, constraint)
	return pb
}

// WithOutputFormat sets the expected output format.
func (pb *PromptBuilder) WithOutputFormat(format OutputFormat) *PromptBuilder {
	pb.outputFormat = format
	return pb
}

// WithJSONSchema sets JSON output with a schema.
func (pb *PromptBuilder) WithJSONSchema(schema string) *PromptBuilder {
	pb.outputFormat = FormatJSON
	pb.outputSchema = schema
	return pb
}

// WithChainOfThought enables chain-of-thought prompting.
func (pb *PromptBuilder) WithChainOfThought() *PromptBuilder {
	pb.chainOfThought = true
	return pb
}

// BuildSystemPrompt generates the system prompt.
func (pb *PromptBuilder) BuildSystemPrompt() string {
	var parts []string

	// Persona/Role
	if pb.persona != "" {
		parts = append(parts, fmt.Sprintf("You are %s.", pb.persona))
	}
	if pb.rolePlay != "" {
		parts = append(parts, fmt.Sprintf("Act as %s.", pb.rolePlay))
	}
	if pb.systemPrompt != "" {
		parts = append(parts, pb.systemPrompt)
	}

	// Instructions
	if len(pb.instructions) > 0 {
		parts = append(parts, "\nInstructions:")
		for i, inst := range pb.instructions {
			parts = append(parts, fmt.Sprintf("%d. %s", i+1, inst))
		}
	}

	// Constraints
	if len(pb.constraints) > 0 {
		parts = append(parts, "\nConstraints:")
		for _, c := range pb.constraints {
			parts = append(parts, fmt.Sprintf("- %s", c))
		}
	}

	// Output format
	switch pb.outputFormat {
	case FormatJSON:
		if pb.outputSchema != "" {
			parts = append(parts, fmt.Sprintf("\nRespond in valid JSON matching this schema:\n%s", pb.outputSchema))
		} else {
			parts = append(parts, "\nRespond in valid JSON format.")
		}
	case FormatMarkdown:
		parts = append(parts, "\nFormat your response using Markdown.")
	case FormatCode:
		parts = append(parts, "\nProvide code in a properly formatted code block with the language specified.")
	case FormatList:
		parts = append(parts, "\nFormat your response as a bulleted or numbered list.")
	case FormatTable:
		parts = append(parts, "\nFormat your response as a table.")
	}

	// Chain of thought
	if pb.chainOfThought {
		parts = append(parts, "\nThink step by step. Show your reasoning process before providing the final answer.")
	}

	return strings.Join(parts, "\n")
}

// BuildUserPrompt generates a user prompt with context and task.
func (pb *PromptBuilder) BuildUserPrompt(task string) string {
	var parts []string

	// Context
	if len(pb.context) > 0 {
		parts = append(parts, "Context:")
		for _, ctx := range pb.context {
			parts = append(parts, ctx)
		}
		parts = append(parts, "")
	}

	// Examples
	if len(pb.examples) > 0 {
		parts = append(parts, "Examples:")
		for _, ex := range pb.examples {
			if ex.Label != "" {
				parts = append(parts, fmt.Sprintf("\nExample (%s):", ex.Label))
			} else {
				parts = append(parts, "\nExample:")
			}
			parts = append(parts, fmt.Sprintf("Input: %s", ex.Input))
			parts = append(parts, fmt.Sprintf("Output: %s", ex.Output))
		}
		parts = append(parts, "")
	}

	// Task
	parts = append(parts, "Task:")
	parts = append(parts, task)

	return strings.Join(parts, "\n")
}

// Build returns both system and user prompts.
func (pb *PromptBuilder) Build(task string) (system, user string) {
	return pb.BuildSystemPrompt(), pb.BuildUserPrompt(task)
}

// PromptLibrary manages a collection of prompt templates.
type PromptLibrary struct {
	templates map[string]*PromptTemplate
}

// NewPromptLibrary creates a new prompt library.
func NewPromptLibrary() *PromptLibrary {
	lib := &PromptLibrary{
		templates: make(map[string]*PromptTemplate),
	}
	lib.loadBuiltinTemplates()
	return lib
}

// Add adds a template to the library.
func (pl *PromptLibrary) Add(template *PromptTemplate) {
	template.CreatedAt = time.Now()
	template.UpdatedAt = time.Now()
	pl.templates[template.ID] = template
}

// Get retrieves a template by ID.
func (pl *PromptLibrary) Get(id string) (*PromptTemplate, bool) {
	t, ok := pl.templates[id]
	return t, ok
}

// List returns all templates, optionally filtered by category.
func (pl *PromptLibrary) List(category *PromptCategory) []*PromptTemplate {
	result := make([]*PromptTemplate, 0)
	for _, t := range pl.templates {
		if category == nil || t.Category == *category {
			result = append(result, t)
		}
	}
	return result
}

// loadBuiltinTemplates loads pre-defined templates.
func (pl *PromptLibrary) loadBuiltinTemplates() {
	// Code review template
	pl.Add(&PromptTemplate{
		ID:          "code-review",
		Name:        "Code Review",
		Description: "Comprehensive code review with suggestions",
		Version:     "1.0",
		Category:    CategoryCodeReview,
		Template: `Review the following {{.language}} code:

{{.code}}

Provide a comprehensive code review covering:
1. Code quality and readability
2. Potential bugs or errors
3. Performance considerations
4. Security vulnerabilities
5. Best practices and improvements
6. Suggested refactoring

{{if .focus}}Focus especially on: {{.focus}}{{end}}`,
		Variables: []TemplateVar{
			{Name: "code", Type: "string", Required: true, Description: "The code to review"},
			{Name: "language", Type: "string", Required: true, Description: "Programming language"},
			{Name: "focus", Type: "string", Required: false, Description: "Specific area to focus on"},
		},
		Config: &PromptConfig{
			PreferredProvider: llm.ProviderAnthropic,
			PreferredModel:    "claude-sonnet-4-20250514",
			Temperature:       0.2,
			OutputFormat:      FormatMarkdown,
		},
	})

	// Summarization template
	pl.Add(&PromptTemplate{
		ID:          "summarize",
		Name:        "Summarize Text",
		Description: "Summarize text with configurable length and style",
		Version:     "1.0",
		Category:    CategorySummarization,
		Template: `Summarize the following text in {{.length}} style:

{{.text}}

{{if .focus}}Focus on: {{.focus}}{{end}}
{{if .audience}}Target audience: {{.audience}}{{end}}`,
		Variables: []TemplateVar{
			{Name: "text", Type: "string", Required: true, Description: "Text to summarize"},
			{Name: "length", Type: "string", Required: false, Default: "medium", Description: "Summary length", Enum: []string{"brief", "medium", "detailed"}},
			{Name: "focus", Type: "string", Required: false, Description: "Aspects to focus on"},
			{Name: "audience", Type: "string", Required: false, Description: "Target audience"},
		},
		Config: &PromptConfig{
			Temperature:  0.3,
			OutputFormat: FormatText,
		},
	})

	// Data extraction template
	pl.Add(&PromptTemplate{
		ID:          "extract-data",
		Name:        "Extract Structured Data",
		Description: "Extract structured data from unstructured text",
		Version:     "1.0",
		Category:    CategoryExtraction,
		Template: `Extract the following information from the text below:

Fields to extract:
{{range .fields}}- {{.}}
{{end}}

Text:
{{.text}}

Return the extracted data as a JSON object with the specified fields. Use null for missing values.`,
		Variables: []TemplateVar{
			{Name: "text", Type: "string", Required: true, Description: "Text to extract from"},
			{Name: "fields", Type: "[]string", Required: true, Description: "Fields to extract"},
		},
		Config: &PromptConfig{
			Temperature:  0.1,
			OutputFormat: FormatJSON,
		},
	})

	// Code generation template
	pl.Add(&PromptTemplate{
		ID:          "generate-code",
		Name:        "Generate Code",
		Description: "Generate code based on requirements",
		Version:     "1.0",
		Category:    CategoryCodeGen,
		Template: `Generate {{.language}} code that accomplishes the following:

Requirements:
{{.requirements}}

{{if .constraints}}
Constraints:
{{range .constraints}}- {{.}}
{{end}}
{{end}}

{{if .examples}}
Example usage:
{{.examples}}
{{end}}

Provide clean, well-documented, production-ready code.`,
		Variables: []TemplateVar{
			{Name: "language", Type: "string", Required: true, Description: "Programming language"},
			{Name: "requirements", Type: "string", Required: true, Description: "What the code should do"},
			{Name: "constraints", Type: "[]string", Required: false, Description: "Any constraints or requirements"},
			{Name: "examples", Type: "string", Required: false, Description: "Example usage"},
		},
		Config: &PromptConfig{
			PreferredProvider: llm.ProviderAnthropic,
			Temperature:       0.2,
			OutputFormat:      FormatCode,
		},
	})

	// Image analysis template
	pl.Add(&PromptTemplate{
		ID:          "analyze-image",
		Name:        "Analyze Image",
		Description: "Analyze and describe an image in detail",
		Version:     "1.0",
		Category:    CategoryMultimodal,
		Template: `Analyze this image and provide:

1. Detailed description of what's shown
2. Key objects, people, or elements identified
3. Colors, composition, and visual style
4. Context and potential meaning
5. Any text visible in the image

{{if .focus}}Focus particularly on: {{.focus}}{{end}}
{{if .purpose}}Analysis purpose: {{.purpose}}{{end}}`,
		Variables: []TemplateVar{
			{Name: "focus", Type: "string", Required: false, Description: "Specific aspect to focus on"},
			{Name: "purpose", Type: "string", Required: false, Description: "Purpose of the analysis"},
		},
		Config: &PromptConfig{
			PreferredProvider: llm.ProviderOpenAI,
			PreferredModel:    "gpt-4o",
			Temperature:       0.3,
			OutputFormat:      FormatMarkdown,
		},
	})

	// Reasoning template
	pl.Add(&PromptTemplate{
		ID:          "reasoning",
		Name:        "Complex Reasoning",
		Description: "Solve complex problems with step-by-step reasoning",
		Version:     "1.0",
		Category:    CategoryReasoning,
		Template: `Solve the following problem using careful, step-by-step reasoning:

Problem:
{{.problem}}

{{if .context}}
Context:
{{.context}}
{{end}}

Think through this problem methodically:
1. First, identify the key components of the problem
2. Consider different approaches and their trade-offs
3. Work through your chosen approach step by step
4. Verify your solution
5. Provide a clear final answer

Show all your work and reasoning.`,
		Variables: []TemplateVar{
			{Name: "problem", Type: "string", Required: true, Description: "The problem to solve"},
			{Name: "context", Type: "string", Required: false, Description: "Additional context"},
		},
		Config: &PromptConfig{
			PreferredProvider: llm.ProviderAnthropic,
			PreferredModel:    "claude-opus-4-20250514",
			Temperature:       0.1,
			OutputFormat:      FormatMarkdown,
		},
	})

	// Translation template
	pl.Add(&PromptTemplate{
		ID:          "translate",
		Name:        "Translate Text",
		Description: "Translate text between languages with style options",
		Version:     "1.0",
		Category:    CategoryTranslation,
		Template: `Translate the following text from {{.source_language}} to {{.target_language}}:

{{.text}}

{{if .style}}Translation style: {{.style}}{{end}}
{{if .preserve_formatting}}Preserve the original formatting.{{end}}
{{if .context}}Context: {{.context}}{{end}}

Provide an accurate, natural-sounding translation.`,
		Variables: []TemplateVar{
			{Name: "text", Type: "string", Required: true, Description: "Text to translate"},
			{Name: "source_language", Type: "string", Required: true, Description: "Source language"},
			{Name: "target_language", Type: "string", Required: true, Description: "Target language"},
			{Name: "style", Type: "string", Required: false, Description: "Translation style", Enum: []string{"formal", "informal", "technical", "literary"}},
			{Name: "preserve_formatting", Type: "bool", Required: false, Description: "Whether to preserve formatting"},
			{Name: "context", Type: "string", Required: false, Description: "Context for better translation"},
		},
		Config: &PromptConfig{
			Temperature:  0.2,
			OutputFormat: FormatText,
		},
	})
}

// PromptOptimizer provides prompt optimization using AI.
type PromptOptimizer struct {
	provider llm.MultimodalProvider
}

// NewPromptOptimizer creates a new prompt optimizer.
func NewPromptOptimizer(provider llm.MultimodalProvider) *PromptOptimizer {
	return &PromptOptimizer{provider: provider}
}

// OptimizationResult contains the optimized prompt and analysis.
type OptimizationResult struct {
	OriginalPrompt  string   `json:"original_prompt"`
	OptimizedPrompt string   `json:"optimized_prompt"`
	Improvements    []string `json:"improvements"`
	Clarity         float64  `json:"clarity"`
	Specificity     float64  `json:"specificity"`
	TokenReduction  int      `json:"token_reduction"`
}

// Optimize improves a prompt for better LLM performance.
func (po *PromptOptimizer) Optimize(ctx context.Context, prompt string) (*OptimizationResult, error) {
	optimizerPrompt := fmt.Sprintf(`Analyze and optimize the following prompt for use with large language models:

Original prompt:
%s

Provide:
1. An optimized version of the prompt that is clearer, more specific, and more effective
2. A list of improvements made
3. Rate the clarity (0-1) and specificity (0-1) of both versions

Format your response as:
OPTIMIZED_PROMPT:
[Your optimized prompt here]

IMPROVEMENTS:
- [Improvement 1]
- [Improvement 2]
...

CLARITY: [original_score] -> [optimized_score]
SPECIFICITY: [original_score] -> [optimized_score]`, prompt)

	messages := []*content.Message{
		content.NewSystemMessage("You are an expert prompt engineer. Your task is to analyze and improve prompts for better AI model performance."),
		content.NewUserMessage(optimizerPrompt),
	}

	resp, err := po.provider.Generate(ctx, messages, &llm.GenerationConfig{
		Temperature: 0.2,
		MaxTokens:   2000,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to optimize prompt: %w", err)
	}

	// Parse the response (simplified parsing)
	responseText := resp.Message.GetText()
	result := &OptimizationResult{
		OriginalPrompt: prompt,
	}

	// Extract optimized prompt
	if idx := strings.Index(responseText, "OPTIMIZED_PROMPT:"); idx != -1 {
		endIdx := strings.Index(responseText[idx:], "IMPROVEMENTS:")
		if endIdx != -1 {
			result.OptimizedPrompt = strings.TrimSpace(responseText[idx+17 : idx+endIdx])
		}
	}

	// Extract improvements
	if idx := strings.Index(responseText, "IMPROVEMENTS:"); idx != -1 {
		endIdx := strings.Index(responseText[idx:], "CLARITY:")
		if endIdx != -1 {
			improvementsText := responseText[idx+13 : idx+endIdx]
			for _, line := range strings.Split(improvementsText, "\n") {
				line = strings.TrimSpace(line)
				if strings.HasPrefix(line, "-") {
					result.Improvements = append(result.Improvements, strings.TrimPrefix(line, "- "))
				}
			}
		}
	}

	return result, nil
}

// PromptChain allows chaining multiple prompts together.
type PromptChain struct {
	steps    []ChainStep
	provider llm.MultimodalProvider
}

// ChainStep represents a step in the prompt chain.
type ChainStep struct {
	Name         string
	Template     *PromptTemplate
	Variables    map[string]any
	Transform    func(response string) map[string]any
	Config       *llm.GenerationConfig
}

// NewPromptChain creates a new prompt chain.
func NewPromptChain(provider llm.MultimodalProvider) *PromptChain {
	return &PromptChain{
		steps:    make([]ChainStep, 0),
		provider: provider,
	}
}

// AddStep adds a step to the chain.
func (pc *PromptChain) AddStep(step ChainStep) *PromptChain {
	pc.steps = append(pc.steps, step)
	return pc
}

// ChainResult contains the results from a chain execution.
type ChainResult struct {
	Steps       []StepResult  `json:"steps"`
	FinalOutput string        `json:"final_output"`
	TotalTokens int           `json:"total_tokens"`
	Duration    time.Duration `json:"duration"`
}

// StepResult contains the result of a single step.
type StepResult struct {
	Name     string        `json:"name"`
	Input    string        `json:"input"`
	Output   string        `json:"output"`
	Duration time.Duration `json:"duration"`
	Tokens   int           `json:"tokens"`
}

// Execute runs the prompt chain.
func (pc *PromptChain) Execute(ctx context.Context, initialVars map[string]any) (*ChainResult, error) {
	startTime := time.Now()
	result := &ChainResult{
		Steps: make([]StepResult, 0),
	}

	currentVars := initialVars

	for _, step := range pc.steps {
		stepStart := time.Now()

		// Merge step variables with current vars
		mergedVars := make(map[string]any)
		for k, v := range currentVars {
			mergedVars[k] = v
		}
		for k, v := range step.Variables {
			mergedVars[k] = v
		}

		// Render the prompt
		prompt, err := step.Template.Render(mergedVars)
		if err != nil {
			return nil, fmt.Errorf("step %s: failed to render template: %w", step.Name, err)
		}

		// Execute
		messages := []*content.Message{
			content.NewUserMessage(prompt),
		}

		config := step.Config
		if config == nil {
			config = &llm.GenerationConfig{}
		}

		resp, err := pc.provider.Generate(ctx, messages, config)
		if err != nil {
			return nil, fmt.Errorf("step %s: generation failed: %w", step.Name, err)
		}

		output := resp.Message.GetText()

		stepResult := StepResult{
			Name:     step.Name,
			Input:    prompt,
			Output:   output,
			Duration: time.Since(stepStart),
		}
		if resp.Usage != nil {
			stepResult.Tokens = resp.Usage.TotalTokens
			result.TotalTokens += resp.Usage.TotalTokens
		}
		result.Steps = append(result.Steps, stepResult)

		// Transform output for next step
		if step.Transform != nil {
			currentVars = step.Transform(output)
		} else {
			currentVars["previous_output"] = output
		}
	}

	result.Duration = time.Since(startTime)
	if len(result.Steps) > 0 {
		result.FinalOutput = result.Steps[len(result.Steps)-1].Output
	}

	return result, nil
}
