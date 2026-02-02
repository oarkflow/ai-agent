package prompt

import (
	"fmt"
	"strings"
)

// SMARTPromptBuilder helps construct structured prompts.
type SMARTPromptBuilder struct {
	Role         string
	Context      string
	Task         string
	Constraints  []string
	Examples     []string
	OutputFormat string
}

// NewSMARTPrompt creates a builder.
func NewSMARTPrompt() *SMARTPromptBuilder {
	return &SMARTPromptBuilder{}
}

func (b *SMARTPromptBuilder) WithRole(role string) *SMARTPromptBuilder {
	b.Role = role
	return b
}

func (b *SMARTPromptBuilder) WithContext(ctx string) *SMARTPromptBuilder {
	b.Context = ctx
	return b
}

func (b *SMARTPromptBuilder) WithTask(task string) *SMARTPromptBuilder {
	b.Task = task
	return b
}

func (b *SMARTPromptBuilder) AddConstraint(c string) *SMARTPromptBuilder {
	b.Constraints = append(b.Constraints, c)
	return b
}

func (b *SMARTPromptBuilder) AddExample(input, output string) *SMARTPromptBuilder {
	b.Examples = append(b.Examples, fmt.Sprintf("Input: %s\nOutput: %s", input, output))
	return b
}

func (b *SMARTPromptBuilder) WithOutputFormat(format string) *SMARTPromptBuilder {
	b.OutputFormat = format
	return b
}

// String compiles the SMART prompt.
func (b *SMARTPromptBuilder) String() string {
	var sb strings.Builder

	if b.Role != "" {
		sb.WriteString(fmt.Sprintf("# ROLE\n%s\n\n", b.Role))
	}

	if b.Context != "" {
		sb.WriteString(fmt.Sprintf("# CONTEXT\n%s\n\n", b.Context))
	}

	if b.Task != "" {
		sb.WriteString(fmt.Sprintf("# TASK\n%s\n\n", b.Task))
	}

	if len(b.Constraints) > 0 {
		sb.WriteString("# CONSTRAINTS\n")
		for _, c := range b.Constraints {
			sb.WriteString(fmt.Sprintf("- %s\n", c))
		}
		sb.WriteString("\n")
	}

	if len(b.Examples) > 0 {
		sb.WriteString("# EXAMPLES\n")
		for _, ex := range b.Examples {
			sb.WriteString(ex + "\n\n")
		}
	}

	if b.OutputFormat != "" {
		sb.WriteString(fmt.Sprintf("# OUTPUT FORMAT\n%s\n", b.OutputFormat))
	}

	return sb.String()
}
