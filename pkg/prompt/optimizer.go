package prompt

import (
	"context"
	"fmt"

	"github.com/oarkflow/ai-agent/pkg/llm"
)

// Optimizer helps refine prompts using an LLM.
type Optimizer struct {
	Provider llm.Provider
}

func NewOptimizer(p llm.Provider) *Optimizer {
	return &Optimizer{Provider: p}
}

// Optimize takes a crude prompt and refines it into a SMART prompt structure.
func (o *Optimizer) Optimize(ctx context.Context, inputPrompt string) (string, error) {
	metaPrompt := `You are an expert Prompt Engineer.
Refine the following user prompt into a generic structured "SMART" prompt (Specific, Measurable, Actionable, Relevant, Time-bound/Token-aware).
The output should be the refined prompt text ONLY, ready to be used by an AI.

Structure:
# ROLE
[Role description]

# TASK
[Detailed task description]

# CONSTRAINTS
- [Constraint 1]
- [Constraint 2]

# OUTPUT FORMAT
[Expected format]

---
USER PROMPT TO REFINE:
%s
`
	return o.Provider.Generate(ctx, fmt.Sprintf(metaPrompt, inputPrompt), nil)
}
