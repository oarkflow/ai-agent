package orchestrator

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/oarkflow/ai-agent/pkg/agent"
	"github.com/oarkflow/ai-agent/pkg/cot"
	"github.com/oarkflow/ai-agent/pkg/llm"
	"github.com/oarkflow/ai-agent/pkg/prompt"
	"github.com/oarkflow/ai-agent/pkg/tot"
)

// AnalysisResult represents the decision made by the orchestrator.
type AnalysisResult struct {
	Complexity string   `json:"complexity"` // "high", "low", "coding", "reasoning"
	Strategy   string   `json:"strategy"`   // "direct", "cot", "tot"
	Reasoning  string   `json:"reasoning"`
	Domain     string   `json:"domain"`      // e.g. "healthcare", "coding", "general"
	MemoryKeys []string `json:"memory_keys"` // Key entities for context
}

// Orchestrator analyzes requests and executes them using the best strategy.
type Orchestrator struct {
	Router *llm.Router
}

func NewOrchestrator(router *llm.Router) *Orchestrator {
	return &Orchestrator{Router: router}
}

// Analyze uses a fast model to determine the best strategy.
func (o *Orchestrator) Analyze(ctx context.Context, input string) (*AnalysisResult, error) {
	// Meta-prompt for analysis
	metaPrompt := `Analyze the following user request.
Output strictly in JSON format.

1. Strategy:
- "direct": Simple questions, lookups (low complexity).
- "cot": Logic puzzles, math, multi-step reasoning (medium/high complexity).
- "tot": Complex planning, finding best paths, scenario analysis (high complexity).

2. Complexity (Intent):
- "low": Simple q&a.
- "high": General complex tasks.
- "coding": specifically for writing, debugging, or explaining code.
- "reasoning": specifically for math, hard logic, or deep research.

3. Domain:
- Identify the subject matter (e.g., "healthcare", "workflow", "coding", "finance", "general").

4. Memory Keys:
- List key entities or concepts to track (e.g., specific names, project IDs, specialized terms).

User Request: %s

JSON Output format:
{"complexity": "...", "strategy": "...", "reasoning": "...", "domain": "...", "memory_keys": ["..."]}
`
	// Use the Fast provider for analysis to save cost/time
	fastProv := o.Router.GetProvider(llm.TierFast)

	resp, err := fastProv.Generate(ctx, fmt.Sprintf(metaPrompt, input), &llm.GenerateOptions{
		Temperature: 0.0,
		Model:       "gpt-4o-mini", // Explicitly prefer mini if possible, though strictness depends on provider impl
	})
	if err != nil {
		return nil, err
	}

	// Clean up response (sometimes LLMs add markdown code blocks)
	cleanResp := strings.TrimSpace(resp)
	cleanResp = strings.TrimPrefix(cleanResp, "```json")
	cleanResp = strings.TrimPrefix(cleanResp, "```")
	cleanResp = strings.TrimSuffix(cleanResp, "```")

	var result AnalysisResult
	if err := json.Unmarshal([]byte(cleanResp), &result); err != nil {
		// Fallback if JSON fails
		fmt.Printf("Orchestrator analysis failed to parse JSON: %v. Defaulting to Smart/CoT.\n", err)
		return &AnalysisResult{Complexity: "high", Strategy: "cot"}, nil
	}

	return &result, nil
}

// SmartExecute orchestrates the entire process.
func (o *Orchestrator) SmartExecute(ctx context.Context, input string) (string, error) {
	// 1. Analyze
	fmt.Println("Orchestrator: Analyzing request...")
	plan, err := o.Analyze(ctx, input)
	if err != nil {
		return "", fmt.Errorf("analysis failed: %w", err)
	}
	fmt.Printf("Orchestrator Plan: %s (Complexity: %s)\nReasoning: %s\n", plan.Strategy, plan.Complexity, plan.Reasoning)

	// 2. Select Provider
	provider := o.Router.RouteByIntent(plan.Complexity)

	fmt.Printf("Orchestrator: Selected Provider Tier for intensity '%s'\n", plan.Complexity)

	// 3. Select Strategy & Execute
	switch plan.Strategy {
	case "tot":
		// Tree of Thought
		gen := &tot.LLMGenerator{Provider: provider, Model: "gpt-4o"} // Assuming smart provider has this model
		eval := &tot.LLMEvaluator{Provider: provider, Model: "gpt-4o"}
		solver := tot.NewTreeOfThoughts(gen, eval, 3, 3)

		fmt.Println("Orchestrator: Running Tree of Thought...")
		return solver.BFS(ctx, []llm.Message{}, input) // Empty history for now, could be passed in

	case "cot":
		// Chain of Thought
		ag := agent.NewAgent("CoTAgent", provider, "You are a helpful and logical assistant.")
		cotAgent := cot.NewCoTAgent(ag)

		fmt.Println("Orchestrator: Running Chain of Thought...")
		return cotAgent.Run(ctx, input)

	default: // "direct"
		// Optimization: Use SMART prompt optimization for direct queries too?
		// Let's do a quick optimization if complexity is "high" but strategy is "direct" (rare),
		// or just use standard agent.

		// Let's use the Prompt Optimizer to ensure "most satisfied answer" as requested
		optimizer := prompt.NewOptimizer(provider)
		optimizedPrompt, err := optimizer.Optimize(ctx, input)
		if err == nil {
			fmt.Printf("Orchestrator: Optimized Prompt:\n%s\n", optimizedPrompt)
			input = optimizedPrompt
		}

		ag := agent.NewAgent("DirectAgent", provider, "You are a helpful assistant.")
		fmt.Println("Orchestrator: Running Direct Agent...")
		return ag.Chat(ctx, input)
	}
}
