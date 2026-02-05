package cot

import (
	"context"
	"fmt"
	"strings"

	"github.com/oarkflow/ai-agent/pkg/agent"
)

// CoTAgent wraps a standard agent to enforce Chain of Thought reasoning.
type CoTAgent struct {
	Agent *agent.Agent
}

// NewCoTAgent creates a new Chain of Thought agent.
func NewCoTAgent(ag *agent.Agent) *CoTAgent {
	// Append CoT instruction to system prompt if not present
	if !strings.Contains(ag.SystemPrompt, "step by step") {
		ag.SystemPrompt += "\n\nYou are a helpful assistant. When answering, you must think step by step. First, explain your reasoning process, then provide the final answer."
	}
	return &CoTAgent{Agent: ag}
}

// Run executes the agent with CoT prompting.
func (c *CoTAgent) Run(ctx context.Context, input string) (string, error) {
	// We can augment the user prompt here as well to ensure CoT
	cotInput := fmt.Sprintf("%s\n\nLet's think step by step.", input)

	response, err := c.Agent.Run(ctx, cotInput)
	if err != nil {
		return "", err
	}

	return response, nil
}
