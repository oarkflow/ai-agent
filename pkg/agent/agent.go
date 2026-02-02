package agent

import (
	"context"
	"fmt"

	"github.com/sujit/ai-agent/pkg/llm"
)

// Agent represents an AI agent with memory and a specific role.
type Agent struct {
	Name         string
	Provider     llm.Provider
	SystemPrompt string
	Memory       []llm.Message
	Options      *llm.GenerateOptions
}

// NewAgent creates a new agent.
func NewAgent(name string, provider llm.Provider, systemPrompt string) *Agent {
	return &Agent{
		Name:         name,
		Provider:     provider,
		SystemPrompt: systemPrompt,
		Memory:       make([]llm.Message, 0),
	}
}

// ClearMemory resets the agent's memory.
func (a *Agent) ClearMemory() {
	a.Memory = make([]llm.Message, 0)
}

// Chat sends a user message to the agent and gets a response.
// It maintains the conversation history (memory).
func (a *Agent) Chat(ctx context.Context, input string) (string, error) {
	// 1. Construct messages: System + Memory + User Input
	messages := []llm.Message{}
	if a.SystemPrompt != "" {
		messages = append(messages, llm.Message{Role: llm.RoleSystem, Content: a.SystemPrompt})
	}
	messages = append(messages, a.Memory...)
	messages = append(messages, llm.Message{Role: llm.RoleUser, Content: input})

	// 2. Call LLM
	response, err := a.Provider.Chat(ctx, messages, a.Options)
	if err != nil {
		return "", fmt.Errorf("agent %s failed to generate response: %w", a.Name, err)
	}

	// 3. Update Memory
	a.Memory = append(a.Memory, llm.Message{Role: llm.RoleUser, Content: input})
	a.Memory = append(a.Memory, llm.Message{Role: llm.RoleAssistant, Content: response})

	return response, nil
}

// Run is a simple alias for Chat that fits the "Run" paradigm for chains.
func (a *Agent) Run(ctx context.Context, input string) (string, error) {
	return a.Chat(ctx, input)
}
