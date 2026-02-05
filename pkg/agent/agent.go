package agent

import (
	"context"
	"fmt"

	"github.com/oarkflow/ai-agent/pkg/llm"
	"github.com/oarkflow/ai-agent/pkg/memory"
)

// Agent represents an AI agent with memory and a specific role.
type Agent struct {
	Name         string
	Provider     llm.Provider
	SystemPrompt string
	Memory       memory.Memory
	Options      *llm.GenerateOptions
}

// NewAgent creates a new agent with a default simple memory.
func NewAgent(name string, provider llm.Provider, systemPrompt string) *Agent {
	return &Agent{
		Name:         name,
		Provider:     provider,
		SystemPrompt: systemPrompt,
		Memory:       memory.NewSimpleMemory(),
	}
}

// WithMemory allows setting a custom memory implementation (chainable).
func (a *Agent) WithMemory(mem memory.Memory) *Agent {
	a.Memory = mem
	return a
}

// ClearMemory resets the agent's memory.
func (a *Agent) ClearMemory() {
	a.Memory.Clear()
}

// Chat sends a user message to the agent and gets a response.
// It maintains the conversation history (memory).
func (a *Agent) Chat(ctx context.Context, input string) (string, error) {
	// 1. Construct messages: System + Memory + User Input
	messages := []llm.Message{}
	if a.SystemPrompt != "" {
		messages = append(messages, llm.Message{Role: llm.RoleSystem, Content: a.SystemPrompt})
	}

	// Get history from memory
	messages = append(messages, a.Memory.GetHistory()...)

	// Add current user input to messages sent to LLM (but not yet to memory, or maybe yes?)
	// Typically we add it to memory immediately or after success.
	// Let's add to request first.
	messages = append(messages, llm.Message{Role: llm.RoleUser, Content: input})

	// 2. Call LLM
	response, err := a.Provider.Chat(ctx, messages, a.Options)
	if err != nil {
		return "", fmt.Errorf("agent %s failed to generate response: %w", a.Name, err)
	}

	// 3. Update Memory
	a.Memory.AddMessage(llm.RoleUser, input)
	a.Memory.AddMessage(llm.RoleAssistant, response)

	return response, nil
}

// Run is a simple alias for Chat that fits the "Run" paradigm for chains.
func (a *Agent) Run(ctx context.Context, input string) (string, error) {
	return a.Chat(ctx, input)
}
