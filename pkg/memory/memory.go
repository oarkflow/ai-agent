package memory

import "github.com/sujit/ai-agent/pkg/llm"

// Memory defines the interface for managing agent conversation history.
type Memory interface {
	// AddMessage adds a message to the memory.
	AddMessage(role, content string)

	// Add creates a message struct and adds it.
	Add(msg llm.Message)

	// GetHistory returns the current context window for the LLM.
	GetHistory() []llm.Message

	// Clear resets the memory.
	Clear()
}
