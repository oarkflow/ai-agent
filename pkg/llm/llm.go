package llm

import "context"

// Message represents a single message in a chat conversation.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
)

// GenerateOptions contains optional parameters for LLM generation.
type GenerateOptions struct {
	Temperature float64
	MaxTokens   int
	StopSequences []string
	Model       string
}

// Provider defines the interface for interacting with LLMs.
type Provider interface {
	// Generate sends a prompt to the LLM and returns the generated text.
	Generate(ctx context.Context, prompt string, options *GenerateOptions) (string, error)

	// Chat sends a list of messages to the LLM and returns the generated response.
	Chat(ctx context.Context, messages []Message, options *GenerateOptions) (string, error)
}
