package memory

import (
	"context"
	"fmt"

	"github.com/sujit/ai-agent/pkg/llm"
)

// SummaryMemory maintains a summary of the past conversation + a recent window of exact messages.
type SummaryMemory struct {
	provider      llm.Provider
	summary       string
	buffer        []llm.Message
	maxWindowSize int
}

// NewSummaryMemory creates a memory that summarizes older messages when the buffer gets too full.
func NewSummaryMemory(provider llm.Provider, maxWindowSize int) *SummaryMemory {
	if maxWindowSize <= 0 {
		maxWindowSize = 5
	}
	return &SummaryMemory{
		provider:      provider,
		maxWindowSize: maxWindowSize,
		buffer:        make([]llm.Message, 0),
	}
}

func (m *SummaryMemory) AddMessage(role, content string) {
	m.Add(llm.Message{Role: role, Content: content})
}

func (m *SummaryMemory) Add(msg llm.Message) {
	m.buffer = append(m.buffer, msg)

	// Triggers summarization if buffer exceeds limit significantly (e.g., 2x window)
	// For simplicity, we trigger it strictly when hitting the limit + 2 (one turn pair)
	if len(m.buffer) > m.maxWindowSize+2 {
		m.pruneAndSummarize()
	}
}

func (m *SummaryMemory) pruneAndSummarize() {
	// We want to summarize the oldest messages in the buffer
	// messages_to_summarize = buffer[:-maxWindowSize]
	// new_buffer = buffer[-maxWindowSize:]

	cutoff := len(m.buffer) - m.maxWindowSize
	toSummarize := m.buffer[:cutoff]
	m.buffer = m.buffer[cutoff:]

	ctx := context.Background() // In a real app, might pass ctx through Add

	// Create prompt for summarization
	conversation := ""
	for _, msg := range toSummarize {
		conversation += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
	}

	prompt := fmt.Sprintf("Progressively summarize the lines of conversation provided, adding to the previous summary returning a new summary.\n\nEXAMPLE\nCurrent summary: The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.\nNew lines of conversation:\nHuman: Why do you think artificial intelligence is a force for good?\nAI: Because artificial intelligence will help humans reach their full potential.\nNew summary: The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.\nEND OF EXAMPLE\n\nCurrent summary: %s\nNew lines of conversation:\n%s\nNew summary:", m.summary, conversation)

	// Call LLM to update summary
	// Note: failing to summarize is non-critical for the immediate request, but degrades memory.
	// We log or ignore error for now to keep Add signature simple.
	newSummary, err := m.provider.Generate(ctx, prompt, nil)
	if err == nil {
		m.summary = newSummary
	} else {
		fmt.Printf("Error updating summary memory: %v\n", err)
	}
}

func (m *SummaryMemory) GetHistory() []llm.Message {
	// Reconstruct history:
	// 1. System message containing the summary (if it exists)
	// 2. The recent buffer

	history := make([]llm.Message, 0)

	if m.summary != "" {
		history = append(history, llm.Message{
			Role:    llm.RoleSystem,
			Content: fmt.Sprintf("Summary of previous conversation: %s", m.summary),
		})
	}

	history = append(history, m.buffer...)
	return history
}

func (m *SummaryMemory) Clear() {
	m.summary = ""
	m.buffer = make([]llm.Message, 0)
}
