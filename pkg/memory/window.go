package memory

import "github.com/oarkflow/ai-agent/pkg/llm"

// SimpleMemory holds all messages indefinitely.
type SimpleMemory struct {
	messages []llm.Message
}

func NewSimpleMemory() *SimpleMemory {
	return &SimpleMemory{
		messages: make([]llm.Message, 0),
	}
}

func (m *SimpleMemory) AddMessage(role, content string) {
	m.messages = append(m.messages, llm.Message{Role: role, Content: content})
}

func (m *SimpleMemory) Add(msg llm.Message) {
	m.messages = append(m.messages, msg)
}

func (m *SimpleMemory) GetHistory() []llm.Message {
	return m.messages
}

func (m *SimpleMemory) Clear() {
	m.messages = make([]llm.Message, 0)
}

// WindowBufferMemory keeps only the last K messages.
type WindowBufferMemory struct {
	windowSize int
	messages   []llm.Message
}

func NewWindowBufferMemory(size int) *WindowBufferMemory {
	if size <= 0 {
		size = 10
	}
	return &WindowBufferMemory{
		windowSize: size,
		messages:   make([]llm.Message, 0),
	}
}

func (m *WindowBufferMemory) Add(msg llm.Message) {
	m.messages = append(m.messages, msg)

	if len(m.messages) > m.windowSize {
		// We need to evict execution to maintain the window size.
		// Comprehensive strategy:
		// 1. Preserve System Prompt (if present at index 0).
		// 2. Ensure we don't hold references to old backing arrays (Copy).

		hasSystem := len(m.messages) > 0 && m.messages[0].Role == llm.RoleSystem

		// Create a new slice to hold the kept messages.
		// This ensures expected garbage collection of evicted messages.
		keptMessages := make([]llm.Message, 0, m.windowSize)

		if hasSystem {
			// Keep System Prompt
			keptMessages = append(keptMessages, m.messages[0])

			// Calculate how many more we can fit: windowSize - 1
			// We need the *last* (windowSize - 1) interactions from the remaining list.
			// Current remaining count = len(m.messages) - 1
			// We want to drop (CurrentRemaining - MaxRemaining)

			remainingSpace := m.windowSize - 1
			if remainingSpace > 0 {
				// Safety check if windowSize is extremely small
				startIndex := len(m.messages) - remainingSpace
				if startIndex < 1 {
					startIndex = 1
				}
				keptMessages = append(keptMessages, m.messages[startIndex:]...)
			}
		} else {
			// Standard sliding window: keep last N
			startIndex := len(m.messages) - m.windowSize
			keptMessages = append(keptMessages, m.messages[startIndex:]...)
		}

		m.messages = keptMessages
	}
}

func (m *WindowBufferMemory) GetHistory() []llm.Message {
	return m.messages
}

func (m *WindowBufferMemory) Clear() {
	m.messages = make([]llm.Message, 0)
}
