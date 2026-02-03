package memory

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/sujit/ai-agent/pkg/content"
	"github.com/sujit/ai-agent/pkg/llm"
)

// LLMScorer uses an LLM provider to score message importance.
type LLMScorer struct {
	provider llm.MultimodalProvider
	model    string
	// optional instruction tweak
	instruction string
}

func NewLLMScorer(provider llm.MultimodalProvider, model string) *LLMScorer {
	return &LLMScorer{provider: provider, model: model, instruction: "Rate the importance of the following message for future conversation context on a scale from 0 (not important) to 1 (very important). Return just a single decimal number"}
}

func (s *LLMScorer) Score(ctx context.Context, msg *content.Message) float64 {
	if s.provider == nil {
		return 0.0
	}

	// Build prompt
	prompt := fmt.Sprintf("%s:\n\nMessage:\n%s", s.instruction, msg.GetText())

	resp, err := s.provider.Generate(ctx, []*content.Message{
		content.NewSystemMessage("You are an importance classifier. Provide a single number between 0 and 1."),
		content.NewUserMessage(prompt),
	}, &llm.GenerationConfig{Temperature: 0.0, MaxTokens: 8, Model: s.model})
	if err != nil || resp == nil || resp.Message == nil {
		return 0.0
	}

	text := strings.TrimSpace(resp.Message.GetText())
	// Try to extract a number from the response
	// Accept lines or plain text
	if idx := strings.Index(text, "\n"); idx != -1 {
		text = text[:idx]
	}
	// Remove any non-numeric punctuation
	text = strings.Trim(text, "., ")
	val, err := strconv.ParseFloat(text, 64)
	if err != nil {
		// try to parse words like 'low','medium','high'
		l := strings.ToLower(text)
		switch l {
		case "low", "small", "0":
			return 0.1
		case "medium", "moderate":
			return 0.5
		case "high", "important", "1":
			return 0.9
		default:
			return 0.0
		}
	}
	if val < 0 {
		val = 0
	}
	if val > 1 {
		val = 1
	}
	return val
}
