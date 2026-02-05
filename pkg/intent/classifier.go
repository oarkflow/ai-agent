package intent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/oarkflow/ai-agent/pkg/content"
	"github.com/oarkflow/ai-agent/pkg/llm"
)

// Classifier handles intent detection logic.
type Classifier struct {
	Router *llm.SmartRouter
	Config ClassifierConfig
}

type ClassifierConfig struct {
	Model       string
	Temperature float64
}

func NewClassifier(router *llm.SmartRouter, cfg ClassifierConfig) *Classifier {
	if cfg.Temperature == 0 {
		cfg.Temperature = 0.0 // Deterministic
	}
	return &Classifier{
		Router: router,
		Config: cfg,
	}
}

// Classify determines the intent of the user Input.
func (c *Classifier) Classify(ctx context.Context, input string) (*IntentResult, error) {
	// Define the classification prompt
	systemPrompt := `You are an Intent Classifier. Analyze the user's request and classify it into one of the following Intent Types:

- chat: General conversation, greeting, or questions not covered below.
- image_generation: Requests to create, generate, or draw images/pictures/art.
- code_generation: Requests to write, fix, debug, or explain code/software.
- data_analysis: Requests to analyze data, find patterns, or inspect files/content.
- search: Requests to search for information, news, or facts outside immediate context.
- task_execution: Requests to perform a specific action, workflow, or tool execution.
- summarization: Requests to shorten or summarize text.
- translation: Requests to translate text between languages.

Also, identify the 'domain' (e.g., healthcare, finance, technical, general) and extract key 'entities' (specific subjects, parameters).

Output strictly valid JSON in the following format:
{
  "intent": "<IntentType>",
  "confidence": <float 0.0-1.0>,
  "domain": "<Domain>",
  "entities": {
    "key1": "value1"
  },
  "summary": "<Brief summary of request>"
}
`

	userPrompt := fmt.Sprintf("Request: %s", input)

	// Combine messages
	messages := []*content.Message{
		content.NewSystemMessage(systemPrompt),
		content.NewUserMessage(userPrompt),
	}

	// Route configuration for fast/analysis model
	req := &llm.ModelRequirements{
		TaskType:    llm.TaskAnalysis,
		Speed:       llm.SpeedFast,
		CheckHealth: true,
	}
	genConfig := &llm.GenerationConfig{
		Temperature: c.Config.Temperature,
		Model:       c.Config.Model,
		MaxTokens:   500, // Short response
	}

	// Execute
	resp, err := c.Router.Route(ctx, messages, genConfig, req)
	if err != nil {
		return nil, fmt.Errorf("intent classification failed: %w", err)
	}

	// Parse Response
	cleanResp := strings.TrimSpace(resp.Message.GetText())
	// Strip Markdown limits if present
	cleanResp = strings.TrimPrefix(cleanResp, "```json")
	cleanResp = strings.TrimPrefix(cleanResp, "```")
	cleanResp = strings.TrimSuffix(cleanResp, "```")

	var result IntentResult
	if err := json.Unmarshal([]byte(cleanResp), &result); err != nil {
		return nil, fmt.Errorf("failed to parse intent JSON: %w", err)
	}

	// Fallback for unknown intent
	if result.Intent == "" {
		result.Intent = IntentUnknown
	}

	return &result, nil
}
