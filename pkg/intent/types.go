package intent

// IntentType represents the category of user intent.
type IntentType string

const (
	IntentUnknown         IntentType = "unknown"
	IntentChat            IntentType = "chat"
	IntentImageGeneration IntentType = "image_generation"
	IntentCodeGeneration  IntentType = "code_generation"
	IntentDataAnalysis    IntentType = "data_analysis"
	IntentSearch          IntentType = "search"
	IntentTaskExecution   IntentType = "task_execution"
	IntentSummarization   IntentType = "summarization"
	IntentTranslation     IntentType = "translation"
)

// IntentResult holds the classification result.
type IntentResult struct {
	Intent     IntentType        `json:"intent"`
	Confidence float64           `json:"confidence"`
	Domain     string            `json:"domain"`
	Entities   map[string]string `json:"entities"`
	Summary    string            `json:"summary"` // Brief summary of the request
}

// IsHighConfidence checks if the result is confident enough.
func (r *IntentResult) IsHighConfidence(threshold float64) bool {
	return r.Confidence >= threshold
}
