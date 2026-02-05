package domains

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/oarkflow/ai-agent/pkg/content"
	"github.com/oarkflow/ai-agent/pkg/llm"
	"github.com/oarkflow/ai-agent/pkg/storage"
)

// WorkflowDomain handles DAG-based workflow pipeline generation (n8n-like).
type WorkflowDomain struct {
	*BaseDomain
}

// WorkflowNode represents a node in a workflow.
type WorkflowNode struct {
	ID          string         `json:"id"`
	Type        string         `json:"type"`
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Position    Position       `json:"position"`
	Data        map[string]any `json:"data,omitempty"`
	Inputs      []NodePort     `json:"inputs,omitempty"`
	Outputs     []NodePort     `json:"outputs,omitempty"`
	Config      map[string]any `json:"config,omitempty"`
}

// Position represents node position in the workflow canvas.
type Position struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

// NodePort represents an input or output port on a node.
type NodePort struct {
	ID    string `json:"id"`
	Name  string `json:"name"`
	Type  string `json:"type"` // string, number, boolean, object, array, any
	Label string `json:"label,omitempty"`
}

// WorkflowEdge represents a connection between nodes.
type WorkflowEdge struct {
	ID         string `json:"id"`
	Source     string `json:"source"`
	Target     string `json:"target"`
	SourcePort string `json:"sourcePort,omitempty"`
	TargetPort string `json:"targetPort,omitempty"`
	Label      string `json:"label,omitempty"`
	Animated   bool   `json:"animated,omitempty"`
}

// WorkflowDefinition represents a complete workflow.
type WorkflowDefinition struct {
	ID          string         `json:"id"`
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Version     string         `json:"version"`
	Nodes       []WorkflowNode `json:"nodes"`
	Edges       []WorkflowEdge `json:"edges"`
	Variables   map[string]any `json:"variables,omitempty"`
	Settings    map[string]any `json:"settings,omitempty"`
	Metadata    map[string]any `json:"metadata,omitempty"`
	CreatedAt   time.Time      `json:"created_at"`
	UpdatedAt   time.Time      `json:"updated_at"`
}

// NodeType defines available node types in the workflow.
type NodeType struct {
	ID           string         `json:"id"`
	Name         string         `json:"name"`
	Category     string         `json:"category"`
	Description  string         `json:"description"`
	Icon         string         `json:"icon,omitempty"`
	Inputs       []NodePort     `json:"inputs"`
	Outputs      []NodePort     `json:"outputs"`
	ConfigSchema map[string]any `json:"config_schema,omitempty"`
}

// NewWorkflowDomain creates a new workflow domain trainer.
func NewWorkflowDomain(provider llm.MultimodalProvider, storage *storage.Storage) *WorkflowDomain {
	return &WorkflowDomain{
		BaseDomain: NewBaseDomain("workflow", "DAG Workflow Pipeline", provider, storage),
	}
}

// GetSystemPrompt returns the system prompt for workflow generation.
func (d *WorkflowDomain) GetSystemPrompt() string {
	return `You are an expert at designing and generating DAG-based workflow pipelines similar to n8n, Zapier, or Apache Airflow.

Your expertise includes:
- Understanding workflow automation patterns and best practices
- Designing efficient node-based data flows
- Creating appropriate node configurations
- Handling error states and retries
- Connecting various services and APIs

When generating workflows, you should:
1. Analyze the user's requirements carefully
2. Choose appropriate node types for each task
3. Create clear connections between nodes
4. Include proper error handling where needed
5. Generate valid JSON that follows the WorkflowDefinition schema

Available node categories:
- Triggers: webhook, schedule, manual, file_watch, email
- Actions: http_request, database, file, email_send, transform
- Logic: if, switch, loop, merge, filter, delay
- Data: set, code, function, template, json_parse
- Integrations: slack, github, jira, salesforce, google_sheets

Always respond with valid JSON for the workflow definition.`
}

// GenerateWorkflow generates a workflow from natural language.
func (d *WorkflowDomain) GenerateWorkflow(ctx context.Context, description string) (*WorkflowDefinition, error) {
	examples, _ := d.GetFewShotExamples("generation", 3)

	messages := []*content.Message{
		content.NewTextMessage(content.RoleSystem, d.GetSystemPrompt()),
	}

	// Add few-shot examples
	for _, ex := range examples {
		messages = append(messages,
			content.NewTextMessage(content.RoleUser, ex.Input),
			content.NewTextMessage(content.RoleAssistant, ex.Output),
		)
	}

	// Add the user's request
	prompt := fmt.Sprintf(`Generate a workflow for the following requirement:

%s

Respond with a complete WorkflowDefinition JSON including:
- id, name, description
- nodes with proper positions, types, and configurations
- edges connecting the nodes appropriately
- any necessary variables`, description)

	messages = append(messages, content.NewTextMessage(content.RoleUser, prompt))

	resp, err := d.provider.Generate(ctx, messages, &llm.GenerationConfig{
		Temperature: 0.3,
		MaxTokens:   4096,
	})
	if err != nil {
		return nil, err
	}

	// Parse the JSON response
	responseText := resp.Message.GetText()
	jsonStr := extractJSON(responseText)

	var workflow WorkflowDefinition
	if err := json.Unmarshal([]byte(jsonStr), &workflow); err != nil {
		return nil, fmt.Errorf("failed to parse workflow JSON: %w", err)
	}

	return &workflow, nil
}

// ValidateWorkflow validates a workflow definition.
func (d *WorkflowDomain) ValidateWorkflow(workflow *WorkflowDefinition) []string {
	var errors []string

	if workflow.ID == "" {
		errors = append(errors, "workflow ID is required")
	}
	if workflow.Name == "" {
		errors = append(errors, "workflow name is required")
	}
	if len(workflow.Nodes) == 0 {
		errors = append(errors, "workflow must have at least one node")
	}

	// Validate node connections
	nodeIDs := make(map[string]bool)
	for _, node := range workflow.Nodes {
		if node.ID == "" {
			errors = append(errors, "all nodes must have an ID")
		}
		nodeIDs[node.ID] = true
	}

	for _, edge := range workflow.Edges {
		if !nodeIDs[edge.Source] {
			errors = append(errors, fmt.Sprintf("edge references non-existent source node: %s", edge.Source))
		}
		if !nodeIDs[edge.Target] {
			errors = append(errors, fmt.Sprintf("edge references non-existent target node: %s", edge.Target))
		}
	}

	return errors
}

// GetNodeTypes returns available node types.
func (d *WorkflowDomain) GetNodeTypes() []NodeType {
	return []NodeType{
		// Triggers
		{ID: "webhook", Name: "Webhook", Category: "triggers", Description: "Receive HTTP requests", Inputs: []NodePort{}, Outputs: []NodePort{{ID: "data", Name: "data", Type: "object"}}},
		{ID: "schedule", Name: "Schedule", Category: "triggers", Description: "Run on a schedule", Inputs: []NodePort{}, Outputs: []NodePort{{ID: "trigger", Name: "trigger", Type: "object"}}},
		{ID: "manual", Name: "Manual Trigger", Category: "triggers", Description: "Manually triggered", Inputs: []NodePort{}, Outputs: []NodePort{{ID: "trigger", Name: "trigger", Type: "object"}}},
		// Actions
		{ID: "http_request", Name: "HTTP Request", Category: "actions", Description: "Make HTTP requests", Inputs: []NodePort{{ID: "data", Name: "data", Type: "object"}}, Outputs: []NodePort{{ID: "response", Name: "response", Type: "object"}}},
		{ID: "database", Name: "Database", Category: "actions", Description: "Database operations", Inputs: []NodePort{{ID: "query", Name: "query", Type: "object"}}, Outputs: []NodePort{{ID: "result", Name: "result", Type: "array"}}},
		{ID: "email_send", Name: "Send Email", Category: "actions", Description: "Send an email", Inputs: []NodePort{{ID: "data", Name: "data", Type: "object"}}, Outputs: []NodePort{{ID: "result", Name: "result", Type: "object"}}},
		// Logic
		{ID: "if", Name: "If", Category: "logic", Description: "Conditional branching", Inputs: []NodePort{{ID: "data", Name: "data", Type: "any"}}, Outputs: []NodePort{{ID: "true", Name: "true", Type: "any"}, {ID: "false", Name: "false", Type: "any"}}},
		{ID: "switch", Name: "Switch", Category: "logic", Description: "Multi-way branching", Inputs: []NodePort{{ID: "data", Name: "data", Type: "any"}}, Outputs: []NodePort{{ID: "default", Name: "default", Type: "any"}}},
		{ID: "loop", Name: "Loop", Category: "logic", Description: "Iterate over items", Inputs: []NodePort{{ID: "items", Name: "items", Type: "array"}}, Outputs: []NodePort{{ID: "item", Name: "item", Type: "any"}, {ID: "done", Name: "done", Type: "array"}}},
		{ID: "merge", Name: "Merge", Category: "logic", Description: "Merge multiple inputs", Inputs: []NodePort{{ID: "input1", Name: "input1", Type: "any"}, {ID: "input2", Name: "input2", Type: "any"}}, Outputs: []NodePort{{ID: "output", Name: "output", Type: "any"}}},
		// Data
		{ID: "set", Name: "Set", Category: "data", Description: "Set values", Inputs: []NodePort{{ID: "data", Name: "data", Type: "any"}}, Outputs: []NodePort{{ID: "data", Name: "data", Type: "any"}}},
		{ID: "code", Name: "Code", Category: "data", Description: "Run custom code", Inputs: []NodePort{{ID: "data", Name: "data", Type: "any"}}, Outputs: []NodePort{{ID: "result", Name: "result", Type: "any"}}},
		{ID: "transform", Name: "Transform", Category: "data", Description: "Transform data", Inputs: []NodePort{{ID: "data", Name: "data", Type: "any"}}, Outputs: []NodePort{{ID: "result", Name: "result", Type: "any"}}},
	}
}

// extractJSON extracts JSON from a text response.
func extractJSON(text string) string {
	// Try to find JSON in code blocks
	if start := strings.Index(text, "```json"); start != -1 {
		start += 7
		if end := strings.Index(text[start:], "```"); end != -1 {
			return strings.TrimSpace(text[start : start+end])
		}
	}
	if start := strings.Index(text, "```"); start != -1 {
		start += 3
		if end := strings.Index(text[start:], "```"); end != -1 {
			return strings.TrimSpace(text[start : start+end])
		}
	}
	// Try to find raw JSON
	if start := strings.Index(text, "{"); start != -1 {
		depth := 0
		for i := start; i < len(text); i++ {
			if text[i] == '{' {
				depth++
			} else if text[i] == '}' {
				depth--
				if depth == 0 {
					return text[start : i+1]
				}
			}
		}
	}
	return text
}
