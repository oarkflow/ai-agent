package tools

import (
	"context"
	"encoding/json"
	"fmt"
)

// ToolHandler is the function signature for tool implementations.
type ToolHandler func(ctx context.Context, params map[string]any) (any, error)

// ParameterSchema defines the JSON Schema for tool parameters.
type ParameterSchema struct {
	Type       string                     `json:"type"`
	Properties map[string]*PropertySchema `json:"properties,omitempty"`
	Required   []string                   `json:"required,omitempty"`
	Items      *PropertySchema            `json:"items,omitempty"`
}

// PropertySchema defines a single property in the schema.
type PropertySchema struct {
	Type        string                     `json:"type"`
	Description string                     `json:"description,omitempty"`
	Enum        []string                   `json:"enum,omitempty"`
	Items       *PropertySchema            `json:"items,omitempty"`
	Properties  map[string]*PropertySchema `json:"properties,omitempty"`
	Required    []string                   `json:"required,omitempty"`
	Default     any                        `json:"default,omitempty"`
	Minimum     *float64                   `json:"minimum,omitempty"`
	Maximum     *float64                   `json:"maximum,omitempty"`
	MinLength   *int                       `json:"minLength,omitempty"`
	MaxLength   *int                       `json:"maxLength,omitempty"`
}

// Tool represents a callable tool/function for function calling.
type Tool struct {
	Name        string           `json:"name"`
	Description string           `json:"description"`
	Parameters  *ParameterSchema `json:"parameters"`
	Handler     ToolHandler      `json:"-"`
	Metadata    map[string]any   `json:"metadata,omitempty"`
}

// ToolCall represents an AI's request to call a tool.
type ToolCall struct {
	ID        string         `json:"id"`
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

// ToolResult represents the result of a tool call.
type ToolResult struct {
	ToolCallID string `json:"tool_call_id"`
	Name       string `json:"name"`
	Content    string `json:"content"`
	Error      error  `json:"error,omitempty"`
}

// ToolRegistry manages available tools.
type ToolRegistry struct {
	tools map[string]*Tool
}

// NewToolRegistry creates a new tool registry.
func NewToolRegistry() *ToolRegistry {
	return &ToolRegistry{
		tools: make(map[string]*Tool),
	}
}

// Register adds a tool to the registry.
func (tr *ToolRegistry) Register(tool *Tool) error {
	if tool.Name == "" {
		return fmt.Errorf("tool name is required")
	}
	if tool.Handler == nil {
		return fmt.Errorf("tool handler is required")
	}
	tr.tools[tool.Name] = tool
	return nil
}

// Get retrieves a tool by name.
func (tr *ToolRegistry) Get(name string) (*Tool, bool) {
	tool, ok := tr.tools[name]
	return tool, ok
}

// List returns all registered tools.
func (tr *ToolRegistry) List() []*Tool {
	tools := make([]*Tool, 0, len(tr.tools))
	for _, t := range tr.tools {
		tools = append(tools, t)
	}
	return tools
}

// Execute runs a tool with given parameters.
func (tr *ToolRegistry) Execute(ctx context.Context, call *ToolCall) *ToolResult {
	tool, ok := tr.tools[call.Name]
	if !ok {
		return &ToolResult{
			ToolCallID: call.ID,
			Name:       call.Name,
			Error:      fmt.Errorf("tool not found: %s", call.Name),
		}
	}

	result, err := tool.Handler(ctx, call.Arguments)
	if err != nil {
		return &ToolResult{
			ToolCallID: call.ID,
			Name:       call.Name,
			Error:      err,
		}
	}

	// Convert result to string
	var content string
	switch v := result.(type) {
	case string:
		content = v
	case []byte:
		content = string(v)
	default:
		jsonBytes, _ := json.MarshalIndent(result, "", "  ")
		content = string(jsonBytes)
	}

	return &ToolResult{
		ToolCallID: call.ID,
		Name:       call.Name,
		Content:    content,
	}
}

// ToOpenAIFormat converts tools to OpenAI's function calling format.
func (tr *ToolRegistry) ToOpenAIFormat() []map[string]any {
	result := make([]map[string]any, 0, len(tr.tools))
	for _, tool := range tr.tools {
		result = append(result, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        tool.Name,
				"description": tool.Description,
				"parameters":  tool.Parameters,
			},
		})
	}
	return result
}

// ToAnthropicFormat converts tools to Anthropic's tool use format.
func (tr *ToolRegistry) ToAnthropicFormat() []map[string]any {
	result := make([]map[string]any, 0, len(tr.tools))
	for _, tool := range tr.tools {
		result = append(result, map[string]any{
			"name":         tool.Name,
			"description":  tool.Description,
			"input_schema": tool.Parameters,
		})
	}
	return result
}

// ToolBuilder provides a fluent API for building tools.
type ToolBuilder struct {
	tool *Tool
}

// NewToolBuilder creates a new tool builder.
func NewToolBuilder(name string) *ToolBuilder {
	return &ToolBuilder{
		tool: &Tool{
			Name: name,
			Parameters: &ParameterSchema{
				Type:       "object",
				Properties: make(map[string]*PropertySchema),
				Required:   make([]string, 0),
			},
		},
	}
}

// Description sets the tool description.
func (tb *ToolBuilder) Description(desc string) *ToolBuilder {
	tb.tool.Description = desc
	return tb
}

// AddStringParam adds a string parameter.
func (tb *ToolBuilder) AddStringParam(name, description string, required bool) *ToolBuilder {
	tb.tool.Parameters.Properties[name] = &PropertySchema{
		Type:        "string",
		Description: description,
	}
	if required {
		tb.tool.Parameters.Required = append(tb.tool.Parameters.Required, name)
	}
	return tb
}

// AddEnumParam adds an enum parameter.
func (tb *ToolBuilder) AddEnumParam(name, description string, options []string, required bool) *ToolBuilder {
	tb.tool.Parameters.Properties[name] = &PropertySchema{
		Type:        "string",
		Description: description,
		Enum:        options,
	}
	if required {
		tb.tool.Parameters.Required = append(tb.tool.Parameters.Required, name)
	}
	return tb
}

// AddNumberParam adds a number parameter.
func (tb *ToolBuilder) AddNumberParam(name, description string, required bool) *ToolBuilder {
	tb.tool.Parameters.Properties[name] = &PropertySchema{
		Type:        "number",
		Description: description,
	}
	if required {
		tb.tool.Parameters.Required = append(tb.tool.Parameters.Required, name)
	}
	return tb
}

// AddBoolParam adds a boolean parameter.
func (tb *ToolBuilder) AddBoolParam(name, description string, required bool) *ToolBuilder {
	tb.tool.Parameters.Properties[name] = &PropertySchema{
		Type:        "boolean",
		Description: description,
	}
	if required {
		tb.tool.Parameters.Required = append(tb.tool.Parameters.Required, name)
	}
	return tb
}

// AddArrayParam adds an array parameter.
func (tb *ToolBuilder) AddArrayParam(name, description, itemType string, required bool) *ToolBuilder {
	tb.tool.Parameters.Properties[name] = &PropertySchema{
		Type:        "array",
		Description: description,
		Items:       &PropertySchema{Type: itemType},
	}
	if required {
		tb.tool.Parameters.Required = append(tb.tool.Parameters.Required, name)
	}
	return tb
}

// AddObjectParam adds an object parameter.
func (tb *ToolBuilder) AddObjectParam(name, description string, properties map[string]*PropertySchema, required bool) *ToolBuilder {
	tb.tool.Parameters.Properties[name] = &PropertySchema{
		Type:        "object",
		Description: description,
		Properties:  properties,
	}
	if required {
		tb.tool.Parameters.Required = append(tb.tool.Parameters.Required, name)
	}
	return tb
}

// Handler sets the tool handler function.
func (tb *ToolBuilder) Handler(h ToolHandler) *ToolBuilder {
	tb.tool.Handler = h
	return tb
}

// Build returns the constructed tool.
func (tb *ToolBuilder) Build() *Tool {
	return tb.tool
}

// BuiltinTools provides commonly used tools.
var BuiltinTools = struct {
	WebSearch   func() *Tool
	Calculator  func() *Tool
	DateTime    func() *Tool
	CodeExecute func() *Tool
}{
	WebSearch: func() *Tool {
		return NewToolBuilder("web_search").
			Description("Search the web for information").
			AddStringParam("query", "The search query", true).
			AddNumberParam("num_results", "Number of results to return (default: 5)", false).
			Handler(func(ctx context.Context, params map[string]any) (any, error) {
				// Placeholder - implement with actual search API
				query := params["query"].(string)
				return map[string]any{
					"query":   query,
					"results": []string{"Result 1", "Result 2"},
				}, nil
			}).
			Build()
	},
	Calculator: func() *Tool {
		return NewToolBuilder("calculator").
			Description("Perform mathematical calculations").
			AddStringParam("expression", "The mathematical expression to evaluate", true).
			Handler(func(ctx context.Context, params map[string]any) (any, error) {
				// Placeholder - implement with actual calculator
				expr := params["expression"].(string)
				return map[string]any{
					"expression": expr,
					"result":     "42",
				}, nil
			}).
			Build()
	},
	DateTime: func() *Tool {
		return NewToolBuilder("datetime").
			Description("Get current date and time information").
			AddStringParam("timezone", "The timezone (default: UTC)", false).
			AddEnumParam("format", "Output format", []string{"iso", "human", "unix"}, false).
			Handler(func(ctx context.Context, params map[string]any) (any, error) {
				// Placeholder - implement with actual datetime logic
				return map[string]any{
					"datetime": "2026-01-15T10:30:00Z",
					"timezone": "UTC",
				}, nil
			}).
			Build()
	},
	CodeExecute: func() *Tool {
		return NewToolBuilder("execute_code").
			Description("Execute code in a sandboxed environment").
			AddStringParam("code", "The code to execute", true).
			AddEnumParam("language", "Programming language", []string{"python", "javascript", "go"}, true).
			Handler(func(ctx context.Context, params map[string]any) (any, error) {
				// Placeholder - implement with actual sandbox
				return map[string]any{
					"status": "executed",
					"output": "Hello, World!",
				}, nil
			}).
			Build()
	},
}

// ToolExecutor handles tool execution in a conversation loop.
type ToolExecutor struct {
	registry *ToolRegistry
	maxCalls int
}

// NewToolExecutor creates a new tool executor.
func NewToolExecutor(registry *ToolRegistry) *ToolExecutor {
	return &ToolExecutor{
		registry: registry,
		maxCalls: 10,
	}
}

// SetMaxCalls sets the maximum number of tool calls per turn.
func (te *ToolExecutor) SetMaxCalls(max int) {
	te.maxCalls = max
}

// ExecuteAll executes all tool calls and returns results.
func (te *ToolExecutor) ExecuteAll(ctx context.Context, calls []*ToolCall) []*ToolResult {
	results := make([]*ToolResult, 0, len(calls))
	for i, call := range calls {
		if i >= te.maxCalls {
			results = append(results, &ToolResult{
				ToolCallID: call.ID,
				Name:       call.Name,
				Error:      fmt.Errorf("maximum tool calls exceeded"),
			})
			break
		}
		result := te.registry.Execute(ctx, call)
		results = append(results, result)
	}
	return results
}

// ParallelExecuteAll executes all tool calls in parallel.
func (te *ToolExecutor) ParallelExecuteAll(ctx context.Context, calls []*ToolCall) []*ToolResult {
	if len(calls) > te.maxCalls {
		calls = calls[:te.maxCalls]
	}

	results := make([]*ToolResult, len(calls))
	done := make(chan struct{}, len(calls))

	for i, call := range calls {
		go func(idx int, c *ToolCall) {
			results[idx] = te.registry.Execute(ctx, c)
			done <- struct{}{}
		}(i, call)
	}

	for range calls {
		<-done
	}

	return results
}
