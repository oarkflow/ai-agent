package tools

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"

	"github.com/sujit/ai-agent/pkg/config"
)

// ToolLoader loads tools from configuration.
type ToolLoader struct {
	config   *config.Config
	handlers map[string]ToolHandler
}

// NewToolLoader creates a new tool loader.
func NewToolLoader(cfg *config.Config) *ToolLoader {
	loader := &ToolLoader{
		config:   cfg,
		handlers: make(map[string]ToolHandler),
	}

	// Register built-in handlers
	loader.registerBuiltinHandlers()

	return loader
}

// RegisterHandler registers a custom handler for a tool.
func (tl *ToolLoader) RegisterHandler(name string, handler ToolHandler) {
	tl.handlers[name] = handler
}

// LoadTools loads all enabled tools from configuration.
func (tl *ToolLoader) LoadTools() ([]*Tool, error) {
	var tools []*Tool

	if tl.config.Tools == nil {
		return tools, nil
	}

	for _, toolCfg := range tl.config.Tools.Tools {
		if !toolCfg.Enabled {
			continue
		}

		tool, err := tl.convertToolCfg(toolCfg)
		if err != nil {
			// Log warning but continue with other tools
			continue
		}

		tools = append(tools, tool)
	}

	return tools, nil
}

// LoadTool loads a single tool by name.
func (tl *ToolLoader) LoadTool(name string) (*Tool, error) {
	toolCfg, ok := tl.config.GetTool(name)
	if !ok {
		return nil, fmt.Errorf("tool not found: %s", name)
	}

	return tl.convertToolCfg(toolCfg)
}

// convertToolCfg converts a config.ToolCfg to Tool.
func (tl *ToolLoader) convertToolCfg(cfg *config.ToolCfg) (*Tool, error) {
	tool := &Tool{
		Name:        cfg.Name,
		Description: cfg.Description,
		Parameters:  tl.convertParameters(cfg.Parameters),
		Metadata:    make(map[string]any),
	}

	// Resolve handler
	handler, err := tl.resolveHandler(cfg)
	if err != nil {
		return nil, err
	}
	tool.Handler = handler

	// Store additional config in metadata
	if cfg.Security != nil {
		tool.Metadata["security"] = cfg.Security
	}
	if cfg.Sandbox != nil {
		tool.Metadata["sandbox"] = cfg.Sandbox
	}

	return tool, nil
}

// convertParameters converts config parameters to ParameterSchema.
func (tl *ToolLoader) convertParameters(params *config.ToolParametersCfg) *ParameterSchema {
	if params == nil {
		return nil
	}

	schema := &ParameterSchema{
		Type:       params.Type,
		Required:   params.Required,
		Properties: make(map[string]*PropertySchema),
	}

	for name, prop := range params.Properties {
		schema.Properties[name] = &PropertySchema{
			Type:        prop.Type,
			Description: prop.Description,
			Enum:        prop.Enum,
			Default:     prop.Default,
			Minimum:     prop.Minimum,
			Maximum:     prop.Maximum,
		}
	}

	return schema
}

// resolveHandler resolves the handler for a tool.
func (tl *ToolLoader) resolveHandler(cfg *config.ToolCfg) (ToolHandler, error) {
	if cfg.Handler == "" && cfg.Endpoint == nil {
		return nil, fmt.Errorf("tool %s has no handler or endpoint configured", cfg.Name)
	}

	// Check for built-in handlers
	if strings.HasPrefix(cfg.Handler, "builtin:") {
		handlerName := strings.TrimPrefix(cfg.Handler, "builtin:")
		if handler, ok := tl.handlers[handlerName]; ok {
			return handler, nil
		}
		return nil, fmt.Errorf("built-in handler not found: %s", handlerName)
	}

	// Check for custom registered handler
	if handler, ok := tl.handlers[cfg.Handler]; ok {
		return handler, nil
	}

	// HTTP endpoint handler
	if cfg.Endpoint != nil {
		return tl.createHTTPHandler(cfg.Endpoint), nil
	}

	return nil, fmt.Errorf("unable to resolve handler for tool: %s", cfg.Name)
}

// createHTTPHandler creates an HTTP-based handler for external endpoints.
func (tl *ToolLoader) createHTTPHandler(endpoint *config.ToolEndpointCfg) ToolHandler {
	return func(ctx context.Context, params map[string]any) (any, error) {
		// This would make an HTTP request to the configured endpoint
		// For now, return an error indicating this needs to be implemented
		return nil, fmt.Errorf("HTTP endpoint handler not yet implemented for: %s", endpoint.URL)
	}
}

// registerBuiltinHandlers registers the built-in tool handlers.
func (tl *ToolLoader) registerBuiltinHandlers() {
	// Calculator
	tl.handlers["calculator"] = func(ctx context.Context, params map[string]any) (any, error) {
		expr, ok := params["expression"].(string)
		if !ok {
			return nil, fmt.Errorf("expression must be a string")
		}

		result, err := evaluateExpression(expr)
		if err != nil {
			return nil, err
		}

		precision := 10
		if p, ok := params["precision"].(float64); ok {
			precision = int(p)
		}

		return map[string]any{
			"expression": expr,
			"result":     roundToPrecision(result, precision),
		}, nil
	}

	// DateTime
	tl.handlers["datetime"] = func(ctx context.Context, params map[string]any) (any, error) {
		operation, _ := params["operation"].(string)
		if operation == "" {
			operation = "now"
		}

		timezone := "UTC"
		if tz, ok := params["timezone"].(string); ok {
			timezone = tz
		}

		loc, err := time.LoadLocation(timezone)
		if err != nil {
			loc = time.UTC
		}

		now := time.Now().In(loc)

		switch operation {
		case "now":
			return map[string]any{
				"datetime":  now.Format(time.RFC3339),
				"unix":      now.Unix(),
				"timezone":  timezone,
				"year":      now.Year(),
				"month":     int(now.Month()),
				"day":       now.Day(),
				"hour":      now.Hour(),
				"minute":    now.Minute(),
				"second":    now.Second(),
				"weekday":   now.Weekday().String(),
			}, nil

		case "format":
			format, _ := params["format"].(string)
			if format == "" {
				format = time.RFC3339
			}
			return map[string]any{
				"formatted": now.Format(format),
			}, nil

		case "add", "subtract":
			duration, _ := params["duration"].(string)
			d, err := parseDuration(duration)
			if err != nil {
				return nil, err
			}
			if operation == "subtract" {
				d = -d
			}
			result := now.Add(d)
			return map[string]any{
				"original": now.Format(time.RFC3339),
				"result":   result.Format(time.RFC3339),
				"duration": duration,
			}, nil

		default:
			return nil, fmt.Errorf("unknown operation: %s", operation)
		}
	}

	// Web Search (stub - would need actual implementation)
	tl.handlers["web_search"] = func(ctx context.Context, params map[string]any) (any, error) {
		query, _ := params["query"].(string)
		return map[string]any{
			"query":   query,
			"results": []any{},
			"message": "Web search requires external API configuration",
		}, nil
	}

	// Code Execute (stub - would need sandbox implementation)
	tl.handlers["code_execute"] = func(ctx context.Context, params map[string]any) (any, error) {
		return nil, fmt.Errorf("code execution is disabled for security reasons")
	}

	// File Read (stub)
	tl.handlers["file_read"] = func(ctx context.Context, params map[string]any) (any, error) {
		return nil, fmt.Errorf("file read requires security configuration")
	}

	// File Write (stub)
	tl.handlers["file_write"] = func(ctx context.Context, params map[string]any) (any, error) {
		return nil, fmt.Errorf("file write requires security configuration")
	}

	// HTTP Request (stub)
	tl.handlers["http_request"] = func(ctx context.Context, params map[string]any) (any, error) {
		return nil, fmt.Errorf("HTTP requests require domain allowlist configuration")
	}

	// Database Query (stub)
	tl.handlers["database_query"] = func(ctx context.Context, params map[string]any) (any, error) {
		return nil, fmt.Errorf("database queries require connection configuration")
	}

	// Vector Search (stub)
	tl.handlers["vector_search"] = func(ctx context.Context, params map[string]any) (any, error) {
		query, _ := params["query"].(string)
		return map[string]any{
			"query":   query,
			"results": []any{},
			"message": "Vector search requires vector store configuration",
		}, nil
	}

	// Image Generate (stub)
	tl.handlers["image_generate"] = func(ctx context.Context, params map[string]any) (any, error) {
		return nil, fmt.Errorf("image generation requires provider configuration")
	}
}

// Helper functions

func evaluateExpression(expr string) (float64, error) {
	// Basic expression evaluator
	// In production, use a proper math parser library
	expr = strings.TrimSpace(expr)

	// Handle simple arithmetic only for now
	// This is a placeholder - real implementation would use a proper parser
	return 0, fmt.Errorf("expression evaluation not implemented: %s", expr)
}

func roundToPrecision(value float64, precision int) float64 {
	multiplier := math.Pow(10, float64(precision))
	return math.Round(value*multiplier) / multiplier
}

func parseDuration(s string) (time.Duration, error) {
	// Standard Go duration
	if d, err := time.ParseDuration(s); err == nil {
		return d, nil
	}

	// Extended format: "7d", "1w", "1M", "1y"
	s = strings.TrimSpace(s)
	if len(s) < 2 {
		return 0, fmt.Errorf("invalid duration: %s", s)
	}

	numStr := s[:len(s)-1]
	unit := s[len(s)-1:]

	num, err := strconv.ParseFloat(numStr, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid duration number: %s", s)
	}

	switch unit {
	case "d":
		return time.Duration(num * 24 * float64(time.Hour)), nil
	case "w":
		return time.Duration(num * 7 * 24 * float64(time.Hour)), nil
	case "M":
		return time.Duration(num * 30 * 24 * float64(time.Hour)), nil
	case "y":
		return time.Duration(num * 365 * 24 * float64(time.Hour)), nil
	default:
		return 0, fmt.Errorf("unknown duration unit: %s", unit)
	}
}

// NewToolRegistryFromConfig creates a tool registry from configuration.
func NewToolRegistryFromConfig(cfg *config.Config) (*ToolRegistry, error) {
	loader := NewToolLoader(cfg)
	registry := NewToolRegistry()

	tools, err := loader.LoadTools()
	if err != nil {
		return nil, err
	}

	for _, tool := range tools {
		if err := registry.Register(tool); err != nil {
			// Log warning but continue
			continue
		}
	}

	return registry, nil
}
