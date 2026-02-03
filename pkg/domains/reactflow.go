package domains

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/sujit/ai-agent/pkg/content"
	"github.com/sujit/ai-agent/pkg/llm"
	"github.com/sujit/ai-agent/pkg/storage"
)

// ReactFlowDomain handles ReactFlow (@xyflow/react) JSON generation.
type ReactFlowDomain struct {
	*BaseDomain
}

// ReactFlowNode represents a node in ReactFlow format.
type ReactFlowNode struct {
	ID             string         `json:"id"`
	Type           string         `json:"type,omitempty"`
	Position       XYPosition     `json:"position"`
	Data           map[string]any `json:"data"`
	Style          map[string]any `json:"style,omitempty"`
	ClassName      string         `json:"className,omitempty"`
	SourcePosition string         `json:"sourcePosition,omitempty"`
	TargetPosition string         `json:"targetPosition,omitempty"`
	Hidden         bool           `json:"hidden,omitempty"`
	Selected       bool           `json:"selected,omitempty"`
	Dragging       bool           `json:"dragging,omitempty"`
	Selectable     bool           `json:"selectable,omitempty"`
	Connectable    bool           `json:"connectable,omitempty"`
	Deletable      bool           `json:"deletable,omitempty"`
	Width          float64        `json:"width,omitempty"`
	Height         float64        `json:"height,omitempty"`
	ParentId       string         `json:"parentId,omitempty"`
	Extent         string         `json:"extent,omitempty"`
	AriaLabel      string         `json:"ariaLabel,omitempty"`
}

// XYPosition represents x,y coordinates.
type XYPosition struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

// ReactFlowEdge represents an edge in ReactFlow format.
type ReactFlowEdge struct {
	ID               string         `json:"id"`
	Source           string         `json:"source"`
	Target           string         `json:"target"`
	Type             string         `json:"type,omitempty"`
	SourceHandle     string         `json:"sourceHandle,omitempty"`
	TargetHandle     string         `json:"targetHandle,omitempty"`
	Animated         bool           `json:"animated,omitempty"`
	Label            string         `json:"label,omitempty"`
	LabelStyle       map[string]any `json:"labelStyle,omitempty"`
	LabelBgStyle     map[string]any `json:"labelBgStyle,omitempty"`
	MarkerStart      *EdgeMarker    `json:"markerStart,omitempty"`
	MarkerEnd        *EdgeMarker    `json:"markerEnd,omitempty"`
	Style            map[string]any `json:"style,omitempty"`
	ClassName        string         `json:"className,omitempty"`
	Hidden           bool           `json:"hidden,omitempty"`
	Data             map[string]any `json:"data,omitempty"`
	Selected         bool           `json:"selected,omitempty"`
	Interactionwidth float64        `json:"interactionWidth,omitempty"`
}

// EdgeMarker represents edge markers (arrows, etc.).
type EdgeMarker struct {
	Type        string `json:"type,omitempty"`
	Color       string `json:"color,omitempty"`
	Width       int    `json:"width,omitempty"`
	Height      int    `json:"height,omitempty"`
	MarkerUnits string `json:"markerUnits,omitempty"`
	Orient      string `json:"orient,omitempty"`
	StrokeWidth int    `json:"strokeWidth,omitempty"`
}

// ReactFlowGraph represents a complete ReactFlow graph.
type ReactFlowGraph struct {
	Nodes     []ReactFlowNode   `json:"nodes"`
	Edges     []ReactFlowEdge   `json:"edges"`
	Viewport  *Viewport         `json:"viewport,omitempty"`
	NodeTypes map[string]string `json:"nodeTypes,omitempty"`
	EdgeTypes map[string]string `json:"edgeTypes,omitempty"`
}

// Viewport represents the viewport state.
type Viewport struct {
	X    float64 `json:"x"`
	Y    float64 `json:"y"`
	Zoom float64 `json:"zoom"`
}

// NewReactFlowDomain creates a new ReactFlow domain trainer.
func NewReactFlowDomain(provider llm.MultimodalProvider, storage *storage.Storage) *ReactFlowDomain {
	return &ReactFlowDomain{
		BaseDomain: NewBaseDomain("reactflow", "ReactFlow JSON Generation", provider, storage),
	}
}

// GetSystemPrompt returns the system prompt for ReactFlow generation.
func (d *ReactFlowDomain) GetSystemPrompt() string {
	return `You are an expert at generating ReactFlow (@xyflow/react) JSON configurations for interactive node-based diagrams.

Your expertise includes:
- Creating valid ReactFlow node and edge configurations
- Proper positioning of nodes for clean layouts
- Configuring node types, handles, and styles
- Creating appropriate edge connections with labels and markers
- Supporting custom node components and styling

ReactFlow JSON Structure:
- nodes: Array of node objects with id, type, position, data
- edges: Array of edge objects with id, source, target
- viewport: Optional viewport configuration

Node types:
- default: Standard node
- input: Input node (no target handles)
- output: Output node (no source handles)
- group: Container for sub-nodes

Edge types:
- default: Straight line
- straight: Straight connection
- step: Step/staircase connection
- smoothstep: Smooth step connection
- bezier: Bezier curve

Position conventions:
- sourcePosition: 'top', 'right', 'bottom', 'left'
- targetPosition: 'top', 'right', 'bottom', 'left'

Always generate valid JSON that can be directly used with ReactFlow.`
}

// GenerateFlowGraph generates a ReactFlow graph from description.
func (d *ReactFlowDomain) GenerateFlowGraph(ctx context.Context, description string, options *GenerationOptions) (*ReactFlowGraph, error) {
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

	optionsStr := ""
	if options != nil {
		if options.Layout != "" {
			optionsStr += fmt.Sprintf("\n- Layout: %s", options.Layout)
		}
		if options.Theme != "" {
			optionsStr += fmt.Sprintf("\n- Theme: %s", options.Theme)
		}
		if len(options.CustomNodeTypes) > 0 {
			optionsStr += fmt.Sprintf("\n- Custom node types: %v", options.CustomNodeTypes)
		}
	}

	prompt := fmt.Sprintf(`Generate a ReactFlow graph for:

%s%s

Respond with a valid ReactFlow JSON object containing:
- nodes: Array of properly positioned nodes with data
- edges: Array of edges connecting the nodes
- viewport: Initial viewport settings`, description, optionsStr)

	messages = append(messages, content.NewTextMessage(content.RoleUser, prompt))

	resp, err := d.provider.Generate(ctx, messages, &llm.GenerationConfig{
		Temperature: 0.3,
		MaxTokens:   4096,
	})
	if err != nil {
		return nil, err
	}

	responseText := resp.Message.GetText()
	jsonStr := extractReactFlowJSON(responseText)

	var graph ReactFlowGraph
	if err := json.Unmarshal([]byte(jsonStr), &graph); err != nil {
		return nil, fmt.Errorf("failed to parse ReactFlow JSON: %w", err)
	}

	// Validate and fix the graph
	graph = d.normalizeGraph(graph)

	return &graph, nil
}

// GenerationOptions for ReactFlow generation.
type GenerationOptions struct {
	Layout          string   `json:"layout"`            // horizontal, vertical, tree, radial
	Theme           string   `json:"theme"`             // light, dark, custom
	CustomNodeTypes []string `json:"custom_node_types"` // custom node type names
	Spacing         float64  `json:"spacing"`           // node spacing
	Animated        bool     `json:"animated"`          // animate edges
}

// GenerateFromData generates a ReactFlow graph from structured data.
func (d *ReactFlowDomain) GenerateFromData(ctx context.Context, data any) (*ReactFlowGraph, error) {
	dataJSON, err := json.Marshal(data)
	if err != nil {
		return nil, err
	}

	prompt := fmt.Sprintf(`Convert this data structure into a ReactFlow graph:

%s

Create appropriate nodes for each data item and edges for relationships.`, string(dataJSON))

	return d.GenerateFlowGraph(ctx, prompt, nil)
}

// normalizeGraph normalizes and validates a ReactFlow graph.
func (d *ReactFlowDomain) normalizeGraph(graph ReactFlowGraph) ReactFlowGraph {
	// Ensure all nodes have IDs
	nodeIDs := make(map[string]bool)
	for i := range graph.Nodes {
		if graph.Nodes[i].ID == "" {
			graph.Nodes[i].ID = fmt.Sprintf("node_%d", i+1)
		}
		nodeIDs[graph.Nodes[i].ID] = true

		// Ensure data is not nil
		if graph.Nodes[i].Data == nil {
			graph.Nodes[i].Data = make(map[string]any)
		}
	}

	// Validate edges
	var validEdges []ReactFlowEdge
	for i := range graph.Edges {
		if graph.Edges[i].ID == "" {
			graph.Edges[i].ID = fmt.Sprintf("edge_%d", i+1)
		}
		// Only include edges with valid source and target
		if nodeIDs[graph.Edges[i].Source] && nodeIDs[graph.Edges[i].Target] {
			validEdges = append(validEdges, graph.Edges[i])
		}
	}
	graph.Edges = validEdges

	// Set default viewport if not provided
	if graph.Viewport == nil {
		graph.Viewport = &Viewport{X: 0, Y: 0, Zoom: 1}
	}

	return graph
}

// ApplyLayout applies automatic layout to nodes.
func (d *ReactFlowDomain) ApplyLayout(graph *ReactFlowGraph, layout string, spacing float64) {
	if spacing <= 0 {
		spacing = 150
	}

	switch layout {
	case "horizontal":
		d.applyHorizontalLayout(graph, spacing)
	case "vertical":
		d.applyVerticalLayout(graph, spacing)
	case "grid":
		d.applyGridLayout(graph, spacing)
	default:
		d.applyVerticalLayout(graph, spacing)
	}
}

func (d *ReactFlowDomain) applyHorizontalLayout(graph *ReactFlowGraph, spacing float64) {
	for i := range graph.Nodes {
		graph.Nodes[i].Position.X = float64(i) * spacing
		graph.Nodes[i].Position.Y = 0
	}
}

func (d *ReactFlowDomain) applyVerticalLayout(graph *ReactFlowGraph, spacing float64) {
	for i := range graph.Nodes {
		graph.Nodes[i].Position.X = 0
		graph.Nodes[i].Position.Y = float64(i) * spacing
	}
}

func (d *ReactFlowDomain) applyGridLayout(graph *ReactFlowGraph, spacing float64) {
	cols := 3
	for i := range graph.Nodes {
		graph.Nodes[i].Position.X = float64(i%cols) * spacing
		graph.Nodes[i].Position.Y = float64(i/cols) * spacing
	}
}

// ExportToReact generates React component code for the graph.
func (d *ReactFlowDomain) ExportToReact(graph *ReactFlowGraph) string {
	nodesJSON := marshalJSON(graph.Nodes)
	edgesJSON := marshalJSON(graph.Edges)

	return fmt.Sprintf(`import React from 'react';
import { ReactFlow, Background, Controls, MiniMap } from '@xyflow/react';
import '@xyflow/react/dist/style.css';

const initialNodes = %s;

const initialEdges = %s;

export default function Flow() {
  return (
    <div style={{ width: '100%%', height: '100vh' }}>
      <ReactFlow
        nodes={initialNodes}
        edges={initialEdges}
        fitView
      >
        <Background />
        <Controls />
        <MiniMap />
      </ReactFlow>
    </div>
  );
}
`, nodesJSON, edgesJSON)
}

func marshalJSON(v any) string {
	data, _ := json.MarshalIndent(v, "", "  ")
	return string(data)
}

func extractReactFlowJSON(text string) string {
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
