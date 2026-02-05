package tot

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/oarkflow/ai-agent/pkg/llm"
)

// Node represents a state in the tree of thoughts.
type Node struct {
	State    string
	Value    float64
	Parent   *Node
	Children []*Node
	Depth    int
}

// Generator proposes new thoughts based on the current state.
type Generator interface {
	GenerateThoughts(ctx context.Context, history []llm.Message, state string, k int) ([]string, error)
}

// Evaluator scores a state.
type Evaluator interface {
	Evaluate(ctx context.Context, history []llm.Message, state string) (float64, error)
}

// TreeOfThoughts implements the ToT search algorithm.
type TreeOfThoughts struct {
	Generator Generator
	Evaluator Evaluator
	MaxDepth  int
	Breadth   int // Number of thoughts to generate at each step (k)
}

// NewTreeOfThoughts creates a new ToT solver.
func NewTreeOfThoughts(gen Generator, eval Evaluator, maxDepth, breadth int) *TreeOfThoughts {
	return &TreeOfThoughts{
		Generator: gen,
		Evaluator: eval,
		MaxDepth:  maxDepth,
		Breadth:   breadth,
	}
}

// BFS executes Breadth-First Search to find the best thought path.
// history is the context (e.g. user question + background)
func (t *TreeOfThoughts) BFS(ctx context.Context, history []llm.Message, initialPrompt string) (string, error) {
	root := &Node{State: initialPrompt, Value: 1.0, Depth: 0}
	currentLevel := []*Node{root}

	for depth := 1; depth <= t.MaxDepth; depth++ {
		nextLevel := []*Node{}
		var mu sync.Mutex
		var wg sync.WaitGroup

		fmt.Printf("Depth %d: Exploring %d nodes...\n", depth, len(currentLevel))

		// Expand each node in current level
		for _, node := range currentLevel {
			// Generate thoughts
			thoughts, err := t.Generator.GenerateThoughts(ctx, history, node.State, t.Breadth)
			if err != nil {
				return "", err
			}

			// Evaluate thoughts concurrently
			for _, thought := range thoughts {
				wg.Add(1)
				go func(parent *Node, th string) {
					defer wg.Done()

					// State aggregation: usually we append the new thought to the history
					// Simple implementation: just append text
					newState := parent.State + "\n" + th

					score, err := t.Evaluator.Evaluate(ctx, history, newState)
					if err != nil {
						fmt.Printf("Eval error: %v\n", err)
						return
					}

					// Pruning: only keep "good" thoughts (e.g., score > 0.5)
					if score > 0.5 {
						newNode := &Node{
							State:  newState,
							Value:  score,
							Parent: parent,
							Depth:  depth,
						}
						mu.Lock()
						parent.Children = append(parent.Children, newNode)
						nextLevel = append(nextLevel, newNode)
						mu.Unlock()
					}
				}(node, thought)
			}
		}
		wg.Wait()

		if len(nextLevel) == 0 {
			break
		}

		// Sort nextLevel by value (optional, for beam search) but here strictly BFS implies keeping all acceptable
		// For simplicity, we just update currentLevel
		currentLevel = nextLevel
	}

	// Find best leaf
	var bestNode *Node
	maxVal := -1.0
	for _, node := range currentLevel {
		if node.Value > maxVal {
			maxVal = node.Value
			bestNode = node
		}
	}

	if bestNode != nil {
		return bestNode.State, nil
	}
	return "", fmt.Errorf("no solution found")
}

// LLMGenerator uses an LLM to propose thoughts.
type LLMGenerator struct {
	Provider llm.Provider
	Model    string
}

func (g *LLMGenerator) GenerateThoughts(ctx context.Context, history []llm.Message, state string, k int) ([]string, error) {
	// combine history + instruction
	messages := make([]llm.Message, len(history))
	copy(messages, history)

	prompt := fmt.Sprintf("Given the current state/plan:\n%s\n\nGenerate %d distinct next steps or thoughts to solve the problem. List them one by one.", state, k)
	messages = append(messages, llm.Message{Role: llm.RoleUser, Content: prompt})

	resp, err := g.Provider.Chat(ctx, messages, &llm.GenerateOptions{Model: g.Model})
	if err != nil {
		return nil, err
	}
	// Naive parsing: split by newlines
	lines := strings.Split(resp, "\n")
	var thoughts []string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" {
			thoughts = append(thoughts, trimmed)
		}
	}
	// Limit to k (though parsing might be loose)
	if len(thoughts) > k {
		thoughts = thoughts[:k]
	}
	return thoughts, nil
}

// LLMEvaluator uses an LLM to score a state.
type LLMEvaluator struct {
	Provider llm.Provider
	Model    string
}

func (e *LLMEvaluator) Evaluate(ctx context.Context, history []llm.Message, state string) (float64, error) {
	// combine history + instruction
	messages := make([]llm.Message, len(history))
	copy(messages, history)

	prompt := fmt.Sprintf("Evaluate the following state for correctness and progress towards the solution (0.0 to 1.0):\n%s\n\nOutput only the number.", state)
	messages = append(messages, llm.Message{Role: llm.RoleUser, Content: prompt})

	resp, err := e.Provider.Chat(ctx, messages, &llm.GenerateOptions{Model: e.Model})
	if err != nil {
		return 0, err
	}

	var score float64
	_, err = fmt.Sscanf(strings.TrimSpace(resp), "%f", &score)
	if err != nil {
		// Fallback if LLM is chatty
		return 0.5, nil
	}
	return score, nil
}
