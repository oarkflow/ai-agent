package chain

import (
	"context"
	"fmt"
)

// Link represents a single step in a chain.
type Link interface {
	Run(ctx context.Context, input string) (string, error)
}

// SequentialChain executes a list of links in order.
// The output of one link becomes the input of the next.
type SequentialChain struct {
	Links []Link
}

// NewSequentialChain creates a new sequential chain.
func NewSequentialChain(links ...Link) *SequentialChain {
	return &SequentialChain{Links: links}
}

// Run executes the chain.
func (c *SequentialChain) Run(ctx context.Context, input string) (string, error) {
	currentInput := input
	var err error

	for i, link := range c.Links {
		// Calculate step output
		currentInput, err = link.Run(ctx, currentInput)
		if err != nil {
			return "", fmt.Errorf("chain failed at step %d: %w", i, err)
		}
	}

	return currentInput, nil
}

// LambdaLink allows wrapping a simple function as a chain link.
type LambdaLink struct {
	Func func(context.Context, string) (string, error)
}

func (l *LambdaLink) Run(ctx context.Context, input string) (string, error) {
	return l.Func(ctx, input)
}
