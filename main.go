package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/sujit/ai-agent/pkg/agent"
	"github.com/sujit/ai-agent/pkg/chain"
	"github.com/sujit/ai-agent/pkg/cot"
	"github.com/sujit/ai-agent/pkg/llm"
	"github.com/sujit/ai-agent/pkg/tot"
)

func main() {
	mode := flag.String("mode", "agent", "Mode to run: agent, chain, cot, tot")
	prompt := flag.String("prompt", "Hello!", "Input prompt")
	apiKey := os.Getenv("OPENAI_API_KEY")

	flag.Parse()

	if apiKey == "" {
		log.Fatal("Please set OPENAI_API_KEY environment variable")
	}

	provider := llm.NewOpenAIProvider(apiKey, "gpt-4o")

	ctx := context.Background()
	var response string
	var err error

	fmt.Printf("Running mode: %s\nInput: %s\n\n", *mode, *prompt)

	switch *mode {
	case "agent":
		ag := agent.NewAgent("SimpleAgent", provider, "You are a helpful assistant.")
		response, err = ag.Chat(ctx, *prompt)

	case "chain":
		// Example: Generating a topic, then writing a poem about it
		link1 := &chain.LambdaLink{
			Func: func(ctx context.Context, input string) (string, error) {
				return provider.Generate(ctx, "Generate a random creative topic based on: "+input, nil)
			},
		}
		link2 := &chain.LambdaLink{
			Func: func(ctx context.Context, input string) (string, error) {
				fmt.Printf("Link 1 Output: %s\n", input)
				return provider.Generate(ctx, "Write a short poem about: "+input, nil)
			},
		}
		c := chain.NewSequentialChain(link1, link2)
		response, err = c.Run(ctx, *prompt)

	case "cot":
		ag := agent.NewAgent("ReasoningAgent", provider, "You are a logical thinker.")
		cotAgent := cot.NewCoTAgent(ag)
		response, err = cotAgent.Run(ctx, *prompt)

	case "tot":
		gen := &tot.LLMGenerator{Provider: provider, Model: "gpt-4o"}
		eval := &tot.LLMEvaluator{Provider: provider, Model: "gpt-4o"}
		solver := tot.NewTreeOfThoughts(gen, eval, 3, 3) // depth 3, breadth 3
		response, err = solver.BFS(ctx, *prompt)
		if err == nil {
			response = "Best solution found via ToT:\n" + response
		}

	default:
		log.Fatalf("Unknown mode: %s", *mode)
	}

	if err != nil {
		log.Fatalf("Error: %v", err)
	}

	fmt.Printf("\n--- Output ---\n%s\n", response)
}
