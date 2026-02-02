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
	"github.com/sujit/ai-agent/pkg/memory"
	"github.com/sujit/ai-agent/pkg/prompt"
	"github.com/sujit/ai-agent/pkg/tot"
)

func main() {
	mode := flag.String("mode", "agent", "Mode to run: agent, chain, cot, tot, smart, memory")
	input := flag.String("prompt", "Hello!", "Input prompt")
	apiKey := os.Getenv("OPENAI_API_KEY")

	flag.Parse()

	if apiKey == "" {
		log.Fatal("Please set OPENAI_API_KEY environment variable")
	}

	provider := llm.NewOpenAIProvider(apiKey, "gpt-4o")
	ctx := context.Background()
	var response string
	var err error

	fmt.Printf("Running mode: %s\nInput: %s\n\n", *mode, *input)

	switch *mode {
	case "agent":
		ag := agent.NewAgent("SimpleAgent", provider, "You are a helpful assistant.")
		response, err = ag.Chat(ctx, *input)

	case "chain":
		link1 := &chain.LambdaLink{
			Func: func(ctx context.Context, in string) (string, error) {
				return provider.Generate(ctx, "Generate a random creative topic based on: "+in, nil)
			},
		}
		link2 := &chain.LambdaLink{
			Func: func(ctx context.Context, in string) (string, error) {
				fmt.Printf("Link 1 Output: %s\n", in)
				return provider.Generate(ctx, "Write a short poem about: "+in, nil)
			},
		}
		c := chain.NewSequentialChain(link1, link2)
		response, err = c.Run(ctx, *input)

	case "cot":
		ag := agent.NewAgent("ReasoningAgent", provider, "You are a logical thinker.")
		cotAgent := cot.NewCoTAgent(ag)
		response, err = cotAgent.Run(ctx, *input)

	case "tot":
		gen := &tot.LLMGenerator{Provider: provider, Model: "gpt-4o"}
		eval := &tot.LLMEvaluator{Provider: provider, Model: "gpt-4o"}
		solver := tot.NewTreeOfThoughts(gen, eval, 3, 3)
		// Demonstrate context usage:
		history := []llm.Message{
			{Role: llm.RoleSystem, Content: "You are a creative writer."},
			{Role: llm.RoleUser, Content: "I want a story about a dragon."},
		}
		// In a real app, you'd get this from memory.GetHistory()
		response, err = solver.BFS(ctx, history, *input)
		if err == nil {
			response = "Best solution found via ToT:\n" + response
		}

	case "smart":
		// Build a SMART prompt programmatically
		builder := prompt.NewSMARTPrompt().
			WithRole("Expert Golang Engineer").
			WithContext("The user wants to write high-performance Go code.").
			WithTask(*input).
			AddConstraint("Use standard library only").
			WithOutputFormat("Markdown code block")

		fullPrompt := builder.String()
		fmt.Printf("--- Constructed SMART Prompt ---\n%s\n------------------------------\n", fullPrompt)

		ag := agent.NewAgent("SmartAgent", provider, "")
		response, err = ag.Chat(ctx, fullPrompt)

	case "memory":
		// Demonstrate Summary Memory
		// We simulate a conversation loop effectively by manually adding messages
		mem := memory.NewSummaryMemory(provider, 2) // Very small window to trigger summary
		ag := agent.NewAgent("MemoryAgent", provider, "You are a chatty friend.").WithMemory(mem)

		inputs := []string{
			"Hi, I'm Sujit.",
			"I like coding in Go.",
			"What is my name?",
			"What do I like?",
		}

		for _, in := range inputs {
			fmt.Printf("User: %s\n", in)
			resp, e := ag.Chat(ctx, in)
			if e != nil {
				err = e
				break
			}
			fmt.Printf("AI: %s\n", resp)
		}
		response = "Conversation finished."

	default:
		log.Fatalf("Unknown mode: %s", *mode)
	}

	if err != nil {
		log.Fatalf("Error: %v", err)
	}

	fmt.Printf("\n--- Output ---\n%s\n", response)
}
