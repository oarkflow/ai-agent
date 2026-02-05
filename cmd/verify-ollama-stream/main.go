package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/oarkflow/ai-agent/pkg/content"
	"github.com/oarkflow/ai-agent/pkg/llm"
)

func main() {
	fmt.Println("🌊 Starting Ollama streaming verification...")

	// 1. Create the Ollama provider
	baseURL := "http://localhost:11434"
	provider := llm.NewOllamaProvider(baseURL)

	// 2. Configure model
	modelName := "mistral"
	fmt.Printf("🔌 Connecting to Ollama at %s using model '%s'...\n", baseURL, modelName)

	// 3. Create a message that requires a bit of generation
	msgs := []*content.Message{
		content.NewUserMessage("Tell me a short story about a robot learning to paint in exactly 5 sentences."),
	}

	// 4. Set timeout
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	fmt.Println("⏳ Requesting stream...")

	config := &llm.GenerationConfig{
		Model: modelName,
	}

	stream, err := provider.GenerateStream(ctx, msgs, config)
	if err != nil {
		log.Fatalf("❌ Error: %v", err)
	}

	fmt.Println("📥 Receiving tokens:")
	fmt.Println("──────────────────────────────────────────")

	start := time.Now()
	var fullText string
	tokenCount := 0

	for chunk := range stream {
		if chunk.Error != nil {
			fmt.Printf("\n❌ Stream error: %v\n", chunk.Error)
			break
		}

		fmt.Print(chunk.Delta)
		fullText += chunk.Delta
		tokenCount++

		if chunk.FinishReason != "" {
			fmt.Printf("\n\n🏁 Finish reason: %s", chunk.FinishReason)
			if chunk.Usage != nil {
				fmt.Printf(" (Usage: %d input, %d output tokens)", chunk.Usage.InputTokens, chunk.Usage.OutputTokens)
			}
		}
	}

	duration := time.Since(start)
	fmt.Println("\n──────────────────────────────────────────")
	fmt.Printf("✅ Stream complete! (took %s, ~%d chunks)\n", duration, tokenCount)
}
