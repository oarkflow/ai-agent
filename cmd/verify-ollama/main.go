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
	fmt.Println("🚀 Starting minimal Ollama verification...")

	// 1. Create the Ollama provider
	baseURL := "http://localhost:11434"
	provider := llm.NewOllamaProvider(baseURL)

	// 2. Configure the specific model we want to test
	modelName := "mistral"
	fmt.Printf("🔌 Connecting to Ollama at %s using model '%s'...\n", baseURL, modelName)

	// 3. Create a simple message using valid types
	msgs := []*content.Message{
		content.NewUserMessage("Say 'Hello, Ollama is working!' and nothing else."),
	}

	// 4. Set a reasonable timeout (30s should be enough if model is pulled)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	fmt.Println("⏳ Sending request (this may take a few seconds if the model needs to load)...")
	start := time.Now()

	// 5. Generate with valid config
	config := &llm.GenerationConfig{
		Model: modelName,
	}

	resp, err := provider.Generate(ctx, msgs, config)
	if err != nil {
		log.Fatalf("❌ Error: %v", err)
	}

	duration := time.Since(start)
	fmt.Printf("✅ Success! (took %s)\n", duration)
	fmt.Println("──────────────────────────────────────────")
	if resp.Message != nil {
		fmt.Printf("Response: %s\n", resp.Message.GetText())
	} else {
		fmt.Println("Response: <empty message>")
	}
	fmt.Println("──────────────────────────────────────────")
}
