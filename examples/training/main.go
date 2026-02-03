package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/sujit/ai-agent/pkg/domains"
	"github.com/sujit/ai-agent/pkg/llm"
	"github.com/sujit/ai-agent/pkg/storage"
)

// This file demonstrates how to use the domain training system.
// Run with: go run examples/training/main.go

func main() {
	ctx := context.Background()

	// Initialize storage
	dataDir := "./data"
	store, err := storage.NewStorage(dataDir)
	if err != nil {
		log.Fatalf("Failed to initialize storage: %v", err)
	}
	fmt.Println("✓ Storage initialized at:", dataDir)

	// Initialize provider (requires API key)
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is required")
	}

	provider := llm.NewOpenAIMultimodalProvider(apiKey)
	fmt.Println("✓ LLM provider initialized")

	// Create domain registry
	registry := domains.NewDomainRegistry(provider, store)

	// Register built-in domains
	registry.RegisterBuiltinDomains()
	fmt.Println("✓ Built-in domains registered")

	// Load training examples from files
	if err := registry.LoadExamplesFromFiles(dataDir); err != nil {
		log.Printf("Warning: Failed to load examples: %v", err)
	}
	fmt.Println("✓ Training examples loaded")

	// List available domains
	fmt.Println("\n📚 Available Domains:")
	for _, info := range registry.ListDomains() {
		fmt.Printf("  - %s (%s)\n", info.Name, info.ID)
	}

	// Example: Generate a workflow
	fmt.Println("\n🔧 Generating a workflow...")
	workflowPrompt := "Create a workflow that sends a Slack notification when a new GitHub issue is created"
	result, err := registry.Generate(ctx, "workflow", workflowPrompt, "")
	if err != nil {
		log.Printf("Workflow generation failed: %v", err)
	} else {
		truncated := result
		if len(result) > 500 {
			truncated = result[:500]
		}
		fmt.Println("Generated workflow:")
		fmt.Println(truncated)
	}

	// Example: Generate a ReactFlow graph
	fmt.Println("\n🎨 Generating a ReactFlow graph...")
	reactflowPrompt := "Create a user onboarding flow with 5 steps"
	result, err = registry.Generate(ctx, "reactflow", reactflowPrompt, "")
	if err != nil {
		log.Printf("ReactFlow generation failed: %v", err)
	} else {
		truncated := result
		if len(result) > 500 {
			truncated = result[:500]
		}
		fmt.Println("Generated ReactFlow graph:")
		fmt.Println(truncated)
	}

	// Example: Analyze clinical note (healthcare domain)
	fmt.Println("\n🏥 Analyzing clinical note...")
	healthcarePrompt := "Patient presents with chest pain and shortness of breath. ECG shows ST elevation. Troponin elevated."
	result, err = registry.Generate(ctx, "healthcare", healthcarePrompt, "medical_coding")
	if err != nil {
		log.Printf("Healthcare analysis failed: %v", err)
	} else {
		truncated := result
		if len(result) > 500 {
			truncated = result[:500]
		}
		fmt.Println("Generated medical coding:")
		fmt.Println(truncated)
	}

	// Example: Add training feedback
	fmt.Println("\n📝 Adding training feedback...")
	err = registry.TrainFromFeedback(
		"workflow",
		"Create a simple email notification workflow",
		`{"id": "email-notify", "name": "Email Notification", "nodes": [...]}`,
		"webhook_trigger",
		4.5, // Rating out of 5
	)
	if err != nil {
		log.Printf("Failed to add training feedback: %v", err)
	} else {
		fmt.Println("✓ Training feedback added")
	}

	// Show stats
	fmt.Println("\n📊 Domain Statistics:")
	stats := registry.GetStats()
	for domain, domainStats := range stats {
		fmt.Printf("  %s: %d examples\n", domain, domainStats["examples"])
	}

	fmt.Println("\n✅ Domain training system demonstration complete!")
}
