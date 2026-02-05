package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"path/filepath"
	"strings"

	"github.com/oarkflow/ai-agent/pkg/agent"
	"github.com/oarkflow/ai-agent/pkg/config"
	"github.com/oarkflow/ai-agent/pkg/content"
	"github.com/oarkflow/ai-agent/pkg/llm"
	"github.com/oarkflow/ai-agent/pkg/memory"
	"github.com/oarkflow/ai-agent/pkg/prompt"
	"github.com/oarkflow/ai-agent/pkg/storage"
	"github.com/oarkflow/ai-agent/pkg/training"
)

var (
	memoryPath = "memory/test_session.json"
)

func main() {
	step := flag.Int("step", 0, "Step to run (1: CPT, 2: DX, 3: E&M)")
	flag.Parse()

	if *step == 0 {
		fmt.Println("Usage: go run main.go --step <1|2|3>")
		return
	}

	ctx := context.Background()
	configDir := "./config"
	absPath, _ := filepath.Abs(configDir)

	// 1. Initialize Config & Agent
	loader := config.NewConfigLoader(absPath)
	cfg, err := loader.Load()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// 1. Initialize Registry & Provider
	registry, _ := llm.NewProviderRegistryFromConfig(cfg)
	ollamaProvider, ok := registry.GetProvider(llm.ProviderOllama)
	if !ok {
		log.Fatalf("Ollama provider not found")
	}

	// 2. Initialize Domain Trainer from Config
	// This will load "medical-coding" and other domains from domains.json
	trainer, err := training.NewDomainTrainerFromConfig(cfg, ollamaProvider.(llm.MultimodalProvider), nil)
	if err != nil {
		log.Fatalf("Failed to initialize domain trainer: %v", err)
	}

	domainID := "medical-coding"
	if _, err := trainer.GetDomain(domainID); err != nil {
		log.Fatalf("Domain %s not found. Check domains.json", domainID)
	}

	// 3. Initialize Agent with Domain
	multimodalAgent := agent.NewMultimodalAgent("MemoryTester", registry,
		agent.WithDomainTrainer(trainer),
		agent.WithConfig(&agent.AgentConfig{
			DefaultModel:    "mistral",
			EnableStreaming: true,
			AutoPreprocess:  true,
			DomainID:        domainID,
			EnableRAG:       true, // Required to trigger BuildSystemPrompt
		}),
	)

	_ = prompt.NewPromptLibraryFromConfig(cfg)

	// 4. Initialize Storage
	store, err := storage.NewStorage("./data_test")
	if err != nil {
		log.Fatalf("Failed to init storage: %v", err)
	}

	// 5. Initialize Smart System Memory
	memConfig := memory.DefaultMemoryConfig()
	memConfig.Strategy = memory.StrategySummary
	memConfig.MaxMessages = 10
	mem := memory.NewConversationMemory(memConfig)

	if ok {
		mem.SetSummaryProvider(ollamaProvider)
	} else {
		// (Fallback logic remains same)
	}

	// 4. Load Memory (except step 1 which starts fresh)
	if *step > 1 {
		fmt.Printf("📂 Loading memory from storage: %s...\n", memoryPath)
		if store.Exists(memoryPath) {
			var rawBytes json.RawMessage
			if err := store.LoadJSON(memoryPath, &rawBytes); err != nil {
				log.Fatalf("Failed to load memory: %v", err)
			}
			if err := mem.Import(rawBytes); err != nil {
				log.Fatalf("Failed to import memory: %v", err)
			}
			fmt.Printf("✅ Loaded %d messages into System Memory.\n", len(mem.Get()))
		} else {
			log.Fatalf("Memory file not found at %s. Please run Step 1 first.", memoryPath)
		}
	} else {
		// Clean start for step 1
		if store.Exists(memoryPath) {
			_ = store.Delete(memoryPath)
		}
	}

	// 5. Execute Step Logic
	switch *step {
	case 1:
		runStep1(ctx, multimodalAgent, mem)
	case 2:
		runStep2(ctx, multimodalAgent, mem)
	case 3:
		runStep3(ctx, multimodalAgent, mem)
	default:
		log.Fatalf("Invalid step: %d", *step)
	}

	// 6. Save Memory (except step 3 which is final, or maybe save step 3 too for completeness)
	// We save at every step to persist the state
	fmt.Printf("💾 Saving memory to storage: %s...\n", memoryPath)

	// Export from memory system (which may have run summarization)
	exportData, err := mem.Export()
	if err != nil {
		log.Fatalf("Failed to export memory: %v", err)
	}

	// Save as raw JSON. Since SaveJSON marshals, and Export returns JSON bytes,
	// we need to unmarshal to interface{} first or assume SaveJSON handles RawMessage.
	var persistData interface{}
	if err := json.Unmarshal(exportData, &persistData); err != nil {
		log.Fatalf("Failed to unmarshal export data: %v", err)
	}

	if err := store.SaveJSON(memoryPath, persistData); err != nil {
		log.Fatalf("Failed to save memory: %v", err)
	}
	fmt.Println("✅ Memory saved successfully.")
}

// Step 1: Get CPT Code
func runStep1(ctx context.Context, a *agent.MultimodalAgent, mem *memory.ConversationMemory) {
	fmt.Println("\n--- STEP 1: CPT Code Acquisition (Clean Prompt) ---")
	input := "What is the CPT code range for an established patient presenting for a standard office consultation?"
	fmt.Printf("User: %s\n", input)

	// Sync Mem -> Agent (System prompt is handled via MultimodalAgent config)
	a.Conversation.Messages = mem.Get()

	// Simple prompt - formatting enforced by SystemPrompt
	respStr, usage := streamChat(ctx, a, input)
	printUsage(usage, a, "mistral")

	// Sync Agent -> Mem
	mem.Add(content.NewUserMessage(input))
	mem.Add(content.NewAssistantMessage(respStr))

	fmt.Println("\n[End of Step 1]")
}

// Step 2: Get DX Code
func runStep2(ctx context.Context, a *agent.MultimodalAgent, mem *memory.ConversationMemory) {
	fmt.Println("\n--- STEP 2: Diagnosis (DX) Code Acquisition (Clean Prompt) ---")
	input := "The patient also has type 2 diabetes mellitus with high blood pressure and hypertension. What are the ICD-10 codes?"
	fmt.Printf("User: %s\n", input)

	// Sync Mem -> Agent
	a.Conversation.Messages = mem.Get()

	// Simple prompt
	respStr, usage := streamChat(ctx, a, input)
	printUsage(usage, a, "mistral")

	// Sync Agent -> Mem
	mem.Add(content.NewUserMessage(input))
	mem.Add(content.NewAssistantMessage(respStr))

	fmt.Println("\n[End of Step 2]")
}

// Step 3: E&M Visit (CoT + JSON)
func runStep3(ctx context.Context, a *agent.MultimodalAgent, mem *memory.ConversationMemory) {
	fmt.Println("\n--- STEP 3: E&M Visit Level (Clean Prompt) ---")

	// Sync Mem -> Agent
	a.Conversation.Messages = mem.Get()

	// Simple requirement - formatting and reasoning enforced by SystemPrompt
	input := "Based on all the information provided so far, what is the appropriate E&M visit level code? Provide full reasoning."
	fmt.Printf("User: %s\n", input)

	respStr, usage := streamChat(ctx, a, input)
	printUsage(usage, a, "mistral")
	_ = respStr

	fmt.Println("\n[End of Step 3]")
}

func streamChat(ctx context.Context, a *agent.MultimodalAgent, input string) (string, *llm.Usage) {
	msg := content.NewUserMessage(input)
	stream, err := a.Stream(ctx, msg)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return "", nil
	}

	var fullText strings.Builder
	var finalUsage *llm.Usage
	for chunk := range stream {
		if chunk.Error != nil {
			if errors.Is(chunk.Error, context.Canceled) {
				return fullText.String(), finalUsage
			}
			fmt.Printf("\n[Stream Error]: %v\n", chunk.Error)
			break
		}
		fmt.Print(chunk.Delta)
		fullText.WriteString(chunk.Delta)
		if chunk.Usage != nil {
			finalUsage = chunk.Usage
		}
	}
	fmt.Println()
	return fullText.String(), finalUsage
}

func printUsage(usage *llm.Usage, a *agent.MultimodalAgent, modelID string) {
	if usage == nil {
		return
	}

	// Calculate cost if model info is available
	var inputCost, outputCost, totalCost float64
	info, ok := a.Router.Registry.GetModel(modelID)
	if ok && info != nil && info.Info != nil {
		inputCost = (float64(usage.InputTokens) / 1000.0) * info.Info.InputCostPer1K
		outputCost = (float64(usage.OutputTokens) / 1000.0) * info.Info.OutputCostPer1K
		totalCost = inputCost + outputCost
	}

	fmt.Printf("\n--- Usage Metadata ---\n")
	fmt.Printf("Tokens: Input=%d, Output=%d, Total=%d\n", usage.InputTokens, usage.OutputTokens, usage.TotalTokens)
	if totalCost > 0 || (info != nil && info.Info != nil && (info.Info.InputCostPer1K > 0 || info.Info.OutputCostPer1K > 0)) {
		fmt.Printf("Cost:   Input=$%.6f, Output=$%.6f, Total=$%.6f\n", inputCost, outputCost, totalCost)
	} else {
		fmt.Printf("Cost:   $0.000000 (Local/Ollama)\n")
	}
	fmt.Printf("-----------------------\n")
}

func printHistory(a *agent.MultimodalAgent) {
	msgs := a.Conversation.Messages
	for i, m := range msgs {
		text := m.GetText()
		if len(text) > 60 {
			text = text[:57] + "..."
		}
		fmt.Printf("[%d] %s: %s\n", i, m.Role, text)
	}
}
