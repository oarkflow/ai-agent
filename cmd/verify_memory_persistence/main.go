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

	"github.com/sujit/ai-agent/pkg/agent"
	"github.com/sujit/ai-agent/pkg/config"
	"github.com/sujit/ai-agent/pkg/content"
	"github.com/sujit/ai-agent/pkg/llm"
	"github.com/sujit/ai-agent/pkg/memory"
	"github.com/sujit/ai-agent/pkg/prompt"
	"github.com/sujit/ai-agent/pkg/storage"
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

	registry, _ := llm.NewProviderRegistryFromConfig(cfg)
	multimodalAgent := agent.NewMultimodalAgent("MemoryTester", registry,
		agent.WithConfig(&agent.AgentConfig{
			DefaultModel:    "mistral",
			EnableStreaming: true,
			AutoPreprocess:  true,
		}),
	)

	_ = prompt.NewPromptLibraryFromConfig(cfg)

	// 2. Initialize Storage
	store, err := storage.NewStorage("./data_test")
	if err != nil {
		log.Fatalf("Failed to init storage: %v", err)
	}

	// 3. Initialize Smart System Memory
	// We use a small MaxMessages to trigger summarization quickly (e.g. key facts condensing)
	memConfig := memory.DefaultMemoryConfig()
	memConfig.Strategy = memory.StrategySummary
	memConfig.MaxMessages = 4 // Low limit to force "condensing" if conversation grows
	mem := memory.NewConversationMemory(memConfig)

	// Wire up the LLM provider for summarization capabilities
	ollamaProvider, ok := registry.GetProvider(llm.ProviderOllama)
	if ok {
		mem.SetSummaryProvider(ollamaProvider)
	} else {
		// Fallback: search for any available provider in the registry
		// This is just for robustness in the verification script
		log.Println("⚠️ Ollama provider not found in registry, search for fallback...")
		// Since we can't iterate private providers, let's try common ones
		found := false
		for _, pType := range []llm.ProviderType{llm.ProviderMistral, llm.ProviderOpenAI, llm.ProviderAnthropic} {
			if p, ok := registry.GetProvider(pType); ok {
				mem.SetSummaryProvider(p)
				found = true
				break
			}
		}
		if !found {
			log.Println("❌ No providers found for summarization fallback.")
		}
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
	fmt.Println("\n--- STEP 1: CPT Code Acquisition (Strict JSON) ---")
	input := "Established patient presenting for a standard office consultation."
	fmt.Printf("User: %s\n", input)

	// Sync Mem -> Agent
	a.Conversation.Messages = mem.Get()

	// Strict instruction for single JSON response
	query := input + ` Identification task: Return ONLY a single JSON object with the key "cpt_range". No pre-amble, no markdown.
Example Output: {"cpt_range": "99211-99215"}`

	respStr, usage := streamChat(ctx, a, query)
	printUsage(usage, a, "mistral")

	// Sync Agent -> Mem
	mem.Add(content.NewUserMessage(query))
	mem.Add(content.NewAssistantMessage(respStr))

	fmt.Println("\n[End of Step 1]")
}

// Step 2: Get DX Code
func runStep2(ctx context.Context, a *agent.MultimodalAgent, mem *memory.ConversationMemory) {
	fmt.Println("\n--- STEP 2: Diagnosis (DX) Code Acquisition (Strict JSON) ---")
	input := "Type 2 diabetes mellitus with high blood pressure and hypertension."
	fmt.Printf("User: %s\n", input)

	// Sync Mem -> Agent
	a.Conversation.Messages = mem.Get()

	// Strict instruction for single JSON response
	query := input + ` Identification task: Return ONLY a single JSON object with the key "dx_codes" (array). No pre-amble, no markdown.
Example Output: {"dx_codes": ["E11.9", "I10"]}`

	respStr, usage := streamChat(ctx, a, query)
	printUsage(usage, a, "mistral")

	// Sync Agent -> Mem
	mem.Add(content.NewUserMessage(query))
	mem.Add(content.NewAssistantMessage(respStr))

	fmt.Println("\n[End of Step 2]")
}

// Step 3: E&M Visit (CoT + JSON)
func runStep3(ctx context.Context, a *agent.MultimodalAgent, mem *memory.ConversationMemory) {
	fmt.Println("\n--- STEP 3: E&M Visit Level (Strict JSON) ---")

	// Sync Mem -> Agent
	a.Conversation.Messages = mem.Get()

	// Strict instruction for single JSON response
	input := `Based on the conversation context, identify the CPT range and DX codes, then determine the E&M visit level (99213/99214).
Output ONLY a single valid JSON object. No conversational text, no pre-amble, no markdown blocks.

Required JSON Structure:
{
  "cpt_range": "STRING",
  "dx_code": ["ARRAY", "OF", "STRINGS"],
  "em_code": "STRING",
  "reasoning": "STRING (include your Chain of Thought analysis here)"
}`
	fmt.Printf("User: %s\n", input)

	respStr, usage := streamChat(ctx, a, input)
	printUsage(usage, a, "mistral")
	_ = respStr // Capture but ignore for now in step 3 as it prints to console

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
