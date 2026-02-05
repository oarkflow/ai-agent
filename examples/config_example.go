package main

/*
This example demonstrates how to use the AI Agent framework with
JSON configuration files for all settings including providers,
models, prompts, tools, and domains.

Configuration files should be in ./config/:
  - config.json   - Main configuration
  - prompts.json  - Prompt templates
  - tools.json    - Tool definitions
  - domains.json  - Domain knowledge
*/

import (
	"context"
	"flag"
	"fmt"
	"log"
	"path/filepath"

	"github.com/oarkflow/ai-agent/pkg/agent"
	"github.com/oarkflow/ai-agent/pkg/config"
	"github.com/oarkflow/ai-agent/pkg/content"
	"github.com/oarkflow/ai-agent/pkg/llm"
	"github.com/oarkflow/ai-agent/pkg/memory"
	"github.com/oarkflow/ai-agent/pkg/processor"
	"github.com/oarkflow/ai-agent/pkg/prompt"
	"github.com/oarkflow/ai-agent/pkg/tools"
	"github.com/oarkflow/ai-agent/pkg/training"
)

var configPath = flag.String("config", "./config", "Path to configuration directory")

func main() {
	flag.Parse()

	ctx := context.Background()

	// Load configuration from JSON
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("    Multimodal AI Agent - JSON Configuration Example       ")
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println()

	absPath, _ := filepath.Abs(*configPath)
	fmt.Printf("Loading configuration from: %s\n\n", absPath)

	// 1. Load Configuration
	loader := config.NewConfigLoader(absPath)
	cfg, err := loader.Load()
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}
	fmt.Printf("✓ Configuration loaded: %s v%s\n", cfg.Name, cfg.Version)

	// 2. Create Provider Registry from Config
	registry, err := llm.NewProviderRegistryFromConfig(cfg)
	if err != nil {
		log.Fatalf("Failed to create registry: %v", err)
	}
	enabledProviders := cfg.GetEnabledProviders()
	fmt.Printf("✓ Providers registered: %v\n", enabledProviders)

	// 3. Create Smart Router from Config
	router := llm.NewSmartRouter(registry)
	fmt.Println("✓ Smart router created")

	// 4. Setup Domain Training from Config
	var trainer *training.DomainTrainer
	if len(enabledProviders) > 0 {
		if p, ok := registry.GetProvider(llm.ProviderType(enabledProviders[0])); ok {
			vectorStore := training.NewInMemoryVectorStore()
			trainer, _ = training.NewDomainTrainerFromConfig(cfg, p, vectorStore)
			if trainer != nil && cfg.Domains != nil {
				fmt.Printf("✓ Domains loaded: %d\n", len(cfg.Domains.Domains))
			}
		}
	}

	// 5. Create Memory from Config
	conversationMemory := memory.NewConversationMemoryFromConfig(cfg)
	fmt.Printf("✓ Memory configured: strategy=%s, max_messages=%d\n",
		cfg.Memory.Strategy, cfg.Memory.MaxMessages)

	// 6. Create Agent from Config
	agentConfig := agent.DefaultAgentConfig()
	agentConfig.EnableRAG = cfg.Features.EnableRAG
	agentConfig.EnableTools = cfg.Features.EnableTools
	agentConfig.Temperature = cfg.Generation.Defaults.Temperature
	agentConfig.MaxTokens = cfg.Generation.Defaults.MaxTokens

	agentOpts := []agent.AgentOption{
		agent.WithSystemPrompt(cfg.Agent.SystemPrompt),
		agent.WithConfig(agentConfig),
	}
	if trainer != nil {
		agentOpts = append(agentOpts, agent.WithDomainTrainer(trainer))
	}

	aiAgent := agent.NewMultimodalAgent(cfg.Agent.Name, registry, agentOpts...)
	fmt.Printf("✓ Agent created: %s\n", cfg.Agent.Name)

	// 7. Load Prompts from Config
	promptLib := prompt.NewPromptLibraryFromConfig(cfg)
	if cfg.Prompts != nil {
		fmt.Printf("✓ Prompt templates loaded: %d\n", len(cfg.Prompts.Prompts))
	}

	// 8. Load Tools from Config
	toolRegistry, _ := tools.NewToolRegistryFromConfig(cfg)
	if cfg.Tools != nil {
		enabledTools := 0
		for _, t := range cfg.Tools.Tools {
			if t.Enabled {
				enabledTools++
			}
		}
		fmt.Printf("✓ Tools loaded: %d enabled\n", enabledTools)
	}

	// 9. Create Analyzer
	analyzer := processor.NewContentAnalyzer(router)

	// Demonstrate usage statistics from config
	usageTracker := agent.NewUsageTracker()
	aiAgent.OnEvent(agent.EventAfterGenerate, func(ctx context.Context, data any) error {
		if resp, ok := data.(*llm.GenerationResponse); ok {
			usageTracker.Track(resp)
		}
		return nil
	})

	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("                    Running Examples                        ")
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println()

	// Example 1: Basic Chat with Config-based Agent
	fmt.Println("📝 Example 1: Basic Chat")
	fmt.Println("────────────────────────")
	resp, err := aiAgent.Chat(ctx, "Explain neural networks in simple terms.")
	if err != nil {
		fmt.Printf("Error: %v\n\n", err)
	} else {
		fmt.Printf("Model: %s\n", resp.Model)
		fmt.Printf("Response: %s\n\n", truncate(resp.Message.GetText(), 400))
	}

	// Example 2: Using Prompt Template from Config
	fmt.Println("📋 Example 2: Prompt Template from Config")
	fmt.Println("──────────────────────────────────────────")
	if template, ok := promptLib.Get("summarize"); ok {
		rendered, err := template.Render(map[string]any{
			"content": "Artificial intelligence is transforming how we work and live. Machine learning algorithms can now recognize images, understand speech, and generate human-like text. These advances are being applied in healthcare, finance, transportation, and many other industries.",
			"style":   "bullet-points",
		})
		if err == nil {
			fmt.Printf("Template: %s\n", template.Name)
			fmt.Printf("Rendered:\n%s\n\n", truncate(rendered, 400))
		}
	} else {
		fmt.Println("Template 'summarize' not found in config\n")
	}

	// Example 3: Domain Query with Config-loaded Domain
	fmt.Println("🎯 Example 3: Domain Query (from config)")
	fmt.Println("─────────────────────────────────────────")
	aiAgent.SetDomain("software")
	resp, err = aiAgent.Chat(ctx, "What are the key principles of TDD?")
	if err != nil {
		fmt.Printf("Error: %v\n\n", err)
	} else {
		fmt.Printf("Domain: software\n")
		fmt.Printf("Model: %s\n", resp.Model)
		fmt.Printf("Response: %s\n\n", truncate(resp.Message.GetText(), 400))
	}

	// Example 4: Code Analysis
	fmt.Println("💻 Example 4: Code Analysis")
	fmt.Println("────────────────────────────")
	code := `func fibonacci(n int) int {
    if n <= 1 {
        return n
    }
    return fibonacci(n-1) + fibonacci(n-2)
}`
	result, err := analyzer.AnalyzeCode(ctx, code, "go", processor.CodeAnalysisOptimize)
	if err != nil {
		fmt.Printf("Error: %v\n\n", err)
	} else {
		fmt.Printf("Model: %s\n", result.Model)
		fmt.Printf("Analysis: %s\n\n", truncate(result.Analysis, 400))
	}

	// Example 5: Multimodal Message
	fmt.Println("🎨 Example 5: Multimodal Message")
	fmt.Println("─────────────────────────────────")
	msg := content.NewUserMessage("Analyze this sorting algorithm:")
	msg.AddCode(`def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr`, "python")

	resp, err = aiAgent.Send(ctx, msg)
	if err != nil {
		fmt.Printf("Error: %v\n\n", err)
	} else {
		fmt.Printf("Model: %s\n", resp.Model)
		fmt.Printf("Response: %s\n\n", truncate(resp.Message.GetText(), 400))
	}

	// Example 6: Show Configuration
	fmt.Println("⚙️  Example 6: Configuration Overview")
	fmt.Println("──────────────────────────────────────")
	fmt.Println("Routing Configuration:")
	for cap, model := range cfg.Routing.DefaultModels {
		fmt.Printf("  %s → %s\n", cap, model)
	}
	fmt.Println()

	// Usage Statistics
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("                    Usage Statistics                        ")
	fmt.Println("═══════════════════════════════════════════════════════════")
	stats := usageTracker.GetStats()
	fmt.Printf("Requests: %v\n", stats["request_count"])
	fmt.Printf("Input Tokens: %v\n", stats["total_input_tokens"])
	fmt.Printf("Output Tokens: %v\n", stats["total_output_tokens"])
	fmt.Printf("Estimated Cost: $%.4f\n", stats["total_cost"])

	// Show that memory and tools are configured
	_ = conversationMemory
	_ = toolRegistry

	fmt.Println("\n✅ All examples completed successfully!")
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
