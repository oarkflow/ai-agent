/*
AI Agent - Comprehensive Prompt Engineering Framework

This framework provides:
1. Multimodal Content Handling (text, images, audio, video, documents, code)
2. Intelligent Model Routing (auto-selects best model based on content)
3. Multiple Provider Support (OpenAI, Anthropic, Google, DeepSeek, Mistral, xAI)
4. Domain-Specific Training with RAG
5. Advanced Prompt Engineering Templates
6. Tool/Function Calling Support
7. Conversation Memory with Summarization
8. JSON-Based Configuration System

Usage:

	Set environment variables for providers:
	  - OPENAI_API_KEY
	  - ANTHROPIC_API_KEY
	  - GOOGLE_API_KEY
	  - DEEPSEEK_API_KEY
	  - MISTRAL_API_KEY
	  - XAI_API_KEY

	Configuration files (in ./config/):
	  - config.json      - Main configuration (providers, routing, generation, etc.)
	  - prompts.json     - Prompt templates
	  - tools.json       - Tool definitions
	  - domains.json     - Domain knowledge bases

	Run:
	  go run cmd/agent/main.go
	  go run cmd/agent/main.go -config /path/to/config
*/
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"github.com/sujit/ai-agent/pkg/agent"
	"github.com/sujit/ai-agent/pkg/config"
	"github.com/sujit/ai-agent/pkg/content"
	"github.com/sujit/ai-agent/pkg/llm"
	"github.com/sujit/ai-agent/pkg/memory"
	"github.com/sujit/ai-agent/pkg/processor"
	"github.com/sujit/ai-agent/pkg/prompt"
	"github.com/sujit/ai-agent/pkg/tools"
	"github.com/sujit/ai-agent/pkg/training"
)

var (
	configPath = flag.String("config", "./config", "Path to configuration directory")
	domainID   = flag.String("domain", "", "Domain ID to use for the request")
	userPrompt = flag.String("prompt", "", "Prompt to execute (if provided, demos are skipped)")
)

func main() {
	flag.Parse()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigChan
		fmt.Println("\nShutting down...")
		cancel()
	}()

	// Initialize the AI agent from JSON configuration
	app, err := initializeFromConfig(*configPath)
	if err != nil {
		log.Fatalf("Failed to initialize agent: %v", err)
	}

	// If prompt is provided, run just that request
	if *userPrompt != "" {
		runSingleRequest(ctx, app, *userPrompt, *domainID)
		return
	}

	// Otherwise run demo scenarios
	runDemos(ctx, app)
}

// runSingleRequest executes a single prompt, optionally using a specific domain.
func runSingleRequest(ctx context.Context, app *AIAgentApp, promptStr string, dID string) {
	fmt.Printf("🚀 Executing request...\n")

	if dID != "" {
		fmt.Printf("🎯 Using domain: %s\n", dID)
		app.Agent.SetDomain(dID)

		// If domain has a preferred model, use it
		if app.Config.Domains != nil {
			if d, ok := app.Config.Domains.Domains[dID]; ok && d.PreferredModel != "" {
				fmt.Printf("🤖 Using champion model for domain: %s\n", d.PreferredModel)
				app.Agent.Config.DefaultModel = d.PreferredModel

				// Apply optimal hyperparameters if available
				app.Agent.Config.Temperature = d.Temperature
				app.Agent.Config.TopP = d.TopP
				app.Agent.Config.MaxTokens = d.MaxTokens
				fmt.Printf("⚙️  Applied optimal hyperparameters: Temp=%.1f, TopP=%.1f, MaxTokens=%d\n", d.Temperature, d.TopP, d.MaxTokens)
			}
		}
	}

	resp, err := app.Agent.Chat(ctx, promptStr)
	if err != nil {
		log.Fatalf("Error: %v", err)
	}

	fmt.Printf("\n--- Response ---\n%s\n----------------\n", resp.Message.GetText())
	fmt.Printf("Model: %s\n", resp.Model)
}

// AIAgentApp is the main application container.
type AIAgentApp struct {
	Config       *config.Config
	Registry     *llm.ProviderRegistry
	Router       *llm.SmartRouter
	Agent        *agent.MultimodalAgent
	Trainer      *training.DomainTrainer
	Memory       *memory.ConversationMemory
	Analyzer     *processor.ContentAnalyzer
	PromptLib    *prompt.PromptLibrary
	ToolRegistry *tools.ToolRegistry
}

// initializeFromConfig loads all configuration from JSON files and initializes the application.
func initializeFromConfig(configDir string) (*AIAgentApp, error) {
	app := &AIAgentApp{}

	// Resolve absolute path
	absPath, err := filepath.Abs(configDir)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve config path: %w", err)
	}

	// 1. Load Configuration from JSON
	fmt.Println("🔧 Loading configuration from JSON...")
	fmt.Printf("   Config directory: %s\n", absPath)

	loader := config.NewConfigLoader(absPath)
	cfg, err := loader.Load()
	if err != nil {
		return nil, fmt.Errorf("failed to load configuration: %w", err)
	}
	app.Config = cfg

	// Validate configuration
	if errors := cfg.Validate(); len(errors) > 0 {
		fmt.Println("   ⚠️  Configuration warnings:")
		for _, e := range errors {
			fmt.Printf("      - %s\n", e)
		}
	}

	fmt.Printf("   ✓ Loaded config version: %s\n", cfg.Version)
	fmt.Printf("   ✓ Name: %s\n", cfg.Name)

	// 2. Initialize Provider Registry from Config
	fmt.Println("\n🔌 Initializing providers from configuration...")
	registry, err := llm.NewProviderRegistryFromConfig(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create provider registry: %w", err)
	}
	app.Registry = registry

	enabledProviders := cfg.GetEnabledProviders()
	if len(enabledProviders) == 0 {
		fmt.Println("   ⚠️  No providers configured. Set API key environment variables.")
	} else {
		for _, p := range enabledProviders {
			providerCfg, _ := cfg.GetProvider(p)
			modelCount := len(providerCfg.Models)
			fmt.Printf("   ✓ %s (%d models)\n", p, modelCount)
		}
	}

	// 3. Create Smart Router
	fmt.Println("\n🚦 Creating smart router...")
	routingCfg := llm.RoutingConfigFromConfig(cfg)
	app.Router = llm.NewSmartRouter(registry)
	fmt.Printf("   ✓ Default text model: %s\n", routingCfg.DefaultTextModel)
	fmt.Printf("   ✓ Default code model: %s\n", routingCfg.DefaultCodeModel)
	fmt.Printf("   ✓ Default vision model: %s\n", routingCfg.DefaultVisionModel)

	// 4. Setup Domain Training (RAG) from Config
	fmt.Println("\n📚 Setting up domain knowledge from configuration...")
	if cfg.Domains != nil && len(cfg.Domains.Domains) > 0 {
		// Find an embedding provider
		var embeddingProvider llm.MultimodalProvider
		for _, providerName := range []string{"openai", "mistral", "google"} {
			if p, ok := registry.GetProvider(llm.ProviderType(providerName)); ok {
				embeddingProvider = p
				break
			}
		}

		if embeddingProvider != nil {
			vectorStore := training.NewInMemoryVectorStore()
			trainer, err := training.NewDomainTrainerFromConfig(cfg, embeddingProvider, vectorStore)
			if err != nil {
				fmt.Printf("   ⚠️  Failed to setup domain training: %v\n", err)
			} else {
				app.Trainer = trainer
				for domainID, domainCfg := range cfg.Domains.Domains {
					if domainCfg.Enabled {
						fmt.Printf("   ✓ Domain: %s (%d terms, %d guidelines)\n",
							domainID, len(domainCfg.Terminology), len(domainCfg.Guidelines))
					}
				}
			}
		} else {
			fmt.Println("   ⚠️  No embedding provider available for RAG")
		}
	}

	// 5. Initialize Memory from Config
	fmt.Println("\n🧠 Initializing conversation memory...")
	app.Memory = memory.NewConversationMemoryFromConfig(cfg)
	fmt.Printf("   ✓ Strategy: %s\n", cfg.Memory.Strategy)
	fmt.Printf("   ✓ Max messages: %d\n", cfg.Memory.MaxMessages)
	fmt.Printf("   ✓ Max tokens: %d\n", cfg.Memory.MaxTokens)

	// 6. Create Multimodal Agent
	fmt.Println("\n🤖 Creating multimodal agent...")
	agentConfig := agent.DefaultAgentConfig()
	agentConfig.EnableRAG = cfg.Features.EnableRAG
	agentConfig.EnableTools = cfg.Features.EnableTools
	agentConfig.AutoPreprocess = cfg.Features.AutoPreprocess
	agentConfig.Temperature = cfg.Generation.Defaults.Temperature
	agentConfig.MaxTokens = cfg.Generation.Defaults.MaxTokens

	agentOpts := []agent.AgentOption{
		agent.WithSystemPrompt(cfg.Agent.SystemPrompt),
		agent.WithConfig(agentConfig),
	}

	if app.Trainer != nil {
		agentOpts = append(agentOpts, agent.WithDomainTrainer(app.Trainer))
	}

	app.Agent = agent.NewMultimodalAgent(cfg.Agent.Name, registry, agentOpts...)
	fmt.Printf("   ✓ Agent name: %s\n", cfg.Agent.Name)

	// 7. Initialize Content Analyzer
	fmt.Println("\n🔍 Initializing content analyzer...")
	app.Analyzer = processor.NewContentAnalyzer(app.Router)

	// 8. Load Prompt Library from Config
	fmt.Println("\n📝 Loading prompt templates from configuration...")
	app.PromptLib = prompt.NewPromptLibraryFromConfig(cfg)
	if cfg.Prompts != nil {
		fmt.Printf("   ✓ Loaded %d prompt templates\n", len(cfg.Prompts.Prompts))
		for id := range cfg.Prompts.Prompts {
			fmt.Printf("      - %s\n", id)
		}
	}

	// 9. Load Tool Registry from Config
	fmt.Println("\n🛠️  Loading tools from configuration...")
	toolRegistry, err := tools.NewToolRegistryFromConfig(cfg)
	if err != nil {
		fmt.Printf("   ⚠️  Failed to load tools: %v\n", err)
		toolRegistry = tools.NewToolRegistry()
	}
	app.ToolRegistry = toolRegistry
	if cfg.Tools != nil {
		enabledCount := 0
		for _, tool := range cfg.Tools.Tools {
			if tool.Enabled {
				enabledCount++
				fmt.Printf("   ✓ %s\n", tool.Name)
			}
		}
		fmt.Printf("   Total: %d tools enabled\n", enabledCount)
	}

	fmt.Println("\n✅ AI Agent initialized successfully from configuration!")
	fmt.Println()

	return app, nil
}

func runDemos(ctx context.Context, app *AIAgentApp) {
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("                    AI AGENT DEMONSTRATIONS                 ")
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println()

	// Demo 1: Basic Chat
	demo1_BasicChat(ctx, app)

	// Demo 2: Code Analysis
	demo2_CodeAnalysis(ctx, app)

	// Demo 3: Prompt Templates from Config
	demo3_PromptTemplates(ctx, app)

	// Demo 4: Domain-Specific Query
	demo4_DomainQuery(ctx, app)

	// Demo 5: Configuration Info
	demo5_ConfigInfo(ctx, app)

	// Demo 6: Multimodal Message
	demo6_MultimodalMessage(ctx, app)

	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("                    DEMOS COMPLETE                          ")
	fmt.Println("═══════════════════════════════════════════════════════════")
}

func demo1_BasicChat(ctx context.Context, app *AIAgentApp) {
	fmt.Println("📝 Demo 1: Basic Chat")
	fmt.Println("─────────────────────")

	resp, err := app.Agent.Chat(ctx, "What are the key principles of clean code architecture?")
	if err != nil {
		log.Printf("Error: %v", err)
		fmt.Println()
		return
	}

	fmt.Printf("Model: %s\n", resp.Model)
	fmt.Printf("Response:\n%s\n\n", truncateText(resp.Message.GetText(), 600))
}

func demo2_CodeAnalysis(ctx context.Context, app *AIAgentApp) {
	fmt.Println("💻 Demo 2: Code Analysis")
	fmt.Println("────────────────────────")

	code := `package main
import (
    "database/sql"
    "fmt"
    "log"
)
func GetUser(db *sql.DB, id int) (string, error) {
    var name string
    query := fmt.Sprintf("SELECT name FROM users WHERE id = %d", id) // SQL injection!
    err := db.QueryRow(query).Scan(&name)
    if err != nil {
        log.Printf("Error: %v", err)
        return "", err
    }
    return name, nil
}`

	result, err := app.Analyzer.AnalyzeCode(ctx, code, "go", processor.CodeAnalysisReview)
	if err != nil {
		log.Printf("Error: %v", err)
		fmt.Println()
		return
	}

	fmt.Printf("Model: %s\n", result.Model)
	fmt.Printf("Analysis:\n%s\n\n", truncateText(result.Analysis, 800))
}

func demo3_PromptTemplates(ctx context.Context, app *AIAgentApp) {
	fmt.Println("📋 Demo 3: Prompt Templates (from config)")
	fmt.Println("──────────────────────────────────────────")

	// Get code generation template from config
	template, ok := app.PromptLib.Get("generate-code")
	if !ok {
		fmt.Println("Template 'generate-code' not found in configuration")
		fmt.Println()
		return
	}

	// Render the template
	rendered, err := template.Render(map[string]any{
		"language":     "Go",
		"requirements": "Create a function that validates email addresses using regex",
		"constraints":  []string{"Must handle edge cases", "Should return detailed error messages"},
	})
	if err != nil {
		log.Printf("Error: %v", err)
		fmt.Println()
		return
	}

	fmt.Printf("Template: %s (v%s)\n", template.Name, template.Version)
	fmt.Printf("Category: %s\n", template.Category)
	if template.Config != nil {
		fmt.Printf("Preferred Model: %s\n", template.Config.PreferredModel)
	}
	fmt.Printf("Rendered Prompt:\n%s\n\n", truncateText(rendered, 500))
}

func demo4_DomainQuery(ctx context.Context, app *AIAgentApp) {
	fmt.Println("🎯 Demo 4: Domain-Specific Query")
	fmt.Println("─────────────────────────────────")

	// Set domain context from config
	app.Agent.SetDomain("software")

	resp, err := app.Agent.Chat(ctx, "What's the best approach for implementing CI/CD in a microservices architecture?")
	if err != nil {
		log.Printf("Error: %v", err)
		fmt.Println()
		return
	}

	fmt.Printf("Domain: software (from config)\n")
	fmt.Printf("Model: %s\n", resp.Model)
	fmt.Printf("Response:\n%s\n\n", truncateText(resp.Message.GetText(), 600))
}

func demo5_ConfigInfo(ctx context.Context, app *AIAgentApp) {
	fmt.Println("⚙️  Demo 5: Configuration Info")
	fmt.Println("──────────────────────────────")

	cfg := app.Config

	fmt.Println("Loaded Configuration:")
	fmt.Printf("  Version: %s\n", cfg.Version)
	fmt.Printf("  Name: %s\n", cfg.Name)

	fmt.Println("\nProviders:")
	for name, provider := range cfg.Providers {
		status := "disabled"
		if provider.Enabled {
			if apiKey := cfg.GetAPIKey(name); apiKey != "" {
				status = "enabled (API key set)"
			} else {
				status = "enabled (no API key)"
			}
		}
		fmt.Printf("  - %s: %s (%d models)\n", name, status, len(provider.Models))
	}

	fmt.Println("\nRouting Defaults:")
	for cap, model := range cfg.Routing.DefaultModels {
		fmt.Printf("  - %s: %s\n", cap, model)
	}

	fmt.Println("\nGeneration Defaults:")
	fmt.Printf("  - Temperature: %.2f\n", cfg.Generation.Defaults.Temperature)
	fmt.Printf("  - Max Tokens: %d\n", cfg.Generation.Defaults.MaxTokens)
	fmt.Printf("  - Streaming: %v\n", cfg.Generation.Defaults.StreamingEnabled)

	fmt.Println("\nFeatures:")
	fmt.Printf("  - RAG: %v\n", cfg.Features.EnableRAG)
	fmt.Printf("  - Tools: %v\n", cfg.Features.EnableTools)
	fmt.Printf("  - Streaming: %v\n", cfg.Features.EnableStreaming)
	fmt.Printf("  - Caching: %v\n", cfg.Features.EnableCaching)
	fmt.Printf("  - Cost Tracking: %v\n", cfg.Features.EnableCostTracking)

	fmt.Println()
}

func demo6_MultimodalMessage(ctx context.Context, app *AIAgentApp) {
	fmt.Println("🎨 Demo 6: Multimodal Message")
	fmt.Println("─────────────────────────────")

	// Create a multimodal message with text and code
	msg := content.NewUserMessage("Analyze this code and explain what it does:")
	msg.AddCode(`func quickSort(arr []int) []int {
    if len(arr) < 2 {
        return arr
    }
    pivot := arr[len(arr)/2]
    var left, right, equal []int
    for _, v := range arr {
        switch {
        case v < pivot:
            left = append(left, v)
        case v > pivot:
            right = append(right, v)
        default:
            equal = append(equal, v)
        }
    }
    result := quickSort(left)
    result = append(result, equal...)
    result = append(result, quickSort(right)...)
    return result
}`, "go")

	resp, err := app.Agent.Send(ctx, msg)
	if err != nil {
		log.Printf("Error: %v", err)
		fmt.Println()
		return
	}

	fmt.Printf("Model: %s\n", resp.Model)
	fmt.Printf("Response:\n%s\n\n", truncateText(resp.Message.GetText(), 600))
}

func truncateText(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
