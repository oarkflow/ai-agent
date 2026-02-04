package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sujit/ai-agent/pkg/agent"
	"github.com/sujit/ai-agent/pkg/config"
	"github.com/sujit/ai-agent/pkg/llm"
	"github.com/sujit/ai-agent/pkg/training"
)

func main() {
	var (
		domainID    string
		datasetPath string
		outputPath  string
		configPath  string
	)

	flag.StringVar(&domainID, "domain", "medical-coding", "Domain ID to optimize")
	flag.StringVar(&datasetPath, "dataset", "data/examples/healthcare/examples.json", "Path to benchmark dataset")
	flag.StringVar(&outputPath, "output", "champion_report.md", "Path to save champion report")
	flag.StringVar(&configPath, "config", "config", "Path to config directory")
	flag.Parse()

	// 1. Load Config
	cfg := config.MustLoadConfig(configPath)

	// 2. Initialize Registry
	registry, err := llm.NewProviderRegistryFromConfig(cfg)
	if err != nil {
		log.Fatalf("Failed to initialize registry: %v", err)
	}

	// 3. Initialize Components
	evaluator := training.NewEvaluator(training.DefaultEvaluationConfig())
	// Use any model for embedding trainer initializion, it doesn't matter for model-only optimization
	trainer, _ := training.NewDomainTrainerFromConfig(cfg, nil, nil)
	optimizer := training.NewOptimizer(trainer, evaluator, registry)

	// 4. Load Dataset
	datasetData, err := os.ReadFile(datasetPath)
	if err != nil {
		log.Fatalf("Failed to read dataset: %v", err)
	}
	var dataPoints []training.DataPoint
	if err := json.Unmarshal(datasetData, &dataPoints); err != nil {
		log.Fatalf("Failed to parse dataset: %v", err)
	}

	fmt.Printf("--- Finding Champion Model for Domain: %s ---\n", domainID)
	fmt.Printf("Testing all models across registry (%d models total)\n\n", len(registry.ListModels()))

	// 5. Run AutoTuneModel
	predictFnFactory := func(modelID string, hp *training.Hyperparameters) func(string) (training.Prediction, error) {
		mAgent := agent.NewMultimodalAgent("ChampionFinder", registry,
			agent.WithDomainTrainer(trainer),
			agent.WithConfig(&agent.AgentConfig{
				DefaultModel: modelID,
				DomainID:     domainID,
				EnableRAG:    true,
				Temperature:  hp.Temperature,
				MaxTokens:    hp.MaxTokens,
			}),
		)

		return func(input string) (training.Prediction, error) {
			ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
			defer cancel()
			resp, err := mAgent.Chat(ctx, input)
			if err != nil {
				return training.Prediction{}, err
			}
			return training.Prediction{
				Input:     input,
				Predicted: resp.Message.GetText(),
			}, nil
		}
	}

	optResult, err := optimizer.AutoTuneModel(context.Background(), domainID, dataPoints, predictFnFactory)
	if err != nil {
		log.Fatalf("Optimization failed: %v", err)
	}

	fmt.Printf("\n🏆 Champion Found: %s\n", optResult.BestModel)

	// 6. Update Configuration
	domainsFile := filepath.Join(configPath, "domains.json")
	domainsData, _ := os.ReadFile(domainsFile)
	var domainsConfig map[string]any
	json.Unmarshal(domainsData, &domainsConfig)

	if domains, ok := domainsConfig["domains"].(map[string]any); ok {
		if domain, ok := domains[domainID].(map[string]any); ok {
			domain["preferred_model"] = optResult.BestModel
			fmt.Printf("✅ Updated domains.json: preferred_model set to %s\n", optResult.BestModel)
		}
	}

	newDomainsData, _ := json.MarshalIndent(domainsConfig, "", "    ")
	os.WriteFile(domainsFile, newDomainsData, 0644)

	// 7. Generate Report
	var sb strings.Builder
	sb.WriteString("# DDMI Champion Model Report\n\n")
	sb.WriteString(fmt.Sprintf("- **Domain**: %s\n", domainID))
	sb.WriteString(fmt.Sprintf("- **Champion**: **%s**\n", optResult.BestModel))
	sb.WriteString(fmt.Sprintf("- **Date**: %s\n\n", time.Now().Format(time.RFC822)))

	sb.WriteString("## Model Leaderboard\n\n")
	sb.WriteString("| Model | Accuracy | Avg Latency (ms) | Champion Score |\n")
	sb.WriteString("|---|---|---|---|\n")
	for _, o := range optResult.ModelOutcomes {
		sb.WriteString(fmt.Sprintf("| %s | %.2f%% | %.2f | %.4f |\n",
			o.ModelID, o.Accuracy*100, o.Latency, o.Score))
	}

	sb.WriteString("\n## Reasoning\n")
	for _, o := range optResult.ModelOutcomes {
		if o.ModelID == optResult.BestModel {
			sb.WriteString(fmt.Sprintf("The model **%s** was selected because it achieved the highest balanced score, favoring accuracy while maintaining acceptable latency.\n", o.ModelID))
			sb.WriteString(fmt.Sprintf("Best Hyperparameters: Temp=%.1f\n", o.BestParams.Temperature))
		}
	}

	os.WriteFile(outputPath, []byte(sb.String()), 0644)
	fmt.Printf("\nFull report saved to: %s\n", outputPath)
}
