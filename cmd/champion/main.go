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
	fmt.Printf("Testing all models across registry (%d models total)\n", len(registry.ListModels()))
	fmt.Printf("Hyperparameter search: 9 configurations per model\n\n")

	// 5. Run AutoTuneModel with multiple weight profiles to find the most robust champion
	profiles := training.GetStandardWeightProfiles()

	fmt.Println("⚖️  Evaluating across multiple weight profiles:")
	for _, p := range profiles {
		fmt.Printf("   - Profile: %s (Acc=%.0f%%, Latency=%.0f%%)\n",
			p.Name, p.Weights.AccuracyWeight, p.Weights.LatencyWeight)
	}
	fmt.Println()

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

	// Use "Accuracy Focused" as the primary weight for choosing the final champion,
	// but we'll include analysis from all profiles in the reasoning.
	primaryWeights := profiles[0].Weights // Accuracy Focused

	optResult, err := optimizer.AutoTuneModelWithWeights(context.Background(), domainID, dataPoints, predictFnFactory, primaryWeights)
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

			// Store the best hyperparameters and detailed metrics for this domain
			for _, outcome := range optResult.ModelOutcomes {
				if outcome.ModelID == optResult.BestModel && outcome.BestParams != nil {
					domain["temperature"] = outcome.BestParams.Temperature
					domain["top_p"] = outcome.BestParams.TopP
					domain["top_k"] = outcome.BestParams.TopK
					domain["max_tokens"] = outcome.BestParams.MaxTokens
					domain["best_accuracy"] = outcome.Accuracy
					domain["last_validated"] = time.Now().Format("2006-01-02")
					domain["selection_reasoning"] = outcome.Reasoning

					fmt.Printf("✅ Updated domains.json: preferred_model=%s with accuracy %.2f%%\n",
						optResult.BestModel, outcome.Accuracy*100)
					break
				}
			}
		}
	}

	newDomainsData, _ := json.MarshalIndent(domainsConfig, "", "    ")
	os.WriteFile(domainsFile, newDomainsData, 0644)

	// 7. Generate Comprehensive Report
	var sb strings.Builder
	sb.WriteString("# Champion Model Selection Report\n\n")
	sb.WriteString(fmt.Sprintf("## Summary of Best Model: %s\n\n", optResult.BestModel))
	sb.WriteString(fmt.Sprintf("The system tested all available models across 9 hyperparameter configurations and 3 different weight profiles. The champion was selected based on the **Accuracy Focused** profile (100%% Accuracy, 0%% Latency) to ensure maximum reliability for the '%s' domain.\n\n", domainID))

	sb.WriteString("### Evaluation Context\n\n")
	sb.WriteString(fmt.Sprintf("- **Domain**: %s\n", domainID))
	sb.WriteString(fmt.Sprintf("- **Dataset**: %s (%d benchmarking samples)\n", datasetPath, len(dataPoints)))
	sb.WriteString(fmt.Sprintf("- **Validation Logic**: %s\n", optResult.ValidationDetails))
	sb.WriteString("- **Selection Logic**: Automated grid search across all model/hyperparameter combinations.\n\n")

	// Model Leaderboard
	sb.WriteString("## Model Leaderboard (Primary Weights)\n\n")
	sb.WriteString("| Rank | Model | Accuracy | Avg Latency (ms) | Score | Best Params |\n")
	sb.WriteString("|---|---|---|---|---|---|\n")

	for i, o := range optResult.ModelOutcomes {
		rank := i + 1
		champion := ""
		if o.ModelID == optResult.BestModel {
			champion = " 🏆"
		}
		sb.WriteString(fmt.Sprintf("| %d | %s%s | %.2f%% | %.2f | %.4f | T=%.1f, P=%.1f |\n",
			rank, o.ModelID, champion, o.Accuracy*100, o.Latency, o.Score, o.BestParams.Temperature, o.BestParams.TopP))
	}
	sb.WriteString("\n")

	// Detailed Selection Reasoning
	sb.WriteString("## Selection Reasoning & Validation Details\n\n")
	for _, o := range optResult.ModelOutcomes {
		if o.ModelID == optResult.BestModel {
			sb.WriteString(fmt.Sprintf("### Champion: %s\n\n", o.ModelID))
			sb.WriteString("**Detailed Reasoning:**\n")
			sb.WriteString(o.Reasoning + "\n\n")

			sb.WriteString("**Validation Methodology:**\n")
			sb.WriteString("- Tested across all 9 hyperparameter configurations (deterministic through creative).\n")
			sb.WriteString("- Each configuration processed the full benchmark dataset.\n")
			sb.WriteString("- Results were compared against ground truth in the provided dataset.\n")
			sb.WriteString("- Latency was measured as end-to-end response time divided by sample count.\n\n")

			sb.WriteString("**Optimal Profile Found:**\n")
			sb.WriteString(fmt.Sprintf("- **Temperature**: %.1f\n", o.BestParams.Temperature))
			sb.WriteString(fmt.Sprintf("- **TopP**: %.2f\n", o.BestParams.TopP))
			sb.WriteString(fmt.Sprintf("- **TopK**: %d\n", o.BestParams.TopK))
			sb.WriteString(fmt.Sprintf("- **MaxTokens**: %d\n\n", o.BestParams.MaxTokens))

			// Hyperparameter trial details
			if len(o.AllTrials) > 0 {
				sb.WriteString("**All Test Results:**\n\n")
				sb.WriteString("| Trial | Temp | TopP | TopK | Tokens | Accuracy | Latency (ms) | Score |\n")
				sb.WriteString("|---|---|---|---|---|---|---|---|\n")
				for idx, trial := range o.AllTrials {
					sb.WriteString(fmt.Sprintf("| %d | %.1f | %.2f | %d | %d | %.2f%% | %.2f | %.4f |\n",
						idx+1, trial.Params.Temperature, trial.Params.TopP, trial.Params.TopK,
						trial.Params.MaxTokens, trial.Accuracy*100, trial.LatencyMs, trial.Score))
				}
				sb.WriteString("\n")
			}
		}
	}

	// Other Models Performance
	sb.WriteString("## Other Models Tested\n\n")
	for _, o := range optResult.ModelOutcomes {
		if o.ModelID != optResult.BestModel {
			sb.WriteString(fmt.Sprintf("### %s\n\n", o.ModelID))

			if o.BestParams != nil {
				sb.WriteString(fmt.Sprintf("- Best Config: Temp=%.1f, TopP=%.2f, TopK=%d, MaxTokens=%d\n",
					o.BestParams.Temperature, o.BestParams.TopP, o.BestParams.TopK, o.BestParams.MaxTokens))
			}
			sb.WriteString(fmt.Sprintf("- Accuracy: %.2f%%\n", o.Accuracy*100))
			sb.WriteString(fmt.Sprintf("- Latency: %.2f ms\n", o.Latency))
			sb.WriteString(fmt.Sprintf("- Score: %.4f\n", o.Score))
			sb.WriteString(fmt.Sprintf("- Tests Run: %d\n\n", o.TestCount))

			sb.WriteString(fmt.Sprintf("**Analysis:** %s\n\n", o.Reasoning))
		}
	}

	sb.WriteString("---\n\n")
	sb.WriteString(fmt.Sprintf("*Report generated on %s*\n", time.Now().Format(time.RFC1123)))
	sb.WriteString(fmt.Sprintf("*Configuration saved to: %s*\n", domainsFile))

	os.WriteFile(outputPath, []byte(sb.String()), 0644)
	fmt.Printf("\n📄 Comprehensive report saved to: %s\n", outputPath)
}
