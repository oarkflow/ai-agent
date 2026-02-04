package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
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
		modelIDs    string
		datasetPath string
		outputPath  string
		configPath  string
	)

	flag.StringVar(&domainID, "domain", "medical-coding", "Domain ID to benchmark")
	flag.StringVar(&modelIDs, "models", "mistral,qwen2.5:14b", "Comma-separated list of model IDs to compare")
	flag.StringVar(&datasetPath, "dataset", "data/examples/healthcare/examples.json", "Path to benchmark dataset")
	flag.StringVar(&outputPath, "output", "bench_report.md", "Path to save markdown report")
	flag.StringVar(&configPath, "config", "config", "Path to config directory")
	flag.Parse()

	// 1. Load Config
	cfg := config.MustLoadConfig(configPath)

	// 2. Initialize Registry
	registry, err := llm.NewProviderRegistryFromConfig(cfg)
	if err != nil {
		log.Fatalf("Failed to initialize registry: %v", err)
	}

	// 3. Initialize Trainer and Load Domain
	models := strings.Split(modelIDs, ",")
	if len(models) == 0 {
		log.Fatal("At least one model must be specified")
	}

	// Verify all models exist, try fuzzy matching if not found
	var resolvedModels []string
	available := registry.ListModels()

	for _, mID := range models {
		mID = strings.TrimSpace(mID)
		if _, ok := registry.GetModel(mID); ok {
			resolvedModels = append(resolvedModels, mID)
			continue
		}

		// Fuzzy match: if "mistral" not found, check if "mistral:latest" or similar exists
		found := false
		for _, v := range available {
			if strings.HasPrefix(v, mID+":") || v == mID {
				fmt.Printf("Model %s not found, resolved to %s\n", mID, v)
				resolvedModels = append(resolvedModels, v)
				found = true
				break
			}
		}

		if !found {
			log.Fatalf("Model %s not found in registry. Available models: %v", mID, available)
		}
	}

	firstModel, _ := registry.GetModel(resolvedModels[0])
	trainer, err := training.NewDomainTrainerFromConfig(cfg, firstModel.Provider, nil)
	if err != nil {
		log.Fatalf("Failed to initialize trainer: %v", err)
	}

	// 4. Load Dataset
	datasetData, err := os.ReadFile(datasetPath)
	if err != nil {
		log.Fatalf("Failed to read dataset: %v", err)
	}

	var rawPoints []*training.DataPoint
	if err := json.Unmarshal(datasetData, &rawPoints); err != nil {
		log.Fatalf("Failed to parse dataset: %v", err)
	}

	var dataPoints []training.DataPoint
	for _, rp := range rawPoints {
		dataPoints = append(dataPoints, *rp)
	}

	// 5. Run Benchmarking
	evaluator := training.NewEvaluator(training.DefaultEvaluationConfig())
	var results []*training.EvaluationResult

	fmt.Printf("Starting benchmark for domain: %s\n", domainID)
	fmt.Printf("Dataset: %s (%d samples)\n", datasetPath, len(dataPoints))
	fmt.Printf("Models: %v\n\n", resolvedModels)

	for _, mID := range resolvedModels {
		fmt.Printf("Evaluating model: %s... ", mID)

		mAgent := agent.NewMultimodalAgent("BenchmarkingAgent", registry,
			agent.WithDomainTrainer(trainer),
			agent.WithConfig(&agent.AgentConfig{
				DefaultModel: mID,
				DomainID:     domainID,
				EnableRAG:    true,
			}),
		)

		var totalTokens int
		predictFn := func(input string) (training.Prediction, error) {
			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second) // Increased timeout
			defer cancel()

			start := time.Now()
			resp, err := mAgent.Chat(ctx, input)
			latency := time.Since(start)

			if err != nil {
				return training.Prediction{}, err
			}

			if resp.Usage != nil {
				totalTokens += resp.Usage.TotalTokens
			}

			return training.Prediction{
				Input:     input,
				Predicted: resp.Message.GetText(),
				Latency:   latency,
			}, nil
		}

		result, err := evaluator.Evaluate(dataPoints, predictFn, mID, domainID)
		if err != nil {
			fmt.Printf("FAILED: %v\n", err)
			continue
		}

		if result.Metrics.Custom == nil {
			result.Metrics.Custom = make(map[string]float64)
		}
		result.Metrics.Custom["total_tokens"] = float64(totalTokens)

		fmt.Printf("DONE (Latency: %.2fs)\n", result.DurationSecs/float64(len(dataPoints)))
		results = append(results, result)
	}

	// 6. Generate Report
	var sb strings.Builder
	sb.WriteString("# DDMI Benchmarking Report\n\n")
	sb.WriteString(fmt.Sprintf("- **Domain**: %s\n", domainID))
	sb.WriteString(fmt.Sprintf("- **Dataset**: %s\n", datasetPath))
	sb.WriteString(fmt.Sprintf("- **Date**: %s\n\n", time.Now().Format(time.RFC822)))

	sb.WriteString("## Summary\n\n")
	sb.WriteString("| Model | Accuracy | Avg Latency | Total Tokens | Exact Match |\n")
	sb.WriteString("|---|---|---|---|---|\n")
	for _, r := range results {
		exactMatch := 0.0
		if r.Metrics.Custom != nil {
			exactMatch = r.Metrics.Custom["exact_match"]
		}
		tokens := 0.0
		if r.Metrics.Custom != nil {
			tokens = r.Metrics.Custom["total_tokens"]
		}
		sb.WriteString(fmt.Sprintf("| %s | %.2f%% | %.2f s | %.0f | %.2f%% |\n",
			r.ModelID, r.Metrics.Accuracy*100, r.Metrics.InferenceTimeMs/1000.0, tokens, exactMatch*100))
	}

	sb.WriteString("\n## Detailed Reports\n")
	for _, r := range results {
		sb.WriteString("### Model: " + r.ModelID + "\n")
		sb.WriteString("```\n")
		sb.WriteString(training.GenerateEvaluationReport(r))
		sb.WriteString("\n```\n")
		sb.WriteString("\n\n---\n\n")
	}

	err = os.WriteFile(outputPath, []byte(sb.String()), 0644)
	if err != nil {
		log.Fatalf("Failed to save report: %v", err)
	}

	fmt.Printf("\nBenchmarking complete. Report saved to: %s\n", outputPath)
}
