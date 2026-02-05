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

	"github.com/oarkflow/ai-agent/pkg/agent"
	"github.com/oarkflow/ai-agent/pkg/config"
	"github.com/oarkflow/ai-agent/pkg/llm"
	"github.com/oarkflow/ai-agent/pkg/training"
)

func main() {
	var (
		domainID    string
		modelID     string
		datasetPath string
		outputPath  string
		configPath  string
	)

	flag.StringVar(&domainID, "domain", "medical-coding", "Domain ID to evolve")
	flag.StringVar(&modelID, "model", "mistral:latest", "Model ID to use for benchmarking")
	flag.StringVar(&datasetPath, "dataset", "data/examples/healthcare/examples.json", "Path to benchmark dataset")
	flag.StringVar(&outputPath, "output", "evolve_report.md", "Path to save evolution report")
	flag.StringVar(&configPath, "config", "config", "Path to config directory")
	flag.Parse()

	// 1. Load Config
	cfg := config.MustLoadConfig(configPath)

	// 2. Initialize Registry
	registry, err := llm.NewProviderRegistryFromConfig(cfg)
	if err != nil {
		log.Fatalf("Failed to initialize registry: %v", err)
	}

	// 3. Initialize Trainer
	mInfo, ok := registry.GetModel(modelID)
	if !ok {
		log.Fatalf("Model %s not found", modelID)
	}
	trainer, err := training.NewDomainTrainerFromConfig(cfg, mInfo.Provider, nil)
	if err != nil {
		log.Fatalf("Failed to initialize trainer: %v", err)
	}

	// 4. STEP 1: Benchmark (Discover Failures)
	fmt.Printf("--- Step 1: Benchmarking %s ---\n", modelID)
	evaluator := training.NewEvaluator(training.DefaultEvaluationConfig())
	optimizer := training.NewOptimizer(trainer, evaluator, registry)

	datasetData, _ := os.ReadFile(datasetPath)
	var dataPoints []training.DataPoint
	json.Unmarshal(datasetData, &dataPoints)

	mAgent := agent.NewMultimodalAgent("EvolveAgent", registry,
		agent.WithDomainTrainer(trainer),
		agent.WithConfig(&agent.AgentConfig{DefaultModel: modelID, DomainID: domainID, EnableRAG: true}),
	)

	predictFn := func(input string) (training.Prediction, error) {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
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

	benchResult, err := evaluator.Evaluate(dataPoints, predictFn, modelID, domainID)
	if err != nil {
		log.Fatalf("Initial benchmark failed: %v", err)
	}
	fmt.Printf("Initial Accuracy: %.2f%%\n\n", benchResult.Metrics.Accuracy*100)

	// 5. STEP 2: RAG Auto-Tune
	fmt.Println("--- Step 2: Auto-Tuning RAG ---")
	ragResult, err := optimizer.AutoTuneRAG(context.Background(), domainID, dataPoints)
	if err != nil {
		fmt.Printf("RAG Auto-Tune failed: %v\n", err)
	} else {
		fmt.Printf("Best Hit Rate found: %.2f (Size=%d, TopK=%d)\n\n",
			ragResult.MetricsDiff["hit_rate"], ragResult.NewConfig.ChunkSize, ragResult.NewConfig.TopK)
	}

	// 6. STEP 3: Knowledge Evolution (AI Synthesis)
	fmt.Println("--- Step 3: Knowledge Evolution ---")
	evolveResult, err := optimizer.EvolveKnowledge(context.Background(), domainID, benchResult)
	if err != nil {
		fmt.Printf("Evolution failed: %v\n", err)
	} else {
		fmt.Printf("Suggested %d new terms and %d new guidelines\n\n",
			len(evolveResult.NewTerms), len(evolveResult.NewGuidelines))
	}

	// 7. Generate Report
	var sb strings.Builder
	sb.WriteString("# DDMI Evolution Report\n\n")
	sb.WriteString(fmt.Sprintf("- **Domain**: %s\n", domainID))
	sb.WriteString(fmt.Sprintf("- **Model**: %s\n", modelID))
	sb.WriteString(fmt.Sprintf("- **Date**: %s\n\n", time.Now().Format(time.RFC822)))

	sb.WriteString("## Recommendations\n\n")
	if ragResult != nil && ragResult.Improved {
		sb.WriteString("### RAG Optimization\n")
		sb.WriteString(fmt.Sprintf("- **Recommended Chunk Size**: %d\n", ragResult.NewConfig.ChunkSize))
		sb.WriteString(fmt.Sprintf("- **Recommended Top K**: %d\n", ragResult.NewConfig.TopK))
		sb.WriteString("- **Est. Hit Rate**: " + fmt.Sprintf("%.2f\n\n", ragResult.MetricsDiff["hit_rate"]))
	}

	if evolveResult != nil && evolveResult.Improved {
		sb.WriteString("### Knowledge Expansion\n")
		sb.WriteString("#### New Terminology\n")
		for k, v := range evolveResult.NewTerms {
			sb.WriteString(fmt.Sprintf("- **%s**: %s\n", k, v))
		}
		sb.WriteString("\n#### New Guidelines\n")
		for _, g := range evolveResult.NewGuidelines {
			sb.WriteString("- " + g + "\n")
		}
	}

	os.WriteFile(outputPath, []byte(sb.String()), 0644)
	fmt.Printf("Evolution complete. Report saved to: %s\n", outputPath)
}
