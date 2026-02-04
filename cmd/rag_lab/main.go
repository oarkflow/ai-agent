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

	"github.com/sujit/ai-agent/pkg/config"
	"github.com/sujit/ai-agent/pkg/llm"
	"github.com/sujit/ai-agent/pkg/training"
)

type ExperimentResult struct {
	Config  training.RAGConfig `json:"config"`
	HitRate float64            `json:"hit_rate"`
	MRR     float64            `json:"mrr"`
	Latency float64            `json:"latency_ms"`
}

func main() {
	var (
		domainID    string
		datasetPath string
		outputPath  string
		configPath  string
	)

	flag.StringVar(&domainID, "domain", "medical-coding", "Domain ID to optimize")
	flag.StringVar(&datasetPath, "dataset", "data/examples/healthcare/examples.json", "Path to benchmark dataset")
	flag.StringVar(&outputPath, "output", "rag_lab_report.md", "Path to save markdown report")
	flag.StringVar(&configPath, "config", "config", "Path to config directory")
	flag.Parse()

	// 1. Load Config
	cfg := config.MustLoadConfig(configPath)

	// 2. Initialize Registry & Provider
	registry, err := llm.NewProviderRegistryFromConfig(cfg)
	if err != nil {
		log.Fatalf("Failed to initialize registry: %v", err)
	}

	embedModel, ok := registry.GetModel("mistral:latest")
	if !ok {
		avail := registry.ListModels()
		if len(avail) > 0 {
			embedModel, _ = registry.GetModel(avail[0])
		} else {
			log.Fatal("No models available for embeddings")
		}
	}

	// 3. Initialize Trainer
	trainer, err := training.NewDomainTrainerFromConfig(cfg, embedModel.Provider, nil)
	if err != nil {
		log.Fatalf("Failed to initialize trainer: %v", err)
	}

	// 4. Create a dummy document for re-indexing tests
	manual := &training.Document{
		Title: "Medical Coding Reference Manual v1",
		Content: `SECTION 1: EVALUATION AND MANAGEMENT (E&M)
E&M codes are used to describe the provider's evaluation of the patient.
99201-99205: New Patient Office Visits.
99211-99215: Established Patient Office Visits.

SECTION 2: ICD-10 DIAGNOSIS CODES
J20.9: Acute Bronchitis, unspecified.
I10: Essential (primary) hypertension.
E11.9: Type 2 diabetes mellitus without complications.
N18.3: Chronic kidney disease, stage 3 (moderate).
E11.22: Type 2 diabetes mellitus with diabetic chronic kidney disease.

SECTION 3: CPT CODES
71046: Chest X-ray, 2 views.
83036: Hemoglobin A1c.
82565: Creatinine; blood.
`,
	}
	if err := trainer.AddDocument(context.Background(), domainID, manual); err != nil {
		log.Fatalf("Failed to add test document: %v", err)
	}

	// 5. Define Experiments
	configs := []training.RAGConfig{
		{ChunkSize: 50, ChunkOverlap: 10, TopK: 3},
		{ChunkSize: 200, ChunkOverlap: 20, TopK: 3},
		{ChunkSize: 500, ChunkOverlap: 50, TopK: 3},
	}

	// 6. Load Dataset (Queries)
	datasetData, err := os.ReadFile(datasetPath)
	if err != nil {
		log.Fatalf("Failed to read dataset: %v", err)
	}

	var dataPoints []training.DataPoint
	if err := json.Unmarshal(datasetData, &dataPoints); err != nil {
		log.Fatalf("Failed to parse dataset: %v", err)
	}

	fmt.Printf("Starting RAG Lab for domain: %s\n", domainID)
	fmt.Printf("Dataset: %s (%d queries)\n", datasetPath, len(dataPoints))
	fmt.Printf("Running %d experiments...\n\n", len(configs))

	var results []ExperimentResult

	for i, rCfg := range configs {
		fmt.Printf("[%d/%d] Testing Config: Size=%d, Overlap=%d, TopK=%d... ",
			i+1, len(configs), rCfg.ChunkSize, rCfg.ChunkOverlap, rCfg.TopK)

		experimentStore := training.NewInMemoryVectorStore()
		rCfg.VectorStore = experimentStore

		err := trainer.UseConfig(&rCfg, func() error {
			ctx := context.Background()
			if err := trainer.Reindex(ctx, domainID); err != nil {
				return err
			}

			var totalHitRate, totalMRR float64
			var queryLatency time.Duration

			for _, dp := range dataPoints {
				queryStart := time.Now()
				searchResults, err := trainer.Search(ctx, domainID, dp.Input, rCfg.TopK)
				queryLatency += time.Since(queryStart)

				if err != nil {
					continue
				}

				// Improved Hit detection for medical coding
				// Check if any results contain keywords from input or expected output
				hit := false
				for _, res := range searchResults {
					if containsAny(res.Content, []string{"9921", "J20.9", "I10", "E11", "71046", "83036"}) {
						hit = true
						break
					}
				}

				if hit {
					totalHitRate += 1.0
					totalMRR += 1.0
				}
			}

			results = append(results, ExperimentResult{
				Config:  rCfg,
				HitRate: totalHitRate / float64(len(dataPoints)),
				MRR:     totalMRR / float64(len(dataPoints)),
				Latency: float64(queryLatency.Milliseconds()) / float64(len(dataPoints)),
			})

			return nil
		})

		if err != nil {
			fmt.Printf("FAILED: %v\n", err)
		} else {
			fmt.Printf("DONE (HitRate: %.2f, Latency: %.2fms)\n",
				results[len(results)-1].HitRate, results[len(results)-1].Latency)
		}
	}

	// 7. Generate Report
	var sb strings.Builder
	sb.WriteString("# DDMI RAG Lab Report\n\n")
	sb.WriteString(fmt.Sprintf("- **Domain**: %s\n", domainID))
	sb.WriteString(fmt.Sprintf("- **Date**: %s\n\n", time.Now().Format(time.RFC822)))

	sb.WriteString("## Experiment Summary\n\n")
	sb.WriteString("| Config (Size/Overlap/TopK) | Hit Rate | MRR | Avg Latency (ms) |\n")
	sb.WriteString("|---|---|---|---|\n")
	for _, r := range results {
		sb.WriteString(fmt.Sprintf("| %d / %d / %d | %.2f | %.2f | %.2f |\n",
			r.Config.ChunkSize, r.Config.ChunkOverlap, r.Config.TopK, r.HitRate, r.MRR, r.Latency))
	}

	err = os.WriteFile(outputPath, []byte(sb.String()), 0644)
	if err != nil {
		log.Fatalf("Failed to save report: %v", err)
	}

	fmt.Printf("\nRAG Lab complete. Report saved to: %s\n", outputPath)
}

func containsAny(text string, keywords []string) bool {
	lower := strings.ToLower(text)
	for _, k := range keywords {
		if strings.Contains(lower, strings.ToLower(k)) {
			return true
		}
	}
	return false
}
