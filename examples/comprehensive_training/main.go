package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/sujit/ai-agent/pkg/training"
)

func main() {
	fmt.Println("==============================================")
	fmt.Println("  COMPREHENSIVE LLM TRAINING SYSTEM DEMO")
	fmt.Println("==============================================\n")

	demo1FlexibleDomainCreation()
	demo2DatasetPreparation()
	demo3Hyperparameters()
	demo4Metrics()
	demo5CompletePipeline()
}

func demo1FlexibleDomainCreation() {
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  Demo 1: Flexible Domain Definition")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

	sentimentDomain := training.NewClassificationDomain(
		"sentiment-analysis",
		"Sentiment Analysis",
		[]string{"positive", "negative", "neutral"},
	)
	fmt.Printf("Created classification domain: %s\n", sentimentDomain.Name)

	entityDomain := training.NewExtractionDomain(
		"entity-extraction",
		"Named Entity Extraction",
		[]string{"name", "organization", "location", "date"},
	)
	fmt.Printf("Created extraction domain: %s\n", entityDomain.Name)

	qaDomain := training.NewQADomain(
		"medical-qa",
		"Medical Question Answering",
		"medical and healthcare topics",
	)
	fmt.Printf("Created Q&A domain: %s\n", qaDomain.Name)

	codeDomain := training.NewCodeDomain(
		"python-gen",
		"Python Code Generator",
		"Python",
	)
	fmt.Printf("Created code domain: %s\n", codeDomain.Name)

	customDomain := training.NewCustomDomain(
		"workflow-gen",
		"Workflow Generator",
		"Generate n8n-like workflow DAGs",
		`You are an expert workflow designer. Generate workflow definitions in JSON format.`,
		"Generate a workflow for: {{input}}",
	)
	fmt.Printf("Created custom domain: %s\n", customDomain.Name)

	registry := training.NewDomainRegistry("./data/domains")
	registry.Register(sentimentDomain)
	registry.Register(entityDomain)
	registry.Register(qaDomain)
	registry.Register(codeDomain)
	registry.Register(customDomain)

	fmt.Printf("\nRegistered %d domains:\n", len(registry.List()))
	for _, id := range registry.List() {
		domain, _ := registry.Get(id)
		fmt.Printf("  - %s (%s)\n", domain.Name, domain.Category)
	}

	runner := training.NewDomainRunner(sentimentDomain)
	runner.AddExample(training.DomainExample{
		ID: "ex1", Input: "I love this!", Output: "positive", Quality: 0.95,
	})
	runner.AddExample(training.DomainExample{
		ID: "ex2", Input: "Terrible.", Output: "negative", Quality: 0.90,
	})

	prompt := runner.BuildPrompt("This is great!", nil, 2)
	fmt.Printf("\nGenerated prompt preview: %s...\n", prompt[:100])
	fmt.Println()
}

func demo2DatasetPreparation() {
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  Demo 2: Dataset Preparation & Cleaning")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

	rawData := []*training.DataPoint{
		{ID: "1", Input: "Great product!", Output: "positive", Quality: 0.9},
		{ID: "2", Input: "Bad quality", Output: "negative", Quality: 0.85},
		{ID: "3", Input: "", Output: "neutral", Quality: 0.5},
		{ID: "4", Input: "Good", Output: "positive", Quality: 0.7},
		{ID: "5", Input: "   Needs improvement   ", Output: "negative", Quality: 0.8},
		{ID: "6", Input: "Great product!", Output: "positive", Quality: 0.9},
		{ID: "7", Input: "Excellent service", Output: "positive", Quality: 0.95},
		{ID: "8", Input: "Not worth it", Output: "negative", Quality: 0.75},
		{ID: "9", Input: "Okay I guess", Output: "neutral", Quality: 0.6},
		{ID: "10", Input: "Love it!!!", Output: "positive", Quality: 0.88},
	}

	fmt.Printf("Raw data: %d samples\n", len(rawData))

	cleaningConfig := &training.CleaningConfig{
		RemoveEmptyInputs:  true,
		RemoveEmptyOutputs: true,
		RemoveDuplicates:   true,
		TrimWhitespace:     true,
		MinInputLength:     3,
		MaxInputLength:     10000,
		MinOutputLength:    1,
		MaxOutputLength:    10000,
		MinQuality:         0.5,
	}

	cleaner := training.NewDataCleaner(cleaningConfig)
	cleanedData, stats := cleaner.Clean(rawData)

	fmt.Printf("\nCleaning Results:\n")
	fmt.Printf("  Original: %d, Cleaned: %d, Removed: %d\n",
		stats.OriginalCount, stats.CleanedCount, stats.RemovedCount)
	fmt.Printf("  Duplicates found: %d\n", stats.DuplicatesFound)

	builder := training.NewDatasetBuilder("sentiment-v1", "Sentiment Dataset", "sentiment-analysis")
	builder.AddData(cleanedData)
	builder.WithSplit(0.7, 0.15, 0.15)
	builder.WithSeed(42)

	dataset, err := builder.Build()
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("\nDataset Built:\n")
	fmt.Printf("  Train: %d, Validation: %d, Test: %d\n",
		len(dataset.Train), len(dataset.Validation), len(dataset.Test))
	fmt.Println()
}

func demo3Hyperparameters() {
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  Demo 3: Hyperparameter Management")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

	hp := training.DefaultHyperparameters()
	fmt.Printf("Default Hyperparameters:\n")
	fmt.Printf("  Learning Rate: %.2e\n", hp.LearningRate)
	fmt.Printf("  Batch Size: %d\n", hp.BatchSize)
	fmt.Printf("  Epochs: %d\n", hp.Epochs)
	fmt.Printf("  Temperature: %.1f\n", hp.Temperature)

	hp.LearningRate = 2e-5
	hp.BatchSize = 16
	hp.Epochs = 5

	if err := hp.Validate(); err != nil {
		fmt.Printf("Invalid: %v\n", err)
	} else {
		fmt.Printf("\nHyperparameters validated successfully\n")
	}

	searchConfig := training.CommonSearchConfigs["quick"]
	fmt.Printf("\nHyperparameter Search (quick):\n")
	fmt.Printf("  Strategy: %s, Max Trials: %d\n", searchConfig.Strategy, searchConfig.MaxTrials)

	scheduler := training.NewHyperparameterScheduler(hp, 1000)
	fmt.Printf("\nLearning Rate Schedule:\n")
	for _, step := range []int{0, 50, 100, 500, 1000} {
		for i := 0; i < step; i++ {
			scheduler.Step()
		}
		fmt.Printf("  Step %4d: %.2e\n", step, scheduler.GetLearningRate())
		scheduler = training.NewHyperparameterScheduler(hp, 1000)
	}
	fmt.Println()
}

func demo4Metrics() {
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  Demo 4: Metrics Calculation")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

	calc := training.NewMetricsCalculator()

	predictions := []training.Prediction{
		{ID: "1", PredictedClass: "positive", ExpectedClass: "positive", Confidence: 0.95},
		{ID: "2", PredictedClass: "negative", ExpectedClass: "negative", Confidence: 0.88},
		{ID: "3", PredictedClass: "positive", ExpectedClass: "negative", Confidence: 0.72},
		{ID: "4", PredictedClass: "neutral", ExpectedClass: "neutral", Confidence: 0.65},
		{ID: "5", PredictedClass: "positive", ExpectedClass: "positive", Confidence: 0.91},
		{ID: "6", PredictedClass: "negative", ExpectedClass: "positive", Confidence: 0.58},
		{ID: "7", PredictedClass: "neutral", ExpectedClass: "neutral", Confidence: 0.77},
		{ID: "8", PredictedClass: "positive", ExpectedClass: "positive", Confidence: 0.89},
		{ID: "9", PredictedClass: "negative", ExpectedClass: "negative", Confidence: 0.93},
		{ID: "10", PredictedClass: "positive", ExpectedClass: "positive", Confidence: 0.96},
	}

	calc.AddPredictions(predictions)
	metrics := calc.Calculate()

	fmt.Printf("Classification Metrics:\n")
	fmt.Printf("  Accuracy: %.2f%% (%d/%d)\n",
		metrics.Accuracy*100, metrics.CorrectSamples, metrics.TotalSamples)
	fmt.Printf("  Precision: %.4f\n", metrics.Precision)
	fmt.Printf("  Recall: %.4f\n", metrics.Recall)
	fmt.Printf("  F1 Score: %.4f\n", metrics.F1Score)

	fmt.Printf("\nPer-Class Metrics:\n")
	for class, cm := range metrics.ClassMetrics {
		fmt.Printf("  %s: P=%.3f, R=%.3f, F1=%.3f\n",
			class, cm.Precision, cm.Recall, cm.F1Score)
	}

	confMatrix := calc.GetConfusionMatrix()
	fmt.Printf("\nConfusion Matrix Labels: %v\n", confMatrix.Labels)
	fmt.Println()
}

func demo5CompletePipeline() {
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  Demo 5: Complete Training Pipeline")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

	pipeline := training.NewPipelineBuilder().
		WithDataDir("./demo_data").
		WithOutputDir("./demo_output").
		WithDataCleaning(true).
		WithEvaluation(true).
		WithVerbose(false).
		Build()

	domain := training.NewClassificationDomain(
		"intent-classification",
		"Intent Classification",
		[]string{"greeting", "query", "complaint", "feedback"},
	)
	pipeline.RegisterDomain(domain)

	fmt.Printf("Created pipeline with domain: %s\n", domain.Name)

	sampleData := []training.DataPoint{
		{ID: "1", Input: "Hello there!", Output: "greeting", Quality: 0.9},
		{ID: "2", Input: "What is your return policy?", Output: "query", Quality: 0.85},
		{ID: "3", Input: "This product is broken!", Output: "complaint", Quality: 0.88},
		{ID: "4", Input: "Great service!", Output: "feedback", Quality: 0.92},
		{ID: "5", Input: "Hi, how are you?", Output: "greeting", Quality: 0.87},
		{ID: "6", Input: "How much is shipping?", Output: "query", Quality: 0.83},
		{ID: "7", Input: "I want a refund", Output: "complaint", Quality: 0.80},
		{ID: "8", Input: "Your app is amazing", Output: "feedback", Quality: 0.95},
	}

	fmt.Printf("Prepared %d training samples\n", len(sampleData))

	trainFn := func(batch []training.DataPoint, step int) (float64, error) {
		loss := 2.0 / (1.0 + float64(step)/100.0)
		return loss, nil
	}

	validateFn := func(data []training.DataPoint) (*training.Metrics, error) {
		return &training.Metrics{Loss: 0.5, Accuracy: 0.85, F1Score: 0.82}, nil
	}

	job, err := pipeline.CreateJob(domain.ID, training.JobConfig{
		DataSource:     "demo",
		DataSplit:      training.DatasetSplit{Train: 0.7, Validation: 0.15, Test: 0.15},
		CleaningConfig: *training.DefaultCleaningConfig(),
		Hyperparams: &training.Hyperparameters{
			LearningRate: 2e-5,
			BatchSize:    4,
			Epochs:       3,
			WarmupSteps:  10,
		},
		TrainerConfig: training.TrainerConfig{
			OutputDir:             "./demo_output",
			Epochs:                3,
			ValidationFrequency:   1,
			EarlyStopping:         true,
			EarlyStoppingPatience: 2,
		},
		EvaluationConfig: training.DefaultEvaluationConfig(),
	})

	if err != nil {
		fmt.Printf("Failed to create job: %v\n", err)
		return
	}

	fmt.Printf("Created job: %s\n", job.ID[:16]+"...")
	fmt.Println("\nRunning training pipeline...")

	ctx := context.Background()
	if err := pipeline.RunJob(ctx, job, sampleData, trainFn, validateFn); err != nil {
		fmt.Printf("Pipeline issue: %v\n", err)
	}

	summary := pipeline.GetJobSummary(job)
	fmt.Println(summary)

	domainJSON, _ := json.MarshalIndent(domain, "", "  ")
	fmt.Println("Domain Definition (truncated):")
	if len(domainJSON) > 300 {
		fmt.Println(string(domainJSON[:300]) + "...")
	} else {
		fmt.Println(string(domainJSON))
	}

	os.RemoveAll("./demo_data")
	os.RemoveAll("./demo_output")

	fmt.Println("\nDemo completed successfully!")
}
