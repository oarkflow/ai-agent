package training

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// Pipeline orchestrates the complete training workflow.
type Pipeline struct {
	config   PipelineConfig
	registry *DomainRegistry
	storage  *PipelineStorage
}

// PipelineConfig configures the training pipeline.
type PipelineConfig struct {
	// Directories
	DataDir       string `json:"data_dir"`
	OutputDir     string `json:"output_dir"`
	CheckpointDir string `json:"checkpoint_dir"`
	LogDir        string `json:"log_dir"`

	// Stages
	EnableDataCleaning   bool `json:"enable_data_cleaning"`
	EnableValidation     bool `json:"enable_validation"`
	EnableHyperparamSearch bool `json:"enable_hyperparam_search"`
	EnableEvaluation     bool `json:"enable_evaluation"`

	// Parallelism
	MaxParallelJobs int `json:"max_parallel_jobs"`

	// Logging
	Verbose      bool `json:"verbose"`
	LogInterval  int  `json:"log_interval"` // Steps
	SaveInterval int  `json:"save_interval"` // Steps
}

// DefaultPipelineConfig returns default configuration.
func DefaultPipelineConfig() PipelineConfig {
	return PipelineConfig{
		DataDir:              "data",
		OutputDir:            "output",
		CheckpointDir:        "checkpoints",
		LogDir:               "logs",
		EnableDataCleaning:   true,
		EnableValidation:     true,
		EnableHyperparamSearch: false,
		EnableEvaluation:     true,
		MaxParallelJobs:      1,
		Verbose:              true,
		LogInterval:          100,
		SaveInterval:         1000,
	}
}

// NewPipeline creates a new training pipeline.
func NewPipeline(config PipelineConfig) *Pipeline {
	return &Pipeline{
		config:   config,
		registry: NewDomainRegistry(config.DataDir),
		storage:  NewPipelineStorage(config.OutputDir),
	}
}

// PipelineStorage manages pipeline artifacts.
type PipelineStorage struct {
	baseDir string
}

// NewPipelineStorage creates a new storage manager.
func NewPipelineStorage(baseDir string) *PipelineStorage {
	return &PipelineStorage{baseDir: baseDir}
}

func (s *PipelineStorage) ensureDir(subdir string) error {
	return os.MkdirAll(filepath.Join(s.baseDir, subdir), 0755)
}

func (s *PipelineStorage) SaveJSON(subdir, filename string, data any) error {
	if err := s.ensureDir(subdir); err != nil {
		return err
	}

	content, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return err
	}

	path := filepath.Join(s.baseDir, subdir, filename)
	return os.WriteFile(path, content, 0644)
}

func (s *PipelineStorage) LoadJSON(subdir, filename string, target any) error {
	path := filepath.Join(s.baseDir, subdir, filename)
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, target)
}

// PipelineJob represents a training job.
type PipelineJob struct {
	ID          string            `json:"id"`
	DomainID    string            `json:"domain_id"`
	Status      JobStatus         `json:"status"`
	Config      JobConfig         `json:"config"`
	Stages      []StageResult     `json:"stages"`
	StartedAt   time.Time         `json:"started_at"`
	CompletedAt time.Time         `json:"completed_at,omitempty"`
	Error       string            `json:"error,omitempty"`
	Metrics     *Metrics          `json:"metrics,omitempty"`
	Artifacts   map[string]string `json:"artifacts,omitempty"`
}

// JobStatus represents job status.
type JobStatus string

const (
	JobPending   JobStatus = "pending"
	JobRunning   JobStatus = "running"
	JobCompleted JobStatus = "completed"
	JobFailed    JobStatus = "failed"
	JobCancelled JobStatus = "cancelled"
)

// JobConfig configures a training job.
type JobConfig struct {
	// Data settings
	DataSource       string        `json:"data_source"`
	DataSplit        DatasetSplit  `json:"data_split"`
	CleaningConfig   CleaningConfig `json:"cleaning_config,omitempty"`

	// Training settings
	Hyperparams      *Hyperparameters   `json:"hyperparams"`
	TrainerConfig    TrainerConfig      `json:"trainer_config"`
	EvaluationConfig EvaluationConfig   `json:"evaluation_config"`

	// Hyperparameter search
	SearchConfig     *HyperparameterSearchConfig `json:"search_config,omitempty"`
}

// StageResult holds the result of a pipeline stage.
type StageResult struct {
	Name      string        `json:"name"`
	Status    JobStatus     `json:"status"`
	StartedAt time.Time     `json:"started_at"`
	EndedAt   time.Time     `json:"ended_at"`
	Duration  time.Duration `json:"duration"`
	Output    any           `json:"output,omitempty"`
	Error     string        `json:"error,omitempty"`
}

// RegisterDomain registers a domain with the pipeline.
func (p *Pipeline) RegisterDomain(domain *FlexibleDomain) error {
	return p.registry.Register(domain)
}

// LoadDomainsFromDir loads domains from a directory.
func (p *Pipeline) LoadDomainsFromDir(dir string) error {
	return p.registry.LoadFromDirectory(dir)
}

// GetDomain retrieves a registered domain.
func (p *Pipeline) GetDomain(id string) (*FlexibleDomain, bool) {
	return p.registry.Get(id)
}

// ListDomains returns all registered domain IDs.
func (p *Pipeline) ListDomains() []string {
	return p.registry.List()
}

// CreateJob creates a new training job.
func (p *Pipeline) CreateJob(domainID string, config JobConfig) (*PipelineJob, error) {
	domain, ok := p.registry.Get(domainID)
	if !ok {
		return nil, fmt.Errorf("domain %s not found", domainID)
	}

	// Apply domain defaults if not specified
	if config.Hyperparams == nil {
		runner := NewDomainRunner(domain)
		config.Hyperparams = runner.GetSuggestedHyperparams()
	}

	if config.DataSplit.Train == 0 {
		config.DataSplit = DatasetSplit{
			Train:      domain.Training.TrainRatio,
			Validation: domain.Training.ValidationRatio,
			Test:       domain.Training.TestRatio,
		}
	}

	job := &PipelineJob{
		ID:        generateID(fmt.Sprintf("%s-%d", domainID, time.Now().UnixNano())),
		DomainID:  domainID,
		Status:    JobPending,
		Config:    config,
		Stages:    make([]StageResult, 0),
		StartedAt: time.Now(),
		Artifacts: make(map[string]string),
	}

	return job, nil
}

// RunJob executes a training job.
func (p *Pipeline) RunJob(ctx context.Context, job *PipelineJob, data []DataPoint, trainFn TrainFunc, validateFn ValidateFunc) error {
	job.Status = JobRunning

	// Stage 1: Data Cleaning
	if p.config.EnableDataCleaning {
		stage := p.runStage("data_cleaning", func() (any, error) {
			cleaner := NewDataCleaner(&job.Config.CleaningConfig)
			// Convert to pointer slice
			dataPtr := make([]*DataPoint, len(data))
			for i := range data {
				dataPtr[i] = &data[i]
			}
			cleanedData, _ := cleaner.Clean(dataPtr)
			// Convert back
			data = make([]DataPoint, len(cleanedData))
			for i, dp := range cleanedData {
				data[i] = *dp
			}
			return map[string]any{
				"original_count": len(dataPtr),
				"cleaned_count":  len(cleanedData),
			}, nil
		})
		job.Stages = append(job.Stages, stage)
		if stage.Status == JobFailed {
			job.Status = JobFailed
			job.Error = stage.Error
			return fmt.Errorf("data cleaning failed: %s", stage.Error)
		}
	}

	// Stage 2: Data Splitting
	var trainData, valData, testData []DataPoint
	{
		stage := p.runStage("data_splitting", func() (any, error) {
			// Convert to pointer slice
			dataPtr := make([]*DataPoint, len(data))
			for i := range data {
				dataPtr[i] = &data[i]
			}

			builder := NewDatasetBuilder(job.ID, job.DomainID, job.DomainID)
			builder.AddData(dataPtr)
			builder.WithSplit(job.Config.DataSplit.Train, job.Config.DataSplit.Validation, job.Config.DataSplit.Test)

			dataset, err := builder.Build()
			if err != nil {
				return nil, err
			}

			// Convert back to value slices
			trainData = make([]DataPoint, len(dataset.Train))
			for i, dp := range dataset.Train {
				trainData[i] = *dp
			}
			valData = make([]DataPoint, len(dataset.Validation))
			for i, dp := range dataset.Validation {
				valData[i] = *dp
			}
			testData = make([]DataPoint, len(dataset.Test))
			for i, dp := range dataset.Test {
				testData[i] = *dp
			}

			return map[string]any{
				"train_size":      len(trainData),
				"validation_size": len(valData),
				"test_size":       len(testData),
			}, nil
		})
		job.Stages = append(job.Stages, stage)
		if stage.Status == JobFailed {
			job.Status = JobFailed
			job.Error = stage.Error
			return fmt.Errorf("data splitting failed: %s", stage.Error)
		}
	}

	// Stage 3: Hyperparameter Search (optional)
	if p.config.EnableHyperparamSearch && job.Config.SearchConfig != nil {
		stage := p.runStage("hyperparam_search", func() (any, error) {
			search := NewHyperparameterSearch(job.Config.SearchConfig)
			baseParams := job.Config.Hyperparams
			if baseParams == nil {
				baseParams = DefaultHyperparameters()
			}

			// Generate trial configurations
			trialParams := search.GenerateTrials(baseParams)

			// Run search with validation
			for i, params := range trialParams {
				if i >= job.Config.SearchConfig.MaxTrials {
					break
				}

				// Quick validation run
				trainer := NewTrainer(TrainerConfig{
					OutputDir:           filepath.Join(p.config.OutputDir, "hp_search", fmt.Sprintf("trial_%d", i)),
					Epochs:              1, // Quick evaluation
					ValidationFrequency: 1,
					EarlyStopping:       false,
				}, params)

				run, err := trainer.Train(trainData[:minInt(100, len(trainData))], valData[:minInt(20, len(valData))], trainFn, validateFn, fmt.Sprintf("hp_trial_%d", i), job.DomainID)
				if err != nil {
					continue
				}

				trial := &HyperparameterTrial{
					ID:        i,
					Params:    params,
					Metrics:   run.FinalMetrics,
					Status:    "completed",
					StartTime: run.StartedAt,
					EndTime:   run.EndedAt,
				}
				search.RecordTrial(trial)
			}

			best := search.GetBestTrial()
			if best != nil {
				job.Config.Hyperparams = best.Params
			}

			return map[string]any{
				"best_metric": func() float64 { if best != nil && best.Metrics != nil { return best.Metrics.Accuracy }; return 0 }(),
				"trials_run":  len(search.GetTrials()),
			}, nil
		})
		job.Stages = append(job.Stages, stage)
		// Don't fail on HP search failure, continue with defaults
	}

	// Stage 4: Training
	var trainingRun *TrainingRun
	{
		stage := p.runStage("training", func() (any, error) {
			trainer := NewTrainer(job.Config.TrainerConfig, job.Config.Hyperparams)
			trainer.AddCallback(&LoggingCallback{Verbose: p.config.Verbose})

			var err error
			trainingRun, err = trainer.Train(trainData, valData, trainFn, validateFn, job.ID, job.DomainID)
			if err != nil {
				return nil, err
			}

			return map[string]any{
				"epochs_completed": trainingRun.BestEpoch,
				"final_loss":       trainingRun.FinalMetrics.Loss,
				"best_accuracy":    trainingRun.BestMetrics.Accuracy,
			}, nil
		})
		job.Stages = append(job.Stages, stage)
		if stage.Status == JobFailed {
			job.Status = JobFailed
			job.Error = stage.Error
			return fmt.Errorf("training failed: %s", stage.Error)
		}
	}

	// Stage 5: Evaluation
	if p.config.EnableEvaluation && len(testData) > 0 {
		stage := p.runStage("evaluation", func() (any, error) {
			evaluator := NewEvaluator(job.Config.EvaluationConfig)

			// Create prediction function from trained model
			predictFn := func(input string) (Prediction, error) {
				// This would use the trained model
				// For now, return a placeholder
				return Prediction{
					Input:          input,
					Predicted:      "",
					PredictedClass: "",
					Confidence:     0,
				}, nil
			}

			result, err := evaluator.Evaluate(testData, predictFn, job.ID, job.DomainID)
			if err != nil {
				return nil, err
			}

			job.Metrics = result.Metrics

			// Save evaluation report
			report := GenerateEvaluationReport(result)
			reportPath := filepath.Join(p.config.OutputDir, job.ID, "evaluation_report.txt")
			os.WriteFile(reportPath, []byte(report), 0644)
			job.Artifacts["evaluation_report"] = reportPath

			return result.Metrics, nil
		})
		job.Stages = append(job.Stages, stage)
		// Don't fail on evaluation failure
	}

	// Save training run
	if trainingRun != nil {
		runPath := filepath.Join(p.config.OutputDir, job.ID, "training_run.json")
		p.storage.SaveJSON(job.ID, "training_run.json", trainingRun)
		job.Artifacts["training_run"] = runPath
	}

	job.Status = JobCompleted
	job.CompletedAt = time.Now()

	// Save job metadata
	p.storage.SaveJSON(job.ID, "job.json", job)

	return nil
}

func (p *Pipeline) runStage(name string, fn func() (any, error)) StageResult {
	stage := StageResult{
		Name:      name,
		Status:    JobRunning,
		StartedAt: time.Now(),
	}

	if p.config.Verbose {
		fmt.Printf("\n[Pipeline] Starting stage: %s\n", name)
	}

	output, err := fn()

	stage.EndedAt = time.Now()
	stage.Duration = stage.EndedAt.Sub(stage.StartedAt)

	if err != nil {
		stage.Status = JobFailed
		stage.Error = err.Error()
		if p.config.Verbose {
			fmt.Printf("[Pipeline] Stage %s failed: %v\n", name, err)
		}
	} else {
		stage.Status = JobCompleted
		stage.Output = output
		if p.config.Verbose {
			fmt.Printf("[Pipeline] Stage %s completed in %v\n", name, stage.Duration)
		}
	}

	return stage
}

// QuickTrain is a convenience method for simple training.
func (p *Pipeline) QuickTrain(domainID string, data []DataPoint, trainFn TrainFunc, validateFn ValidateFunc) (*PipelineJob, error) {
	domain, ok := p.registry.Get(domainID)
	if !ok {
		return nil, fmt.Errorf("domain %s not found", domainID)
	}

	runner := NewDomainRunner(domain)

	job, err := p.CreateJob(domainID, JobConfig{
		DataSource:       "direct",
		DataSplit:        DatasetSplit{
			Train:      domain.Training.TrainRatio,
			Validation: domain.Training.ValidationRatio,
			Test:       domain.Training.TestRatio,
		},
		CleaningConfig:   *DefaultCleaningConfig(),
		Hyperparams:      runner.GetSuggestedHyperparams(),
		TrainerConfig:    DefaultTrainerConfig(),
		EvaluationConfig: DefaultEvaluationConfig(),
	})
	if err != nil {
		return nil, err
	}

	ctx := context.Background()
	if err := p.RunJob(ctx, job, data, trainFn, validateFn); err != nil {
		return job, err
	}

	return job, nil
}

// LoadJob loads a previously saved job.
func (p *Pipeline) LoadJob(jobID string) (*PipelineJob, error) {
	var job PipelineJob
	if err := p.storage.LoadJSON(jobID, "job.json", &job); err != nil {
		return nil, err
	}
	return &job, nil
}

// ListJobs returns all job IDs.
func (p *Pipeline) ListJobs() ([]string, error) {
	entries, err := os.ReadDir(p.config.OutputDir)
	if err != nil {
		if os.IsNotExist(err) {
			return []string{}, nil
		}
		return nil, err
	}

	var jobs []string
	for _, entry := range entries {
		if entry.IsDir() {
			jobPath := filepath.Join(p.config.OutputDir, entry.Name(), "job.json")
			if _, err := os.Stat(jobPath); err == nil {
				jobs = append(jobs, entry.Name())
			}
		}
	}

	return jobs, nil
}

// GetJobSummary returns a summary of a job.
func (p *Pipeline) GetJobSummary(job *PipelineJob) string {
	duration := job.CompletedAt.Sub(job.StartedAt)
	if job.CompletedAt.IsZero() {
		duration = time.Since(job.StartedAt)
	}

	summary := fmt.Sprintf(`
=======================================================
               PIPELINE JOB SUMMARY
=======================================================

Job ID:    %s
Domain:    %s
Status:    %s
Duration:  %v

Stages:
`, job.ID, job.DomainID, job.Status, duration)

	for _, stage := range job.Stages {
		status := "✓"
		if stage.Status == JobFailed {
			status = "✗"
		}
		summary += fmt.Sprintf("  %s %s (%v)\n", status, stage.Name, stage.Duration)
	}

	if job.Metrics != nil {
		summary += fmt.Sprintf(`
Metrics:
  Accuracy:  %.2f%%
  F1 Score:  %.4f
  Precision: %.4f
  Recall:    %.4f
`, job.Metrics.Accuracy*100, job.Metrics.F1Score, job.Metrics.Precision, job.Metrics.Recall)
	}

	if len(job.Artifacts) > 0 {
		summary += "\nArtifacts:\n"
		for name, path := range job.Artifacts {
			summary += fmt.Sprintf("  - %s: %s\n", name, path)
		}
	}

	if job.Error != "" {
		summary += fmt.Sprintf("\nError: %s\n", job.Error)
	}

	summary += "=======================================================\n"

	return summary
}

// PipelineBuilder provides a fluent API for building pipelines.
type PipelineBuilder struct {
	pipeline *Pipeline
}

// NewPipelineBuilder creates a new builder.
func NewPipelineBuilder() *PipelineBuilder {
	return &PipelineBuilder{
		pipeline: NewPipeline(DefaultPipelineConfig()),
	}
}

// WithConfig sets the pipeline configuration.
func (b *PipelineBuilder) WithConfig(config PipelineConfig) *PipelineBuilder {
	b.pipeline.config = config
	b.pipeline.storage = NewPipelineStorage(config.OutputDir)
	return b
}

// WithDataDir sets the data directory.
func (b *PipelineBuilder) WithDataDir(dir string) *PipelineBuilder {
	b.pipeline.config.DataDir = dir
	b.pipeline.registry = NewDomainRegistry(dir)
	return b
}

// WithOutputDir sets the output directory.
func (b *PipelineBuilder) WithOutputDir(dir string) *PipelineBuilder {
	b.pipeline.config.OutputDir = dir
	b.pipeline.storage = NewPipelineStorage(dir)
	return b
}

// WithDataCleaning enables/disables data cleaning.
func (b *PipelineBuilder) WithDataCleaning(enabled bool) *PipelineBuilder {
	b.pipeline.config.EnableDataCleaning = enabled
	return b
}

// WithEvaluation enables/disables evaluation.
func (b *PipelineBuilder) WithEvaluation(enabled bool) *PipelineBuilder {
	b.pipeline.config.EnableEvaluation = enabled
	return b
}

// WithHyperparamSearch enables/disables hyperparameter search.
func (b *PipelineBuilder) WithHyperparamSearch(enabled bool) *PipelineBuilder {
	b.pipeline.config.EnableHyperparamSearch = enabled
	return b
}

// WithVerbose enables/disables verbose output.
func (b *PipelineBuilder) WithVerbose(verbose bool) *PipelineBuilder {
	b.pipeline.config.Verbose = verbose
	return b
}

// AddDomain adds a domain to the pipeline.
func (b *PipelineBuilder) AddDomain(domain *FlexibleDomain) *PipelineBuilder {
	b.pipeline.RegisterDomain(domain)
	return b
}

// LoadDomains loads domains from a directory.
func (b *PipelineBuilder) LoadDomains(dir string) *PipelineBuilder {
	b.pipeline.LoadDomainsFromDir(dir)
	return b
}

// Build returns the configured pipeline.
func (b *PipelineBuilder) Build() *Pipeline {
	// Ensure directories exist
	os.MkdirAll(b.pipeline.config.DataDir, 0755)
	os.MkdirAll(b.pipeline.config.OutputDir, 0755)
	os.MkdirAll(b.pipeline.config.CheckpointDir, 0755)
	os.MkdirAll(b.pipeline.config.LogDir, 0755)

	return b.pipeline
}

// Helper function
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
