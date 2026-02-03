package training

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// Trainer manages the complete training lifecycle.
type Trainer struct {
	config      TrainerConfig
	hyperparams *Hyperparameters
	metrics     *MetricsCalculator
	evaluator   *Evaluator

	// State
	currentEpoch   int
	currentStep    int
	bestMetric     float64
	bestCheckpoint string
	trainingLoss   []float64
	validationLoss []float64

	// Callbacks
	callbacks []TrainingCallback

	mu sync.RWMutex
}

// TrainerConfig configures the trainer.
type TrainerConfig struct {
	// Directories
	OutputDir     string `json:"output_dir"`
	CheckpointDir string `json:"checkpoint_dir"`

	// Training settings
	Epochs              int `json:"epochs"`
	StepsPerEpoch       int `json:"steps_per_epoch"`      // 0 = auto
	ValidationFrequency int `json:"validation_frequency"` // Every N epochs
	LoggingSteps        int `json:"logging_steps"`

	// Early stopping
	EarlyStopping         bool    `json:"early_stopping"`
	EarlyStoppingPatience int     `json:"early_stopping_patience"`
	EarlyStoppingMetric   string  `json:"early_stopping_metric"`
	EarlyStoppingMode     string  `json:"early_stopping_mode"` // "min" or "max"
	MinDelta              float64 `json:"min_delta"`

	// Checkpointing
	SaveBestOnly     bool `json:"save_best_only"`
	SaveEveryNEpochs int  `json:"save_every_n_epochs"`
	MaxCheckpoints   int  `json:"max_checkpoints"`

	// Resume
	ResumeFromCheckpoint string `json:"resume_from_checkpoint,omitempty"`

	// Metrics
	TrackMetrics []string `json:"track_metrics"`
}

// DefaultTrainerConfig returns default configuration.
func DefaultTrainerConfig() TrainerConfig {
	return TrainerConfig{
		OutputDir:             "training_output",
		CheckpointDir:         "checkpoints",
		Epochs:                10,
		StepsPerEpoch:         0,
		ValidationFrequency:   1,
		LoggingSteps:          100,
		EarlyStopping:         true,
		EarlyStoppingPatience: 3,
		EarlyStoppingMetric:   "loss",
		EarlyStoppingMode:     "min",
		MinDelta:              0.0001,
		SaveBestOnly:          true,
		SaveEveryNEpochs:      1,
		MaxCheckpoints:        3,
		TrackMetrics:          []string{"loss", "accuracy", "f1_score"},
	}
}

// TrainingCallback is called during training.
type TrainingCallback interface {
	OnTrainBegin(trainer *Trainer)
	OnTrainEnd(trainer *Trainer)
	OnEpochBegin(trainer *Trainer, epoch int)
	OnEpochEnd(trainer *Trainer, epoch int, metrics *Metrics)
	OnBatchBegin(trainer *Trainer, batch int)
	OnBatchEnd(trainer *Trainer, batch int, loss float64)
}

// TrainingRun represents a complete training run.
type TrainingRun struct {
	// Identifiers
	RunID     string    `json:"run_id"`
	ModelName string    `json:"model_name"`
	StartedAt time.Time `json:"started_at"`
	EndedAt   time.Time `json:"ended_at"`

	// Configuration
	Config      TrainerConfig    `json:"config"`
	Hyperparams *Hyperparameters `json:"hyperparams"`

	// Results
	FinalMetrics    *Metrics       `json:"final_metrics"`
	BestMetrics     *Metrics       `json:"best_metrics"`
	BestEpoch       int            `json:"best_epoch"`
	TotalSteps      int            `json:"total_steps"`
	TrainingHistory []EpochHistory `json:"training_history"`

	// Status
	Status       string `json:"status"` // "running", "completed", "failed", "stopped"
	ErrorMessage string `json:"error_message,omitempty"`

	// Artifacts
	Checkpoints    []string `json:"checkpoints"`
	BestCheckpoint string   `json:"best_checkpoint"`
}

// EpochHistory stores metrics for each epoch.
type EpochHistory struct {
	Epoch          int       `json:"epoch"`
	TrainingLoss   float64   `json:"training_loss"`
	ValidationLoss float64   `json:"validation_loss,omitempty"`
	Metrics        *Metrics  `json:"metrics,omitempty"`
	LearningRate   float64   `json:"learning_rate"`
	Duration       float64   `json:"duration_secs"`
	Timestamp      time.Time `json:"timestamp"`
}

// NewTrainer creates a new trainer.
func NewTrainer(config TrainerConfig, hyperparams *Hyperparameters) *Trainer {
	evalConfig := DefaultEvaluationConfig()
	evalConfig.OutputDir = filepath.Join(config.OutputDir, "eval")

	return &Trainer{
		config:         config,
		hyperparams:    hyperparams,
		metrics:        NewMetricsCalculator(),
		evaluator:      NewEvaluator(evalConfig),
		trainingLoss:   make([]float64, 0),
		validationLoss: make([]float64, 0),
		callbacks:      make([]TrainingCallback, 0),
	}
}

// AddCallback adds a training callback.
func (t *Trainer) AddCallback(cb TrainingCallback) {
	t.callbacks = append(t.callbacks, cb)
}

// TrainFunc is called for each batch, returns loss.
type TrainFunc func(batch []DataPoint, step int) (float64, error)

// ValidateFunc is called for validation.
type ValidateFunc func(data []DataPoint) (*Metrics, error)

// Train runs the training loop.
func (t *Trainer) Train(
	trainData []DataPoint,
	valData []DataPoint,
	trainFn TrainFunc,
	validateFn ValidateFunc,
	runID string,
	modelName string,
) (*TrainingRun, error) {
	run := &TrainingRun{
		RunID:           runID,
		ModelName:       modelName,
		StartedAt:       time.Now(),
		Config:          t.config,
		Hyperparams:     t.hyperparams,
		TrainingHistory: make([]EpochHistory, 0),
		Checkpoints:     make([]string, 0),
		Status:          "running",
	}

	// Create directories
	if err := os.MkdirAll(t.config.OutputDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create output dir: %w", err)
	}
	if err := os.MkdirAll(filepath.Join(t.config.OutputDir, t.config.CheckpointDir), 0755); err != nil {
		return nil, fmt.Errorf("failed to create checkpoint dir: %w", err)
	}

	// Initialize
	for _, cb := range t.callbacks {
		cb.OnTrainBegin(t)
	}

	totalSteps := t.config.Epochs * ((len(trainData) + t.hyperparams.BatchSize - 1) / t.hyperparams.BatchSize)
	scheduler := NewHyperparameterScheduler(t.hyperparams, totalSteps)
	patienceCounter := 0
	batchSize := t.hyperparams.BatchSize
	if batchSize == 0 {
		batchSize = 32
	}

	stepsPerEpoch := t.config.StepsPerEpoch
	if stepsPerEpoch == 0 {
		stepsPerEpoch = (len(trainData) + batchSize - 1) / batchSize
	}

	defer func() {
		for _, cb := range t.callbacks {
			cb.OnTrainEnd(t)
		}
	}()

	// Training loop
	for epoch := 0; epoch < t.config.Epochs; epoch++ {
		t.currentEpoch = epoch
		epochStart := time.Now()

		for _, cb := range t.callbacks {
			cb.OnEpochBegin(t, epoch)
		}

		// Training phase
		var epochLoss float64
		numBatches := 0

		for step := 0; step < stepsPerEpoch; step++ {
			t.currentStep = epoch*stepsPerEpoch + step

			// Get batch
			batchStart := (step * batchSize) % len(trainData)
			batchEnd := batchStart + batchSize
			if batchEnd > len(trainData) {
				batchEnd = len(trainData)
			}
			batch := trainData[batchStart:batchEnd]

			for _, cb := range t.callbacks {
				cb.OnBatchBegin(t, step)
			}

			// Update learning rate
			lr := scheduler.Step()
			_ = lr // Would be passed to optimizer in real implementation

			// Train step
			loss, err := trainFn(batch, t.currentStep)
			if err != nil {
				run.Status = "failed"
				run.ErrorMessage = err.Error()
				return run, err
			}

			epochLoss += loss
			numBatches++

			for _, cb := range t.callbacks {
				cb.OnBatchEnd(t, step, loss)
			}

			// Logging
			if t.config.LoggingSteps > 0 && t.currentStep%t.config.LoggingSteps == 0 {
				t.logStep(t.currentStep, loss, lr)
			}
		}

		avgTrainLoss := epochLoss / float64(numBatches)
		t.trainingLoss = append(t.trainingLoss, avgTrainLoss)

		// Validation phase
		var valMetrics *Metrics
		var avgValLoss float64
		if epoch%t.config.ValidationFrequency == 0 && validateFn != nil && len(valData) > 0 {
			var err error
			valMetrics, err = validateFn(valData)
			if err != nil {
				run.Status = "failed"
				run.ErrorMessage = err.Error()
				return run, err
			}
			avgValLoss = valMetrics.Loss
			t.validationLoss = append(t.validationLoss, avgValLoss)
		}

		epochDuration := time.Since(epochStart).Seconds()

		// Record history
		history := EpochHistory{
			Epoch:          epoch,
			TrainingLoss:   avgTrainLoss,
			ValidationLoss: avgValLoss,
			Metrics:        valMetrics,
			LearningRate:   scheduler.GetLearningRate(),
			Duration:       epochDuration,
			Timestamp:      time.Now(),
		}
		run.TrainingHistory = append(run.TrainingHistory, history)

		for _, cb := range t.callbacks {
			cb.OnEpochEnd(t, epoch, valMetrics)
		}

		// Checkpointing
		currentMetric := t.getMetricValue(valMetrics, avgValLoss, t.config.EarlyStoppingMetric)
		improved := t.checkImprovement(currentMetric)

		if improved {
			t.bestMetric = currentMetric
			patienceCounter = 0
			run.BestEpoch = epoch
			run.BestMetrics = valMetrics

			if t.config.SaveBestOnly {
				checkpoint, err := t.saveCheckpoint(run, epoch, valMetrics)
				if err == nil {
					t.bestCheckpoint = checkpoint
					run.BestCheckpoint = checkpoint
				}
			}
		} else {
			patienceCounter++
		}

		// Save periodic checkpoints
		if !t.config.SaveBestOnly && t.config.SaveEveryNEpochs > 0 && epoch%t.config.SaveEveryNEpochs == 0 {
			checkpoint, err := t.saveCheckpoint(run, epoch, valMetrics)
			if err == nil {
				run.Checkpoints = append(run.Checkpoints, checkpoint)
				t.cleanupCheckpoints(run)
			}
		}

		// Early stopping
		if t.config.EarlyStopping && patienceCounter >= t.config.EarlyStoppingPatience {
			run.Status = "completed"
			run.EndedAt = time.Now()
			run.FinalMetrics = valMetrics
			run.TotalSteps = t.currentStep
			t.saveRun(run)
			return run, nil
		}
	}

	run.Status = "completed"
	run.EndedAt = time.Now()
	run.TotalSteps = t.currentStep
	if len(run.TrainingHistory) > 0 {
		run.FinalMetrics = run.TrainingHistory[len(run.TrainingHistory)-1].Metrics
	}

	t.saveRun(run)
	return run, nil
}

func (t *Trainer) getMetricValue(metrics *Metrics, defaultLoss float64, metricName string) float64 {
	if metrics == nil {
		return defaultLoss
	}

	switch metricName {
	case "loss":
		return metrics.Loss
	case "accuracy":
		return metrics.Accuracy
	case "f1_score":
		return metrics.F1Score
	case "precision":
		return metrics.Precision
	case "recall":
		return metrics.Recall
	default:
		if v, ok := metrics.Custom[metricName]; ok {
			return v
		}
		return defaultLoss
	}
}

func (t *Trainer) checkImprovement(current float64) bool {
	if t.bestMetric == 0 {
		return true
	}

	switch t.config.EarlyStoppingMode {
	case "min":
		return current < (t.bestMetric - t.config.MinDelta)
	case "max":
		return current > (t.bestMetric + t.config.MinDelta)
	default:
		return current < (t.bestMetric - t.config.MinDelta)
	}
}

func (t *Trainer) saveCheckpoint(run *TrainingRun, epoch int, metrics *Metrics) (string, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	checkpoint := Checkpoint{
		RunID:       run.RunID,
		Epoch:       epoch,
		Step:        t.currentStep,
		Metrics:     metrics,
		Hyperparams: t.hyperparams,
		SavedAt:     time.Now(),
	}

	filename := fmt.Sprintf("checkpoint_epoch_%d.json", epoch)
	path := filepath.Join(t.config.OutputDir, t.config.CheckpointDir, filename)

	data, err := json.MarshalIndent(checkpoint, "", "  ")
	if err != nil {
		return "", err
	}

	return path, os.WriteFile(path, data, 0644)
}

func (t *Trainer) cleanupCheckpoints(run *TrainingRun) {
	if t.config.MaxCheckpoints <= 0 || len(run.Checkpoints) <= t.config.MaxCheckpoints {
		return
	}

	toRemove := len(run.Checkpoints) - t.config.MaxCheckpoints
	for i := 0; i < toRemove; i++ {
		os.Remove(run.Checkpoints[i])
	}
	run.Checkpoints = run.Checkpoints[toRemove:]
}

func (t *Trainer) saveRun(run *TrainingRun) error {
	path := filepath.Join(t.config.OutputDir, fmt.Sprintf("run_%s.json", run.RunID))
	data, err := json.MarshalIndent(run, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func (t *Trainer) logStep(step int, loss, lr float64) {
	fmt.Printf("Step %d | Loss: %.4f | LR: %.6f\n", step, loss, lr)
}

// Checkpoint represents a saved model state.
type Checkpoint struct {
	RunID       string           `json:"run_id"`
	Epoch       int              `json:"epoch"`
	Step        int              `json:"step"`
	Metrics     *Metrics         `json:"metrics,omitempty"`
	Hyperparams *Hyperparameters `json:"hyperparams"`
	SavedAt     time.Time        `json:"saved_at"`
	ModelPath   string           `json:"model_path,omitempty"`
}

// LoadCheckpoint loads a checkpoint from file.
func LoadCheckpoint(path string) (*Checkpoint, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var checkpoint Checkpoint
	if err := json.Unmarshal(data, &checkpoint); err != nil {
		return nil, err
	}

	return &checkpoint, nil
}

// TrainingProgress returns current progress.
func (t *Trainer) TrainingProgress() (epoch int, step int, loss float64) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	epoch = t.currentEpoch
	step = t.currentStep
	if len(t.trainingLoss) > 0 {
		loss = t.trainingLoss[len(t.trainingLoss)-1]
	}
	return
}

// GetTrainingHistory returns training history.
func (t *Trainer) GetTrainingHistory() (trainLoss, valLoss []float64) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.trainingLoss, t.validationLoss
}

// LoggingCallback logs training progress.
type LoggingCallback struct {
	Verbose bool
}

func (c *LoggingCallback) OnTrainBegin(trainer *Trainer) {
	if c.Verbose {
		fmt.Println("Training started...")
	}
}

func (c *LoggingCallback) OnTrainEnd(trainer *Trainer) {
	if c.Verbose {
		fmt.Println("Training completed.")
	}
}

func (c *LoggingCallback) OnEpochBegin(trainer *Trainer, epoch int) {
	if c.Verbose {
		fmt.Printf("\nEpoch %d/%d\n", epoch+1, trainer.config.Epochs)
	}
}

func (c *LoggingCallback) OnEpochEnd(trainer *Trainer, epoch int, metrics *Metrics) {
	if c.Verbose && metrics != nil {
		fmt.Printf("Epoch %d completed | Loss: %.4f | Accuracy: %.2f%%\n",
			epoch+1, metrics.Loss, metrics.Accuracy*100)
	}
}

func (c *LoggingCallback) OnBatchBegin(trainer *Trainer, batch int) {}

func (c *LoggingCallback) OnBatchEnd(trainer *Trainer, batch int, loss float64) {}

// ProgressBarCallback shows progress bar.
type ProgressBarCallback struct {
	width int
}

func NewProgressBarCallback(width int) *ProgressBarCallback {
	return &ProgressBarCallback{width: width}
}

func (c *ProgressBarCallback) OnTrainBegin(trainer *Trainer)                            {}
func (c *ProgressBarCallback) OnTrainEnd(trainer *Trainer)                              { fmt.Println() }
func (c *ProgressBarCallback) OnEpochBegin(trainer *Trainer, epoch int)                 {}
func (c *ProgressBarCallback) OnEpochEnd(trainer *Trainer, epoch int, metrics *Metrics) {}
func (c *ProgressBarCallback) OnBatchBegin(trainer *Trainer, batch int)                 {}

func (c *ProgressBarCallback) OnBatchEnd(trainer *Trainer, batch int, loss float64) {
	progress := float64(batch+1) / float64(trainer.config.StepsPerEpoch)
	filled := int(progress * float64(c.width))
	bar := ""
	for i := 0; i < c.width; i++ {
		if i < filled {
			bar += "█"
		} else {
			bar += "░"
		}
	}
	fmt.Printf("\r[%s] %.1f%% | Loss: %.4f", bar, progress*100, loss)
}

// GenerateTrainingSummary generates a training summary.
func GenerateTrainingSummary(run *TrainingRun) string {
	duration := run.EndedAt.Sub(run.StartedAt)

	summary := fmt.Sprintf(`
=======================================================
            TRAINING RUN SUMMARY
=======================================================

Run ID:      %s
Model:       %s
Status:      %s
Duration:    %s

Configuration:
  Epochs:       %d
  Batch Size:   %d
  Learning Rate: %.6f

Results:
  Total Steps: %d
  Best Epoch:  %d
`,
		run.RunID,
		run.ModelName,
		run.Status,
		duration.String(),
		run.Config.Epochs,
		run.Hyperparams.BatchSize,
		run.Hyperparams.LearningRate,
		run.TotalSteps,
		run.BestEpoch,
	)

	if run.BestMetrics != nil {
		summary += fmt.Sprintf(`
Best Metrics:
  Loss:      %.4f
  Accuracy:  %.2f%%
  F1 Score:  %.4f
`,
			run.BestMetrics.Loss,
			run.BestMetrics.Accuracy*100,
			run.BestMetrics.F1Score,
		)
	}

	if run.FinalMetrics != nil {
		summary += fmt.Sprintf(`
Final Metrics:
  Loss:      %.4f
  Accuracy:  %.2f%%
  F1 Score:  %.4f
`,
			run.FinalMetrics.Loss,
			run.FinalMetrics.Accuracy*100,
			run.FinalMetrics.F1Score,
		)
	}

	if run.BestCheckpoint != "" {
		summary += fmt.Sprintf("\nBest Checkpoint: %s\n", run.BestCheckpoint)
	}

	if run.ErrorMessage != "" {
		summary += fmt.Sprintf("\nError: %s\n", run.ErrorMessage)
	}

	summary += "=======================================================\n"

	return summary
}
