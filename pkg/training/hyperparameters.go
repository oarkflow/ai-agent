package training

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"time"
)

// Hyperparameters defines all tunable training parameters.
type Hyperparameters struct {
	// Learning parameters
	LearningRate     float64 `json:"learning_rate"`
	LearningRateMin  float64 `json:"learning_rate_min"`
	LearningRateMax  float64 `json:"learning_rate_max"`
	WarmupSteps      int     `json:"warmup_steps"`
	WeightDecay      float64 `json:"weight_decay"`
	GradientClipping float64 `json:"gradient_clipping"`

	// Batch settings
	BatchSize          int `json:"batch_size"`
	GradientAccumSteps int `json:"gradient_accumulation_steps"`
	EffectiveBatchSize int `json:"effective_batch_size"` // BatchSize * GradientAccumSteps

	// Training schedule
	Epochs             int     `json:"epochs"`
	MaxSteps           int     `json:"max_steps"`
	EvalSteps          int     `json:"eval_steps"`
	SaveSteps          int     `json:"save_steps"`
	LoggingSteps       int     `json:"logging_steps"`
	EarlyStoppingPat   int     `json:"early_stopping_patience"`
	EarlyStoppingDelta float64 `json:"early_stopping_delta"`

	// Model-specific
	Temperature     float64 `json:"temperature"`
	TopP            float64 `json:"top_p"`
	TopK            int     `json:"top_k"`
	MaxTokens       int     `json:"max_tokens"`
	ContextLength   int     `json:"context_length"`
	FrequencyPenalty float64 `json:"frequency_penalty"`
	PresencePenalty  float64 `json:"presence_penalty"`

	// Regularization
	Dropout       float64 `json:"dropout"`
	LabelSmoothing float64 `json:"label_smoothing"`

	// Few-shot specific
	NumFewShot      int  `json:"num_few_shot"`
	FewShotStrategy string `json:"few_shot_strategy"` // random, similar, diverse

	// Fine-tuning specific
	LoraRank       int     `json:"lora_rank,omitempty"`
	LoraAlpha      float64 `json:"lora_alpha,omitempty"`
	LoraDropout    float64 `json:"lora_dropout,omitempty"`

	// Custom parameters
	Custom map[string]any `json:"custom,omitempty"`
}

// DefaultHyperparameters returns sensible defaults.
func DefaultHyperparameters() *Hyperparameters {
	return &Hyperparameters{
		LearningRate:       3e-4,
		LearningRateMin:    1e-6,
		LearningRateMax:    1e-3,
		WarmupSteps:        100,
		WeightDecay:        0.01,
		GradientClipping:   1.0,
		BatchSize:          8,
		GradientAccumSteps: 4,
		EffectiveBatchSize: 32,
		Epochs:             3,
		MaxSteps:           -1,
		EvalSteps:          100,
		SaveSteps:          500,
		LoggingSteps:       10,
		EarlyStoppingPat:   3,
		EarlyStoppingDelta: 0.001,
		Temperature:        0.7,
		TopP:               0.9,
		TopK:               50,
		MaxTokens:          2048,
		ContextLength:      4096,
		FrequencyPenalty:   0.0,
		PresencePenalty:    0.0,
		Dropout:            0.1,
		LabelSmoothing:     0.0,
		NumFewShot:         3,
		FewShotStrategy:    "similar",
		LoraRank:           8,
		LoraAlpha:          16,
		LoraDropout:        0.05,
		Custom:             make(map[string]any),
	}
}

// HyperparameterRange defines a range for hyperparameter search.
type HyperparameterRange struct {
	Name   string    `json:"name"`
	Type   string    `json:"type"` // float, int, choice
	Min    float64   `json:"min,omitempty"`
	Max    float64   `json:"max,omitempty"`
	Step   float64   `json:"step,omitempty"`
	Log    bool      `json:"log,omitempty"` // Log scale for search
	Values []any     `json:"values,omitempty"` // For choice type
}

// HyperparameterSearchConfig defines search configuration.
type HyperparameterSearchConfig struct {
	Strategy      string                 `json:"strategy"` // grid, random, bayesian
	MaxTrials     int                    `json:"max_trials"`
	MaxTime       time.Duration          `json:"max_time"`
	Metric        string                 `json:"metric"` // loss, accuracy, f1, etc.
	Direction     string                 `json:"direction"` // minimize, maximize
	Ranges        []*HyperparameterRange `json:"ranges"`
	Seed          int64                  `json:"seed"`
}

// HyperparameterTrial represents one trial in hyperparameter search.
type HyperparameterTrial struct {
	ID           int                `json:"id"`
	Params       *Hyperparameters   `json:"params"`
	Metrics      *Metrics           `json:"metrics"`
	Status       string             `json:"status"` // pending, running, completed, failed
	StartTime    time.Time          `json:"start_time"`
	EndTime      time.Time          `json:"end_time"`
	Duration     time.Duration      `json:"duration"`
}

// HyperparameterSearch manages hyperparameter optimization.
type HyperparameterSearch struct {
	config  *HyperparameterSearchConfig
	trials  []*HyperparameterTrial
	best    *HyperparameterTrial
}

// NewHyperparameterSearch creates a new search.
func NewHyperparameterSearch(config *HyperparameterSearchConfig) *HyperparameterSearch {
	return &HyperparameterSearch{
		config: config,
		trials: make([]*HyperparameterTrial, 0),
	}
}

// GenerateTrials generates trial configurations based on search strategy.
func (h *HyperparameterSearch) GenerateTrials(baseParams *Hyperparameters) []*Hyperparameters {
	switch h.config.Strategy {
	case "grid":
		return h.gridSearch(baseParams)
	case "random":
		return h.randomSearch(baseParams)
	default:
		return []*Hyperparameters{baseParams}
	}
}

func (h *HyperparameterSearch) gridSearch(base *Hyperparameters) []*Hyperparameters {
	// Generate all combinations
	combinations := []map[string]any{{}}

	for _, r := range h.config.Ranges {
		var values []any
		switch r.Type {
		case "float":
			for v := r.Min; v <= r.Max; v += r.Step {
				values = append(values, v)
			}
		case "int":
			for v := int(r.Min); v <= int(r.Max); v += int(r.Step) {
				values = append(values, v)
			}
		case "choice":
			values = r.Values
		}

		var newCombos []map[string]any
		for _, combo := range combinations {
			for _, v := range values {
				newCombo := make(map[string]any)
				for k, val := range combo {
					newCombo[k] = val
				}
				newCombo[r.Name] = v
				newCombos = append(newCombos, newCombo)
			}
		}
		combinations = newCombos

		// Limit combinations
		if len(combinations) > h.config.MaxTrials {
			combinations = combinations[:h.config.MaxTrials]
			break
		}
	}

	var params []*Hyperparameters
	for _, combo := range combinations {
		p := h.applyParams(base, combo)
		params = append(params, p)
	}

	return params
}

func (h *HyperparameterSearch) randomSearch(base *Hyperparameters) []*Hyperparameters {
	var params []*Hyperparameters

	for i := 0; i < h.config.MaxTrials; i++ {
		combo := make(map[string]any)
		for _, r := range h.config.Ranges {
			switch r.Type {
			case "float":
				if r.Log {
					logMin := math.Log(r.Min)
					logMax := math.Log(r.Max)
					combo[r.Name] = math.Exp(logMin + randFloat()*(logMax-logMin))
				} else {
					combo[r.Name] = r.Min + randFloat()*(r.Max-r.Min)
				}
			case "int":
				combo[r.Name] = int(r.Min) + randInt(int(r.Max-r.Min+1))
			case "choice":
				combo[r.Name] = r.Values[randInt(len(r.Values))]
			}
		}
		params = append(params, h.applyParams(base, combo))
	}

	return params
}

func (h *HyperparameterSearch) applyParams(base *Hyperparameters, updates map[string]any) *Hyperparameters {
	// Clone base
	data, _ := json.Marshal(base)
	var result Hyperparameters
	json.Unmarshal(data, &result)

	// Apply updates via reflection-like approach using JSON
	updateData, _ := json.Marshal(updates)
	json.Unmarshal(updateData, &result)

	return &result
}

// RecordTrial records the result of a trial.
func (h *HyperparameterSearch) RecordTrial(trial *HyperparameterTrial) {
	trial.EndTime = time.Now()
	trial.Duration = trial.EndTime.Sub(trial.StartTime)
	trial.Status = "completed"
	h.trials = append(h.trials, trial)

	// Update best if applicable
	if h.best == nil || h.isBetter(trial, h.best) {
		h.best = trial
	}
}

func (h *HyperparameterSearch) isBetter(a, b *HyperparameterTrial) bool {
	if a.Metrics == nil || b.Metrics == nil {
		return false
	}

	var aVal, bVal float64
	switch h.config.Metric {
	case "loss":
		aVal = a.Metrics.Loss
		bVal = b.Metrics.Loss
	case "accuracy":
		aVal = a.Metrics.Accuracy
		bVal = b.Metrics.Accuracy
	case "f1":
		aVal = a.Metrics.F1Score
		bVal = b.Metrics.F1Score
	default:
		return false
	}

	if h.config.Direction == "minimize" {
		return aVal < bVal
	}
	return aVal > bVal
}

// GetBestTrial returns the best trial so far.
func (h *HyperparameterSearch) GetBestTrial() *HyperparameterTrial {
	return h.best
}

// GetTrials returns all trials.
func (h *HyperparameterSearch) GetTrials() []*HyperparameterTrial {
	return h.trials
}

// GetTrialsSorted returns trials sorted by metric.
func (h *HyperparameterSearch) GetTrialsSorted() []*HyperparameterTrial {
	sorted := make([]*HyperparameterTrial, len(h.trials))
	copy(sorted, h.trials)

	sort.Slice(sorted, func(i, j int) bool {
		return h.isBetter(sorted[i], sorted[j])
	})

	return sorted
}

// Helper random functions
func randFloat() float64 {
	return float64(time.Now().UnixNano()%1000) / 1000.0
}

func randInt(n int) int {
	if n <= 0 {
		return 0
	}
	return int(time.Now().UnixNano()) % n
}

// CommonSearchConfigs provides preset search configurations.
var CommonSearchConfigs = map[string]*HyperparameterSearchConfig{
	"quick": {
		Strategy:  "random",
		MaxTrials: 10,
		Metric:    "loss",
		Direction: "minimize",
		Ranges: []*HyperparameterRange{
			{Name: "learning_rate", Type: "float", Min: 1e-5, Max: 1e-3, Log: true},
			{Name: "batch_size", Type: "choice", Values: []any{4, 8, 16}},
		},
	},
	"thorough": {
		Strategy:  "random",
		MaxTrials: 50,
		Metric:    "loss",
		Direction: "minimize",
		Ranges: []*HyperparameterRange{
			{Name: "learning_rate", Type: "float", Min: 1e-6, Max: 1e-2, Log: true},
			{Name: "batch_size", Type: "choice", Values: []any{2, 4, 8, 16, 32}},
			{Name: "warmup_steps", Type: "int", Min: 0, Max: 500, Step: 50},
			{Name: "weight_decay", Type: "float", Min: 0.0, Max: 0.1, Step: 0.01},
			{Name: "dropout", Type: "float", Min: 0.0, Max: 0.5, Step: 0.1},
		},
	},
	"fewshot": {
		Strategy:  "grid",
		MaxTrials: 20,
		Metric:    "accuracy",
		Direction: "maximize",
		Ranges: []*HyperparameterRange{
			{Name: "num_few_shot", Type: "choice", Values: []any{1, 2, 3, 5, 10}},
			{Name: "temperature", Type: "choice", Values: []any{0.0, 0.3, 0.5, 0.7, 1.0}},
			{Name: "few_shot_strategy", Type: "choice", Values: []any{"random", "similar", "diverse"}},
		},
	},
}

// HyperparameterScheduler manages learning rate and other param schedules.
type HyperparameterScheduler struct {
	hp          *Hyperparameters
	currentStep int
	totalSteps  int
}

// NewHyperparameterScheduler creates a scheduler.
func NewHyperparameterScheduler(hp *Hyperparameters, totalSteps int) *HyperparameterScheduler {
	return &HyperparameterScheduler{
		hp:         hp,
		totalSteps: totalSteps,
	}
}

// Step advances the scheduler and returns current learning rate.
func (s *HyperparameterScheduler) Step() float64 {
	s.currentStep++
	return s.GetLearningRate()
}

// GetLearningRate returns the current learning rate with warmup and decay.
func (s *HyperparameterScheduler) GetLearningRate() float64 {
	if s.currentStep < s.hp.WarmupSteps {
		// Linear warmup
		return s.hp.LearningRate * float64(s.currentStep) / float64(s.hp.WarmupSteps)
	}

	// Cosine decay after warmup
	progress := float64(s.currentStep-s.hp.WarmupSteps) / float64(s.totalSteps-s.hp.WarmupSteps)
	decay := 0.5 * (1.0 + math.Cos(math.Pi*progress))
	return s.hp.LearningRateMin + (s.hp.LearningRate-s.hp.LearningRateMin)*decay
}

// GetCurrentStep returns the current step.
func (s *HyperparameterScheduler) GetCurrentStep() int {
	return s.currentStep
}

// Export exports hyperparameters to JSON.
func (hp *Hyperparameters) Export() (string, error) {
	data, err := json.MarshalIndent(hp, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// Import imports hyperparameters from JSON.
func ImportHyperparameters(jsonStr string) (*Hyperparameters, error) {
	var hp Hyperparameters
	if err := json.Unmarshal([]byte(jsonStr), &hp); err != nil {
		return nil, err
	}
	return &hp, nil
}

// Merge merges two hyperparameter sets.
func (hp *Hyperparameters) Merge(other *Hyperparameters) *Hyperparameters {
	result := *hp
	if other.LearningRate != 0 {
		result.LearningRate = other.LearningRate
	}
	if other.BatchSize != 0 {
		result.BatchSize = other.BatchSize
	}
	if other.Epochs != 0 {
		result.Epochs = other.Epochs
	}
	if other.Temperature != 0 {
		result.Temperature = other.Temperature
	}
	if other.MaxTokens != 0 {
		result.MaxTokens = other.MaxTokens
	}
	// Merge custom
	for k, v := range other.Custom {
		result.Custom[k] = v
	}
	return &result
}

// Validate validates hyperparameters.
func (hp *Hyperparameters) Validate() error {
	if hp.LearningRate <= 0 {
		return fmt.Errorf("learning_rate must be positive")
	}
	if hp.BatchSize <= 0 {
		return fmt.Errorf("batch_size must be positive")
	}
	if hp.Temperature < 0 || hp.Temperature > 2 {
		return fmt.Errorf("temperature must be between 0 and 2")
	}
	if hp.TopP < 0 || hp.TopP > 1 {
		return fmt.Errorf("top_p must be between 0 and 1")
	}
	if hp.Dropout < 0 || hp.Dropout > 1 {
		return fmt.Errorf("dropout must be between 0 and 1")
	}
	return nil
}
