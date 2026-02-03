package training

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// Evaluator performs model evaluation.
type Evaluator struct {
	config     EvaluationConfig
	calculator *MetricsCalculator
}

// EvaluationConfig configures evaluation.
type EvaluationConfig struct {
	// Batch settings
	BatchSize  int  `json:"batch_size"`
	MaxSamples int  `json:"max_samples"` // 0 = all
	Shuffle    bool `json:"shuffle"`

	// Evaluation types
	EvalClassification bool `json:"eval_classification"`
	EvalGeneration     bool `json:"eval_generation"`
	EvalRetrieval      bool `json:"eval_retrieval"`

	// Output
	OutputDir       string `json:"output_dir"`
	SavePredictions bool   `json:"save_predictions"`

	// Statistical significance
	BootstrapSamples int     `json:"bootstrap_samples"`
	ConfidenceLevel  float64 `json:"confidence_level"`

	// Thresholds
	ClassificationThreshold float64 `json:"classification_threshold"`
}

// DefaultEvaluationConfig returns default configuration.
func DefaultEvaluationConfig() EvaluationConfig {
	return EvaluationConfig{
		BatchSize:               32,
		MaxSamples:              0,
		Shuffle:                 false,
		EvalClassification:      true,
		EvalGeneration:          true,
		EvalRetrieval:           false,
		OutputDir:               "eval_output",
		SavePredictions:         true,
		BootstrapSamples:        1000,
		ConfidenceLevel:         0.95,
		ClassificationThreshold: 0.5,
	}
}

// NewEvaluator creates a new evaluator.
func NewEvaluator(config EvaluationConfig) *Evaluator {
	calc := NewMetricsCalculator()
	calc.SetThreshold(config.ClassificationThreshold)

	return &Evaluator{
		config:     config,
		calculator: calc,
	}
}

// EvaluationResult holds evaluation results.
type EvaluationResult struct {
	// Identifiers
	ModelID      string    `json:"model_id"`
	DatasetID    string    `json:"dataset_id"`
	EvaluatedAt  time.Time `json:"evaluated_at"`
	DurationSecs float64   `json:"duration_secs"`

	// Metrics
	Metrics *Metrics `json:"metrics"`

	// Detailed results
	ConfusionMatrix *ConfusionMatrix `json:"confusion_matrix,omitempty"`
	ClassReports    []*ClassReport   `json:"class_reports,omitempty"`

	// Statistical analysis
	ConfidenceIntervals map[string]*ConfidenceInterval `json:"confidence_intervals,omitempty"`

	// Predictions (if saved)
	PredictionsFile string `json:"predictions_file,omitempty"`

	// Errors
	Errors []string `json:"errors,omitempty"`
}

// ClassReport detailed report for each class.
type ClassReport struct {
	Class     string         `json:"class"`
	Precision float64        `json:"precision"`
	Recall    float64        `json:"recall"`
	F1Score   float64        `json:"f1_score"`
	Support   int            `json:"support"`
	ErrorRate float64        `json:"error_rate"`
	TopErrors []ErrorExample `json:"top_errors,omitempty"`
}

// ErrorExample represents a misclassified example.
type ErrorExample struct {
	ID        string `json:"id"`
	Input     string `json:"input"`
	Predicted string `json:"predicted"`
	Expected  string `json:"expected"`
}

// ConfidenceInterval represents a confidence interval.
type ConfidenceInterval struct {
	Lower float64 `json:"lower"`
	Upper float64 `json:"upper"`
	Level float64 `json:"level"`
}

// EvalPredictionFunc is called to get a prediction for an input.
type EvalPredictionFunc func(input string) (Prediction, error)

// Evaluate runs evaluation on a dataset.
func (e *Evaluator) Evaluate(dataPoints []DataPoint, predictFn EvalPredictionFunc, modelID, datasetID string) (*EvaluationResult, error) {
	start := time.Now()

	result := &EvaluationResult{
		ModelID:             modelID,
		DatasetID:           datasetID,
		EvaluatedAt:         start,
		ConfidenceIntervals: make(map[string]*ConfidenceInterval),
		Errors:              make([]string, 0),
	}

	// Limit samples if configured
	samples := dataPoints
	if e.config.MaxSamples > 0 && len(samples) > e.config.MaxSamples {
		samples = samples[:e.config.MaxSamples]
	}

	// Run predictions
	predictions := make([]Prediction, 0, len(samples))
	for i, dp := range samples {
		predStart := time.Now()
		pred, err := predictFn(dp.Input)
		pred.Latency = time.Since(predStart)

		if err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("sample %d: %v", i, err))
			continue
		}

		pred.ID = fmt.Sprintf("eval_%d", i)
		pred.Expected = dp.Output
		pred.ExpectedClass = dp.Label

		predictions = append(predictions, pred)
	}

	e.calculator.Clear()
	e.calculator.AddPredictions(predictions)

	// Calculate metrics
	result.Metrics = e.calculator.Calculate()
	result.ConfusionMatrix = e.calculator.GetConfusionMatrix()
	result.DurationSecs = time.Since(start).Seconds()

	// Generate class reports
	result.ClassReports = e.generateClassReports(predictions)

	// Calculate confidence intervals using bootstrap
	if e.config.BootstrapSamples > 0 {
		result.ConfidenceIntervals = e.bootstrapConfidenceIntervals(predictions)
	}

	// Save predictions if configured
	if e.config.SavePredictions && e.config.OutputDir != "" {
		predFile, err := e.savePredictions(predictions, modelID, datasetID)
		if err == nil {
			result.PredictionsFile = predFile
		} else {
			result.Errors = append(result.Errors, fmt.Sprintf("failed to save predictions: %v", err))
		}
	}

	return result, nil
}

func (e *Evaluator) generateClassReports(predictions []Prediction) []*ClassReport {
	classStats := make(map[string]*struct {
		tp, fp, fn int
		errors     []ErrorExample
	})

	for _, p := range predictions {
		expected := p.ExpectedClass
		predicted := p.PredictedClass

		if classStats[expected] == nil {
			classStats[expected] = &struct {
				tp, fp, fn int
				errors     []ErrorExample
			}{}
		}

		if predicted == expected {
			classStats[expected].tp++
		} else {
			classStats[expected].fn++
			if len(classStats[expected].errors) < 5 {
				classStats[expected].errors = append(classStats[expected].errors, ErrorExample{
					ID:        p.ID,
					Input:     p.Input,
					Predicted: predicted,
					Expected:  expected,
				})
			}
		}
	}

	// Count false positives
	for _, p := range predictions {
		if p.PredictedClass != p.ExpectedClass {
			if classStats[p.PredictedClass] == nil {
				classStats[p.PredictedClass] = &struct {
					tp, fp, fn int
					errors     []ErrorExample
				}{}
			}
			classStats[p.PredictedClass].fp++
		}
	}

	var reports []*ClassReport
	for class, stats := range classStats {
		precision := 0.0
		if stats.tp+stats.fp > 0 {
			precision = float64(stats.tp) / float64(stats.tp+stats.fp)
		}
		recall := 0.0
		if stats.tp+stats.fn > 0 {
			recall = float64(stats.tp) / float64(stats.tp+stats.fn)
		}
		f1 := 0.0
		if precision+recall > 0 {
			f1 = 2 * precision * recall / (precision + recall)
		}

		support := stats.tp + stats.fn
		errorRate := 0.0
		if support > 0 {
			errorRate = float64(stats.fn) / float64(support)
		}

		reports = append(reports, &ClassReport{
			Class:     class,
			Precision: precision,
			Recall:    recall,
			F1Score:   f1,
			Support:   support,
			ErrorRate: errorRate,
			TopErrors: stats.errors,
		})
	}

	sort.Slice(reports, func(i, j int) bool {
		return reports[i].Class < reports[j].Class
	})

	return reports
}

func (e *Evaluator) bootstrapConfidenceIntervals(predictions []Prediction) map[string]*ConfidenceInterval {
	n := len(predictions)
	if n == 0 {
		return nil
	}

	nBootstrap := e.config.BootstrapSamples
	accuracies := make([]float64, nBootstrap)

	for b := 0; b < nBootstrap; b++ {
		// Bootstrap sample
		sample := make([]Prediction, n)
		for i := range sample {
			idx := int(float64(n) * pseudoRandom(int64(b*1000+i)))
			if idx >= n {
				idx = n - 1
			}
			sample[i] = predictions[idx]
		}

		// Calculate accuracy for sample
		correct := 0
		for _, p := range sample {
			if p.PredictedClass == p.ExpectedClass {
				correct++
			}
		}
		accuracies[b] = float64(correct) / float64(n)
	}

	sort.Float64s(accuracies)

	alpha := 1 - e.config.ConfidenceLevel
	lowerIdx := int(float64(nBootstrap) * alpha / 2)
	upperIdx := int(float64(nBootstrap) * (1 - alpha/2))

	if lowerIdx < 0 {
		lowerIdx = 0
	}
	if upperIdx >= nBootstrap {
		upperIdx = nBootstrap - 1
	}

	return map[string]*ConfidenceInterval{
		"accuracy": {
			Lower: accuracies[lowerIdx],
			Upper: accuracies[upperIdx],
			Level: e.config.ConfidenceLevel,
		},
	}
}

// Simple pseudo-random using linear congruential generator
func pseudoRandom(seed int64) float64 {
	a := int64(1103515245)
	c := int64(12345)
	m := int64(1 << 31)
	seed = (a*seed + c) % m
	return float64(seed) / float64(m)
}

func (e *Evaluator) savePredictions(predictions []Prediction, modelID, datasetID string) (string, error) {
	if err := os.MkdirAll(e.config.OutputDir, 0755); err != nil {
		return "", err
	}

	filename := fmt.Sprintf("predictions_%s_%s_%s.json",
		modelID, datasetID, time.Now().Format("20060102_150405"))
	path := filepath.Join(e.config.OutputDir, filename)

	data, err := json.MarshalIndent(predictions, "", "  ")
	if err != nil {
		return "", err
	}

	return path, os.WriteFile(path, data, 0644)
}

// ABTestResult holds A/B test results.
type ABTestResult struct {
	ModelA      string  `json:"model_a"`
	ModelB      string  `json:"model_b"`
	MetricName  string  `json:"metric_name"`
	ScoreA      float64 `json:"score_a"`
	ScoreB      float64 `json:"score_b"`
	Difference  float64 `json:"difference"` // B - A
	PValue      float64 `json:"p_value"`
	Significant bool    `json:"significant"`
	Winner      string  `json:"winner"`
}

// CompareModels performs A/B comparison between two models.
func CompareModels(resultA, resultB *EvaluationResult, metricName string, alpha float64) *ABTestResult {
	var scoreA, scoreB float64

	switch metricName {
	case "accuracy":
		scoreA = resultA.Metrics.Accuracy
		scoreB = resultB.Metrics.Accuracy
	case "f1_score":
		scoreA = resultA.Metrics.F1Score
		scoreB = resultB.Metrics.F1Score
	case "precision":
		scoreA = resultA.Metrics.Precision
		scoreB = resultB.Metrics.Precision
	case "recall":
		scoreA = resultA.Metrics.Recall
		scoreB = resultB.Metrics.Recall
	default:
		scoreA = resultA.Metrics.Custom[metricName]
		scoreB = resultB.Metrics.Custom[metricName]
	}

	diff := scoreB - scoreA

	// Approximate p-value using McNemar's test approximation
	nA := resultA.Metrics.TotalSamples
	nB := resultB.Metrics.TotalSamples
	n := (nA + nB) / 2 // Average sample size

	se := math.Sqrt((scoreA * (1 - scoreA) / float64(n)) + (scoreB * (1 - scoreB) / float64(n)))
	if se == 0 {
		se = 0.001
	}

	z := math.Abs(diff) / se
	pValue := 2 * (1 - normalCDF(z)) // Two-tailed

	significant := pValue < alpha

	winner := ""
	if significant {
		if diff > 0 {
			winner = resultB.ModelID
		} else {
			winner = resultA.ModelID
		}
	}

	return &ABTestResult{
		ModelA:      resultA.ModelID,
		ModelB:      resultB.ModelID,
		MetricName:  metricName,
		ScoreA:      scoreA,
		ScoreB:      scoreB,
		Difference:  diff,
		PValue:      pValue,
		Significant: significant,
		Winner:      winner,
	}
}

// Approximate normal CDF
func normalCDF(x float64) float64 {
	return 0.5 * (1 + math.Erf(x/math.Sqrt(2)))
}

// EvaluationReport generates a human-readable report.
func GenerateEvaluationReport(result *EvaluationResult) string {
	report := fmt.Sprintf(`
=======================================================
             MODEL EVALUATION REPORT
=======================================================

Model:    %s
Dataset:  %s
Date:     %s
Duration: %.2f seconds

-------------------------------------------------------
                  OVERALL METRICS
-------------------------------------------------------
Accuracy:   %.2f%% (%d/%d correct)
Precision:  %.4f
Recall:     %.4f
F1 Score:   %.4f
Loss:       %.4f

`,
		result.ModelID,
		result.DatasetID,
		result.EvaluatedAt.Format("2006-01-02 15:04:05"),
		result.DurationSecs,
		result.Metrics.Accuracy*100,
		result.Metrics.CorrectSamples,
		result.Metrics.TotalSamples,
		result.Metrics.Precision,
		result.Metrics.Recall,
		result.Metrics.F1Score,
		result.Metrics.Loss,
	)

	if len(result.ConfidenceIntervals) > 0 {
		report += "-------------------------------------------------------\n"
		report += "              CONFIDENCE INTERVALS\n"
		report += "-------------------------------------------------------\n"
		for metric, ci := range result.ConfidenceIntervals {
			report += fmt.Sprintf("%-12s: [%.4f, %.4f] (%.0f%% CI)\n",
				metric, ci.Lower, ci.Upper, ci.Level*100)
		}
		report += "\n"
	}

	if len(result.ClassReports) > 0 {
		report += "-------------------------------------------------------\n"
		report += "                PER-CLASS METRICS\n"
		report += "-------------------------------------------------------\n"
		report += fmt.Sprintf("%-15s %8s %8s %8s %8s\n", "Class", "Prec", "Recall", "F1", "Support")
		report += "-------------------------------------------------------\n"
		for _, cr := range result.ClassReports {
			report += fmt.Sprintf("%-15s %8.4f %8.4f %8.4f %8d\n",
				cr.Class, cr.Precision, cr.Recall, cr.F1Score, cr.Support)
		}
		report += "\n"
	}

	if result.ConfusionMatrix != nil && len(result.ConfusionMatrix.Labels) > 0 {
		report += "-------------------------------------------------------\n"
		report += "                CONFUSION MATRIX\n"
		report += "-------------------------------------------------------\n"
		report += "Predicted →\n"
		report += "Actual ↓\n\n"

		// Header
		report += fmt.Sprintf("%12s", "")
		for _, l := range result.ConfusionMatrix.Labels {
			label := l
			if len(label) > 8 {
				label = label[:8]
			}
			report += fmt.Sprintf("%8s", label)
		}
		report += "\n"

		// Rows
		for i, row := range result.ConfusionMatrix.Matrix {
			label := result.ConfusionMatrix.Labels[i]
			if len(label) > 12 {
				label = label[:12]
			}
			report += fmt.Sprintf("%-12s", label)
			for _, val := range row {
				report += fmt.Sprintf("%8d", val)
			}
			report += "\n"
		}
		report += "\n"
	}

	if len(result.Errors) > 0 {
		report += "-------------------------------------------------------\n"
		report += "                    ERRORS\n"
		report += "-------------------------------------------------------\n"
		for _, err := range result.Errors {
			report += fmt.Sprintf("• %s\n", err)
		}
		report += "\n"
	}

	report += "=======================================================\n"

	return report
}
