package training

import (
	"fmt"
	"math"
	"sort"
	"time"
)

// Metrics holds all evaluation metrics for a model/training run.
type Metrics struct {
	// Core metrics
	Loss       float64 `json:"loss"`
	Accuracy   float64 `json:"accuracy"`
	Perplexity float64 `json:"perplexity"`

	// Classification metrics
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1Score   float64 `json:"f1_score"`

	// Per-class metrics
	ClassMetrics map[string]*ClassMetrics `json:"class_metrics,omitempty"`

	// Generation metrics
	BLEU      float64 `json:"bleu,omitempty"`
	ROUGE1    float64 `json:"rouge_1,omitempty"`
	ROUGE2    float64 `json:"rouge_2,omitempty"`
	ROUGEL    float64 `json:"rouge_l,omitempty"`
	Coherence float64 `json:"coherence,omitempty"`
	Fluency   float64 `json:"fluency,omitempty"`

	// Retrieval metrics
	MRR      float64         `json:"mrr,omitempty"`       // Mean Reciprocal Rank
	NDCG     float64         `json:"ndcg,omitempty"`      // Normalized Discounted Cumulative Gain
	MAP      float64         `json:"map,omitempty"`       // Mean Average Precision
	RecallAt map[int]float64 `json:"recall_at,omitempty"` // Recall@K

	// Timing
	InferenceTimeMs float64 `json:"inference_time_ms,omitempty"`
	TokensPerSecond float64 `json:"tokens_per_second,omitempty"`

	// Counts
	TotalSamples   int `json:"total_samples"`
	CorrectSamples int `json:"correct_samples"`
	TruePositives  int `json:"true_positives"`
	TrueNegatives  int `json:"true_negatives"`
	FalsePositives int `json:"false_positives"`
	FalseNegatives int `json:"false_negatives"`

	// Custom metrics
	Custom map[string]float64 `json:"custom,omitempty"`
}

// ClassMetrics holds metrics for a single class.
type ClassMetrics struct {
	Class     string  `json:"class"`
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1Score   float64 `json:"f1_score"`
	Support   int     `json:"support"`
	TruePos   int     `json:"true_positives"`
	FalsePos  int     `json:"false_positives"`
	FalseNeg  int     `json:"false_negatives"`
}

// ConfusionMatrix represents a confusion matrix.
type ConfusionMatrix struct {
	Labels []string `json:"labels"`
	Matrix [][]int  `json:"matrix"`
}

// MetricsCalculator calculates various metrics.
type MetricsCalculator struct {
	predictions []Prediction
	threshold   float64
}

// Prediction represents a single prediction with ground truth.
type Prediction struct {
	ID             string             `json:"id"`
	Input          string             `json:"input"`
	Predicted      string             `json:"predicted"`
	Expected       string             `json:"expected"`
	PredictedClass string             `json:"predicted_class,omitempty"`
	ExpectedClass  string             `json:"expected_class,omitempty"`
	Confidence     float64            `json:"confidence"`
	Scores         map[string]float64 `json:"scores,omitempty"` // Class probabilities
	Latency        time.Duration      `json:"latency"`
	Metadata       map[string]any     `json:"metadata,omitempty"`
}

// NewMetricsCalculator creates a new calculator.
func NewMetricsCalculator() *MetricsCalculator {
	return &MetricsCalculator{
		predictions: make([]Prediction, 0),
		threshold:   0.5,
	}
}

// SetThreshold sets the classification threshold.
func (c *MetricsCalculator) SetThreshold(threshold float64) {
	c.threshold = threshold
}

// AddPrediction adds a prediction for evaluation.
func (c *MetricsCalculator) AddPrediction(pred Prediction) {
	c.predictions = append(c.predictions, pred)
}

// AddPredictions adds multiple predictions.
func (c *MetricsCalculator) AddPredictions(preds []Prediction) {
	c.predictions = append(c.predictions, preds...)
}

// Clear clears all predictions.
func (c *MetricsCalculator) Clear() {
	c.predictions = make([]Prediction, 0)
}

// Calculate computes all metrics.
func (c *MetricsCalculator) Calculate() *Metrics {
	if len(c.predictions) == 0 {
		return &Metrics{}
	}

	metrics := &Metrics{
		TotalSamples: len(c.predictions),
		ClassMetrics: make(map[string]*ClassMetrics),
		Custom:       make(map[string]float64),
		RecallAt:     make(map[int]float64),
	}

	// Classification metrics
	c.calculateClassificationMetrics(metrics)

	// Generation metrics (if applicable)
	c.calculateGenerationMetrics(metrics)

	// Timing metrics
	c.calculateTimingMetrics(metrics)

	return metrics
}

func (c *MetricsCalculator) calculateClassificationMetrics(metrics *Metrics) {
	// Collect unique classes
	classSet := make(map[string]bool)
	for _, p := range c.predictions {
		if p.ExpectedClass != "" {
			classSet[p.ExpectedClass] = true
		}
		if p.PredictedClass != "" {
			classSet[p.PredictedClass] = true
		}
	}

	// Calculate per-class metrics
	for class := range classSet {
		var tp, fp, fn int
		for _, p := range c.predictions {
			predicted := p.PredictedClass
			expected := p.ExpectedClass

			if predicted == class && expected == class {
				tp++
			} else if predicted == class && expected != class {
				fp++
			} else if predicted != class && expected == class {
				fn++
			}
		}

		precision := float64(0)
		if tp+fp > 0 {
			precision = float64(tp) / float64(tp+fp)
		}
		recall := float64(0)
		if tp+fn > 0 {
			recall = float64(tp) / float64(tp+fn)
		}
		f1 := float64(0)
		if precision+recall > 0 {
			f1 = 2 * precision * recall / (precision + recall)
		}

		metrics.ClassMetrics[class] = &ClassMetrics{
			Class:     class,
			Precision: precision,
			Recall:    recall,
			F1Score:   f1,
			Support:   tp + fn,
			TruePos:   tp,
			FalsePos:  fp,
			FalseNeg:  fn,
		}

		metrics.TruePositives += tp
		metrics.FalsePositives += fp
		metrics.FalseNegatives += fn
	}

	// Overall accuracy
	correct := 0
	for _, p := range c.predictions {
		if p.PredictedClass == p.ExpectedClass {
			correct++
		}
	}
	metrics.CorrectSamples = correct
	metrics.Accuracy = float64(correct) / float64(len(c.predictions))

	// Macro-averaged precision, recall, F1
	if len(metrics.ClassMetrics) > 0 {
		var totalP, totalR, totalF1 float64
		for _, cm := range metrics.ClassMetrics {
			totalP += cm.Precision
			totalR += cm.Recall
			totalF1 += cm.F1Score
		}
		n := float64(len(metrics.ClassMetrics))
		metrics.Precision = totalP / n
		metrics.Recall = totalR / n
		metrics.F1Score = totalF1 / n
	}
}

func (c *MetricsCalculator) calculateGenerationMetrics(metrics *Metrics) {
	// Simple BLEU approximation (unigram precision)
	var totalBleu float64
	for _, p := range c.predictions {
		if p.Expected == "" || p.Predicted == "" {
			continue
		}
		bleu := c.calculateSimpleBLEU(p.Predicted, p.Expected)
		totalBleu += bleu
	}
	if len(c.predictions) > 0 {
		metrics.BLEU = totalBleu / float64(len(c.predictions))
	}

	// Exact match (useful for structured outputs)
	exactMatches := 0
	for _, p := range c.predictions {
		if p.Predicted == p.Expected {
			exactMatches++
		}
	}
	metrics.Custom["exact_match"] = float64(exactMatches) / float64(len(c.predictions))
}

func (c *MetricsCalculator) calculateSimpleBLEU(predicted, expected string) float64 {
	// Tokenize (simple whitespace split)
	predTokens := tokenize(predicted)
	expTokens := tokenize(expected)

	if len(predTokens) == 0 {
		return 0
	}

	// Count matching unigrams
	expSet := make(map[string]int)
	for _, t := range expTokens {
		expSet[t]++
	}

	matches := 0
	for _, t := range predTokens {
		if expSet[t] > 0 {
			matches++
			expSet[t]--
		}
	}

	precision := float64(matches) / float64(len(predTokens))

	// Brevity penalty
	bp := 1.0
	if len(predTokens) < len(expTokens) {
		bp = math.Exp(1 - float64(len(expTokens))/float64(len(predTokens)))
	}

	return bp * precision
}

func (c *MetricsCalculator) calculateTimingMetrics(metrics *Metrics) {
	var totalLatency time.Duration
	for _, p := range c.predictions {
		totalLatency += p.Latency
	}

	if len(c.predictions) > 0 {
		avgLatency := totalLatency / time.Duration(len(c.predictions))
		metrics.InferenceTimeMs = float64(avgLatency.Milliseconds())
	}
}

// GetConfusionMatrix generates a confusion matrix.
func (c *MetricsCalculator) GetConfusionMatrix() *ConfusionMatrix {
	// Collect labels
	labelSet := make(map[string]bool)
	for _, p := range c.predictions {
		if p.ExpectedClass != "" {
			labelSet[p.ExpectedClass] = true
		}
		if p.PredictedClass != "" {
			labelSet[p.PredictedClass] = true
		}
	}

	labels := make([]string, 0, len(labelSet))
	for l := range labelSet {
		labels = append(labels, l)
	}
	sort.Strings(labels)

	labelIndex := make(map[string]int)
	for i, l := range labels {
		labelIndex[l] = i
	}

	// Build matrix
	n := len(labels)
	matrix := make([][]int, n)
	for i := range matrix {
		matrix[i] = make([]int, n)
	}

	for _, p := range c.predictions {
		if p.ExpectedClass == "" || p.PredictedClass == "" {
			continue
		}
		i := labelIndex[p.ExpectedClass]
		j := labelIndex[p.PredictedClass]
		matrix[i][j]++
	}

	return &ConfusionMatrix{
		Labels: labels,
		Matrix: matrix,
	}
}

// CalculateROC calculates ROC curve points and AUC.
func (c *MetricsCalculator) CalculateROC(positiveClass string) ([]float64, []float64, float64) {
	type scoredPred struct {
		score    float64
		positive bool
	}

	var scored []scoredPred
	for _, p := range c.predictions {
		score := p.Confidence
		if p.Scores != nil && p.Scores[positiveClass] > 0 {
			score = p.Scores[positiveClass]
		}
		scored = append(scored, scoredPred{
			score:    score,
			positive: p.ExpectedClass == positiveClass,
		})
	}

	// Sort by score descending
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	var tpr, fpr []float64
	var totalPos, totalNeg int
	for _, s := range scored {
		if s.positive {
			totalPos++
		} else {
			totalNeg++
		}
	}

	if totalPos == 0 || totalNeg == 0 {
		return nil, nil, 0
	}

	var tp, fp int
	for _, s := range scored {
		if s.positive {
			tp++
		} else {
			fp++
		}
		tpr = append(tpr, float64(tp)/float64(totalPos))
		fpr = append(fpr, float64(fp)/float64(totalNeg))
	}

	// Calculate AUC using trapezoidal rule
	auc := 0.0
	for i := 1; i < len(fpr); i++ {
		auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
	}

	return fpr, tpr, auc
}

// MetricsAggregator aggregates metrics across multiple runs.
type MetricsAggregator struct {
	runs []*Metrics
}

// NewMetricsAggregator creates a new aggregator.
func NewMetricsAggregator() *MetricsAggregator {
	return &MetricsAggregator{
		runs: make([]*Metrics, 0),
	}
}

// Add adds a metrics run.
func (a *MetricsAggregator) Add(m *Metrics) {
	a.runs = append(a.runs, m)
}

// Aggregate computes aggregate statistics.
func (a *MetricsAggregator) Aggregate() *AggregatedMetrics {
	if len(a.runs) == 0 {
		return nil
	}

	agg := &AggregatedMetrics{
		NumRuns: len(a.runs),
	}

	// Collect values for each metric
	var losses, accuracies, f1s []float64
	for _, m := range a.runs {
		losses = append(losses, m.Loss)
		accuracies = append(accuracies, m.Accuracy)
		f1s = append(f1s, m.F1Score)
	}

	agg.Loss = computeStats(losses)
	agg.Accuracy = computeStats(accuracies)
	agg.F1Score = computeStats(f1s)

	return agg
}

// AggregatedMetrics holds aggregate statistics.
type AggregatedMetrics struct {
	NumRuns  int          `json:"num_runs"`
	Loss     *MetricStats `json:"loss"`
	Accuracy *MetricStats `json:"accuracy"`
	F1Score  *MetricStats `json:"f1_score"`
}

// MetricStats holds statistics for a single metric.
type MetricStats struct {
	Mean   float64 `json:"mean"`
	StdDev float64 `json:"std_dev"`
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
	Median float64 `json:"median"`
}

func computeStats(values []float64) *MetricStats {
	if len(values) == 0 {
		return &MetricStats{}
	}

	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)

	var sum float64
	min := sorted[0]
	max := sorted[len(sorted)-1]

	for _, v := range values {
		sum += v
	}
	mean := sum / float64(len(values))

	var varianceSum float64
	for _, v := range values {
		varianceSum += (v - mean) * (v - mean)
	}
	stdDev := math.Sqrt(varianceSum / float64(len(values)))

	median := sorted[len(sorted)/2]

	return &MetricStats{
		Mean:   mean,
		StdDev: stdDev,
		Min:    min,
		Max:    max,
		Median: median,
	}
}

// Helper functions
func tokenize(s string) []string {
	var tokens []string
	var current string
	for _, r := range s {
		if r == ' ' || r == '\t' || r == '\n' {
			if current != "" {
				tokens = append(tokens, current)
				current = ""
			}
		} else {
			current += string(r)
		}
	}
	if current != "" {
		tokens = append(tokens, current)
	}
	return tokens
}

// PrintMetrics formats metrics for display.
func PrintMetrics(m *Metrics) string {
	return fmt.Sprintf(`
Metrics Summary:
================
Loss:       %.4f
Accuracy:   %.2f%%
Precision:  %.4f
Recall:     %.4f
F1 Score:   %.4f
Perplexity: %.2f

Samples:
  Total:    %d
  Correct:  %d
  TP: %d  FP: %d  FN: %d  TN: %d

Generation:
  BLEU:     %.4f
  Inference: %.2fms
`,
		m.Loss,
		m.Accuracy*100,
		m.Precision,
		m.Recall,
		m.F1Score,
		m.Perplexity,
		m.TotalSamples,
		m.CorrectSamples,
		m.TruePositives, m.FalsePositives, m.FalseNegatives, m.TrueNegatives,
		m.BLEU,
		m.InferenceTimeMs,
	)
}
