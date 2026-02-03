package training

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"regexp"
	"strings"
	"time"
	"unicode"
)

// DatasetSplit defines train/test/validation split ratios.
type DatasetSplit struct {
	Train      float64 `json:"train"`      // e.g., 0.8
	Validation float64 `json:"validation"` // e.g., 0.1
	Test       float64 `json:"test"`       // e.g., 0.1
}

// Dataset represents a training dataset with splits.
type Dataset struct {
	ID          string         `json:"id"`
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Domain      string         `json:"domain"`
	Version     string         `json:"version"`
	Split       DatasetSplit   `json:"split"`
	Train       []*DataPoint   `json:"train"`
	Validation  []*DataPoint   `json:"validation"`
	Test        []*DataPoint   `json:"test"`
	Stats       *DatasetStats  `json:"stats"`
	Metadata    map[string]any `json:"metadata,omitempty"`
	CreatedAt   time.Time      `json:"created_at"`
	UpdatedAt   time.Time      `json:"updated_at"`
}

// DataPoint represents a single training/test data point.
type DataPoint struct {
	ID           string         `json:"id"`
	Input        string         `json:"input"`
	Output       string         `json:"output"`
	Label        string         `json:"label,omitempty"`
	InputTokens  int            `json:"input_tokens,omitempty"`
	OutputTokens int            `json:"output_tokens,omitempty"`
	Category     string         `json:"category,omitempty"`
	Weight       float64        `json:"weight"`
	Quality      float64        `json:"quality"`
	IsValid      bool           `json:"is_valid"`
	Metadata     map[string]any `json:"metadata,omitempty"`
}

// DatasetStats holds statistics about the dataset.
type DatasetStats struct {
	TotalSamples      int                `json:"total_samples"`
	TrainSamples      int                `json:"train_samples"`
	ValidationSamples int                `json:"validation_samples"`
	TestSamples       int                `json:"test_samples"`
	Categories        map[string]int     `json:"categories"`
	AvgInputLength    float64            `json:"avg_input_length"`
	AvgOutputLength   float64            `json:"avg_output_length"`
	QualityDistribution map[string]int   `json:"quality_distribution"`
	WeightStats       *WeightStats       `json:"weight_stats"`
}

// WeightStats holds weight statistics.
type WeightStats struct {
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
	Mean   float64 `json:"mean"`
	StdDev float64 `json:"std_dev"`
}

// CleaningConfig defines data cleaning parameters.
type CleaningConfig struct {
	RemoveEmptyInputs      bool     `json:"remove_empty_inputs"`
	RemoveEmptyOutputs     bool     `json:"remove_empty_outputs"`
	RemoveDuplicates       bool     `json:"remove_duplicates"`
	TrimWhitespace         bool     `json:"trim_whitespace"`
	NormalizeUnicode       bool     `json:"normalize_unicode"`
	RemoveHTMLTags         bool     `json:"remove_html_tags"`
	MinInputLength         int      `json:"min_input_length"`
	MaxInputLength         int      `json:"max_input_length"`
	MinOutputLength        int      `json:"min_output_length"`
	MaxOutputLength        int      `json:"max_output_length"`
	MinQuality             float64  `json:"min_quality"`
	AllowedCategories      []string `json:"allowed_categories,omitempty"`
	CustomFilters          []string `json:"custom_filters,omitempty"` // Regex patterns to remove
}

// DefaultCleaningConfig returns sensible defaults.
func DefaultCleaningConfig() *CleaningConfig {
	return &CleaningConfig{
		RemoveEmptyInputs:  true,
		RemoveEmptyOutputs: true,
		RemoveDuplicates:   true,
		TrimWhitespace:     true,
		NormalizeUnicode:   true,
		RemoveHTMLTags:     false,
		MinInputLength:     5,
		MaxInputLength:     100000,
		MinOutputLength:    1,
		MaxOutputLength:    100000,
		MinQuality:         0.0,
	}
}

// DataCleaner handles data preprocessing and cleaning.
type DataCleaner struct {
	config *CleaningConfig
	stats  *CleaningStats
}

// CleaningStats tracks cleaning operations.
type CleaningStats struct {
	OriginalCount   int            `json:"original_count"`
	CleanedCount    int            `json:"cleaned_count"`
	RemovedCount    int            `json:"removed_count"`
	RemovedReasons  map[string]int `json:"removed_reasons"`
	DuplicatesFound int            `json:"duplicates_found"`
}

// NewDataCleaner creates a new data cleaner.
func NewDataCleaner(config *CleaningConfig) *DataCleaner {
	if config == nil {
		config = DefaultCleaningConfig()
	}
	return &DataCleaner{
		config: config,
		stats: &CleaningStats{
			RemovedReasons: make(map[string]int),
		},
	}
}

// Clean processes and cleans a slice of data points.
func (c *DataCleaner) Clean(data []*DataPoint) ([]*DataPoint, *CleaningStats) {
	c.stats = &CleaningStats{
		OriginalCount:  len(data),
		RemovedReasons: make(map[string]int),
	}

	var cleaned []*DataPoint
	seen := make(map[string]bool)

	for _, dp := range data {
		// Apply cleaning transformations
		dp = c.transform(dp)

		// Validate
		valid, reason := c.validate(dp)
		if !valid {
			c.stats.RemovedReasons[reason]++
			continue
		}

		// Check duplicates
		if c.config.RemoveDuplicates {
			hash := c.hash(dp)
			if seen[hash] {
				c.stats.DuplicatesFound++
				c.stats.RemovedReasons["duplicate"]++
				continue
			}
			seen[hash] = true
		}

		dp.IsValid = true
		cleaned = append(cleaned, dp)
	}

	c.stats.CleanedCount = len(cleaned)
	c.stats.RemovedCount = c.stats.OriginalCount - c.stats.CleanedCount

	return cleaned, c.stats
}

func (c *DataCleaner) transform(dp *DataPoint) *DataPoint {
	result := &DataPoint{
		ID:       dp.ID,
		Input:    dp.Input,
		Output:   dp.Output,
		Category: dp.Category,
		Weight:   dp.Weight,
		Quality:  dp.Quality,
		Metadata: dp.Metadata,
	}

	if c.config.TrimWhitespace {
		result.Input = strings.TrimSpace(result.Input)
		result.Output = strings.TrimSpace(result.Output)
	}

	if c.config.NormalizeUnicode {
		result.Input = normalizeUnicode(result.Input)
		result.Output = normalizeUnicode(result.Output)
	}

	if c.config.RemoveHTMLTags {
		result.Input = removeHTMLTags(result.Input)
		result.Output = removeHTMLTags(result.Output)
	}

	// Apply custom filters
	for _, pattern := range c.config.CustomFilters {
		if re, err := regexp.Compile(pattern); err == nil {
			result.Input = re.ReplaceAllString(result.Input, "")
			result.Output = re.ReplaceAllString(result.Output, "")
		}
	}

	return result
}

func (c *DataCleaner) validate(dp *DataPoint) (bool, string) {
	if c.config.RemoveEmptyInputs && len(dp.Input) == 0 {
		return false, "empty_input"
	}
	if c.config.RemoveEmptyOutputs && len(dp.Output) == 0 {
		return false, "empty_output"
	}
	if len(dp.Input) < c.config.MinInputLength {
		return false, "input_too_short"
	}
	if len(dp.Input) > c.config.MaxInputLength {
		return false, "input_too_long"
	}
	if len(dp.Output) < c.config.MinOutputLength {
		return false, "output_too_short"
	}
	if len(dp.Output) > c.config.MaxOutputLength {
		return false, "output_too_long"
	}
	if dp.Quality < c.config.MinQuality {
		return false, "low_quality"
	}
	if len(c.config.AllowedCategories) > 0 {
		found := false
		for _, cat := range c.config.AllowedCategories {
			if cat == dp.Category {
				found = true
				break
			}
		}
		if !found {
			return false, "invalid_category"
		}
	}
	return true, ""
}

func (c *DataCleaner) hash(dp *DataPoint) string {
	return fmt.Sprintf("%s|%s", dp.Input, dp.Output)
}

// DatasetBuilder builds datasets with proper splits.
type DatasetBuilder struct {
	id          string
	name        string
	description string
	domain      string
	split       DatasetSplit
	data        []*DataPoint
	cleaner     *DataCleaner
	seed        int64
}

// NewDatasetBuilder creates a new dataset builder.
func NewDatasetBuilder(id, name, domain string) *DatasetBuilder {
	return &DatasetBuilder{
		id:     id,
		name:   name,
		domain: domain,
		split:  DatasetSplit{Train: 0.8, Validation: 0.1, Test: 0.1},
		seed:   time.Now().UnixNano(),
	}
}

// WithDescription sets the dataset description.
func (b *DatasetBuilder) WithDescription(desc string) *DatasetBuilder {
	b.description = desc
	return b
}

// WithSplit sets custom split ratios.
func (b *DatasetBuilder) WithSplit(train, validation, test float64) *DatasetBuilder {
	b.split = DatasetSplit{Train: train, Validation: validation, Test: test}
	return b
}

// WithCleaner sets a custom cleaner.
func (b *DatasetBuilder) WithCleaner(cleaner *DataCleaner) *DatasetBuilder {
	b.cleaner = cleaner
	return b
}

// WithSeed sets the random seed for reproducible splits.
func (b *DatasetBuilder) WithSeed(seed int64) *DatasetBuilder {
	b.seed = seed
	return b
}

// AddData adds raw data points.
func (b *DatasetBuilder) AddData(data []*DataPoint) *DatasetBuilder {
	b.data = append(b.data, data...)
	return b
}

// AddFromJSON adds data from a JSON string.
func (b *DatasetBuilder) AddFromJSON(jsonData string) error {
	var points []*DataPoint
	if err := json.Unmarshal([]byte(jsonData), &points); err != nil {
		return err
	}
	b.data = append(b.data, points...)
	return nil
}

// Build creates the final dataset with splits.
func (b *DatasetBuilder) Build() (*Dataset, error) {
	if len(b.data) == 0 {
		return nil, fmt.Errorf("no data provided")
	}

	// Validate split ratios
	total := b.split.Train + b.split.Validation + b.split.Test
	if math.Abs(total-1.0) > 0.001 {
		return nil, fmt.Errorf("split ratios must sum to 1.0, got %f", total)
	}

	// Clean data if cleaner is set
	data := b.data
	if b.cleaner != nil {
		data, _ = b.cleaner.Clean(data)
	}

	// Calculate weights if not set
	data = calculateWeights(data)

	// Shuffle with seed
	rng := rand.New(rand.NewSource(b.seed))
	shuffled := make([]*DataPoint, len(data))
	copy(shuffled, data)
	rng.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})

	// Split data
	n := len(shuffled)
	trainEnd := int(float64(n) * b.split.Train)
	valEnd := trainEnd + int(float64(n)*b.split.Validation)

	dataset := &Dataset{
		ID:          b.id,
		Name:        b.name,
		Description: b.description,
		Domain:      b.domain,
		Version:     "1.0.0",
		Split:       b.split,
		Train:       shuffled[:trainEnd],
		Validation:  shuffled[trainEnd:valEnd],
		Test:        shuffled[valEnd:],
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}

	// Calculate stats
	dataset.Stats = calculateDatasetStats(dataset)

	return dataset, nil
}

func calculateWeights(data []*DataPoint) []*DataPoint {
	// Count categories for inverse frequency weighting
	catCounts := make(map[string]int)
	for _, dp := range data {
		catCounts[dp.Category]++
	}

	total := len(data)
	for i, dp := range data {
		if dp.Weight == 0 {
			// Inverse frequency weighting
			catCount := catCounts[dp.Category]
			if catCount > 0 {
				data[i].Weight = float64(total) / float64(catCount*len(catCounts))
			} else {
				data[i].Weight = 1.0
			}
			// Adjust by quality
			if dp.Quality > 0 {
				data[i].Weight *= dp.Quality
			}
		}
	}
	return data
}

func calculateDatasetStats(ds *Dataset) *DatasetStats {
	stats := &DatasetStats{
		TotalSamples:      len(ds.Train) + len(ds.Validation) + len(ds.Test),
		TrainSamples:      len(ds.Train),
		ValidationSamples: len(ds.Validation),
		TestSamples:       len(ds.Test),
		Categories:        make(map[string]int),
		QualityDistribution: make(map[string]int),
	}

	all := append(append(ds.Train, ds.Validation...), ds.Test...)

	var totalInputLen, totalOutputLen float64
	var weights []float64

	for _, dp := range all {
		stats.Categories[dp.Category]++
		totalInputLen += float64(len(dp.Input))
		totalOutputLen += float64(len(dp.Output))
		weights = append(weights, dp.Weight)

		// Quality buckets
		switch {
		case dp.Quality >= 0.9:
			stats.QualityDistribution["excellent"]++
		case dp.Quality >= 0.7:
			stats.QualityDistribution["good"]++
		case dp.Quality >= 0.5:
			stats.QualityDistribution["fair"]++
		default:
			stats.QualityDistribution["poor"]++
		}
	}

	if len(all) > 0 {
		stats.AvgInputLength = totalInputLen / float64(len(all))
		stats.AvgOutputLength = totalOutputLen / float64(len(all))
	}

	stats.WeightStats = calculateWeightStats(weights)

	return stats
}

func calculateWeightStats(weights []float64) *WeightStats {
	if len(weights) == 0 {
		return &WeightStats{}
	}

	var sum, min, max float64
	min = weights[0]
	max = weights[0]

	for _, w := range weights {
		sum += w
		if w < min {
			min = w
		}
		if w > max {
			max = w
		}
	}

	mean := sum / float64(len(weights))

	var varianceSum float64
	for _, w := range weights {
		varianceSum += (w - mean) * (w - mean)
	}
	stdDev := math.Sqrt(varianceSum / float64(len(weights)))

	return &WeightStats{
		Min:    min,
		Max:    max,
		Mean:   mean,
		StdDev: stdDev,
	}
}

// Helper functions

func normalizeUnicode(s string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsPrint(r) {
			return r
		}
		return -1
	}, s)
}

var htmlTagRegex = regexp.MustCompile(`<[^>]*>`)

func removeHTMLTags(s string) string {
	return htmlTagRegex.ReplaceAllString(s, "")
}
