package memory

import (
	"testing"
)

const sampleText = `This is a sample piece of text that is reasonably long and includes several sentences to mimic realistic messages. It will be used for benchmarking tokenizers and estimators.`

func BenchmarkDefaultTokenEstimator_EstimateText(b *testing.B) {
	d := DefaultTokenEstimator{}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		d.EstimateText(sampleText)
	}
}

func BenchmarkTiktokenEstimator_EstimateText(b *testing.B) {
	enc, err := NewTiktokenEstimator("cl100k_base")
	if err != nil {
		b.Skipf("tiktoken unavailable: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		enc.EstimateText(sampleText)
	}
}
