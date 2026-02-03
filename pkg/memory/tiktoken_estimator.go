package memory

import (
	"github.com/pkoukk/tiktoken-go"
	"github.com/sujit/ai-agent/pkg/content"
)

// TiktokenEstimator uses the tiktoken library to estimate tokens.
type TiktokenEstimator struct {
	enc *tiktoken.Tiktoken
}

// NewTiktokenEstimator initializes a tiktoken estimator for the given encoding name.
// If initialization fails, it returns an error.
func NewTiktokenEstimator(encoding string) (*TiktokenEstimator, error) {
	enc, err := tiktoken.GetEncoding(encoding)
	if err != nil {
		return nil, err
	}
	return &TiktokenEstimator{enc: enc}, nil
}

func (t *TiktokenEstimator) EstimateMessage(msg *content.Message) int {
	if msg == nil {
		return 0
	}
	return t.EstimateText(msg.GetText())
}

func (t *TiktokenEstimator) EstimateText(text string) int {
	if t.enc == nil || text == "" {
		return 0
	}
	ids := t.enc.Encode(text, nil, nil)
	return len(ids)
}
