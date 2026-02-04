package main

import (
	"math"
	"math/rand"
)

// Matrix represents a 2D matrix
type Matrix struct {
	Rows int
	Cols int
	Data []float64
}

// NewMatrix creates a new matrix
func NewMatrix(rows, cols int) *Matrix {
	return &Matrix{
		Rows: rows,
		Cols: cols,
		Data: make([]float64, rows*cols),
	}
}

// RandomMatrix creates a matrix with random values
func RandomMatrix(rows, cols int, scale float64) *Matrix {
	m := NewMatrix(rows, cols)
	for i := range m.Data {
		m.Data[i] = (rand.Float64()*2 - 1) * scale
	}
	return m
}

// Get returns the value at position (i, j)
func (m *Matrix) Get(i, j int) float64 {
	return m.Data[i*m.Cols+j]
}

// Set sets the value at position (i, j)
func (m *Matrix) Set(i, j int, val float64) {
	m.Data[i*m.Cols+j] = val
}

// MatMul performs matrix multiplication
func MatMul(a, b *Matrix) *Matrix {
	if a.Cols != b.Rows {
		panic("incompatible matrix dimensions")
	}
	result := NewMatrix(a.Rows, b.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < b.Cols; j++ {
			sum := 0.0
			for k := 0; k < a.Cols; k++ {
				sum += a.Get(i, k) * b.Get(k, j)
			}
			result.Set(i, j, sum)
		}
	}
	return result
}

// Softmax applies softmax function along rows
func Softmax(m *Matrix) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		max := m.Get(i, 0)
		for j := 1; j < m.Cols; j++ {
			if m.Get(i, j) > max {
				max = m.Get(i, j)
			}
		}
		sum := 0.0
		for j := 0; j < m.Cols; j++ {
			val := math.Exp(m.Get(i, j) - max)
			result.Set(i, j, val)
			sum += val
		}
		for j := 0; j < m.Cols; j++ {
			result.Set(i, j, result.Get(i, j)/sum)
		}
	}
	return result
}

// GELU activation function
func GELU(x float64) float64 {
	return 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*math.Pow(x, 3))))
}

// ApplyGELU applies GELU activation to all elements
func ApplyGELU(m *Matrix) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := range m.Data {
		result.Data[i] = GELU(m.Data[i])
	}
	return result
}

// LayerNorm applies layer normalization
func LayerNorm(m *Matrix, eps float64) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		mean := 0.0
		for j := 0; j < m.Cols; j++ {
			mean += m.Get(i, j)
		}
		mean /= float64(m.Cols)

		variance := 0.0
		for j := 0; j < m.Cols; j++ {
			diff := m.Get(i, j) - mean
			variance += diff * diff
		}
		variance /= float64(m.Cols)

		std := math.Sqrt(variance + eps)
		for j := 0; j < m.Cols; j++ {
			result.Set(i, j, (m.Get(i, j)-mean)/std)
		}
	}
	return result
}

// Add performs element-wise addition
func Add(a, b *Matrix) *Matrix {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("incompatible matrix dimensions")
	}
	result := NewMatrix(a.Rows, a.Cols)
	for i := range a.Data {
		result.Data[i] = a.Data[i] + b.Data[i]
	}
	return result
}

// MultiHeadAttention implements multi-head self-attention
type MultiHeadAttention struct {
	NumHeads int
	HeadDim  int
	EmbedDim int
	WQ       *Matrix
	WK       *Matrix
	WV       *Matrix
	WO       *Matrix
}

// NewMultiHeadAttention creates a new multi-head attention layer
func NewMultiHeadAttention(embedDim, numHeads int) *MultiHeadAttention {
	headDim := embedDim / numHeads
	scale := math.Sqrt(float64(embedDim))
	return &MultiHeadAttention{
		NumHeads: numHeads,
		HeadDim:  headDim,
		EmbedDim: embedDim,
		WQ:       RandomMatrix(embedDim, embedDim, 1.0/scale),
		WK:       RandomMatrix(embedDim, embedDim, 1.0/scale),
		WV:       RandomMatrix(embedDim, embedDim, 1.0/scale),
		WO:       RandomMatrix(embedDim, embedDim, 1.0/scale),
	}
}

// Forward performs the forward pass
func (mha *MultiHeadAttention) Forward(x *Matrix) *Matrix {
	seqLen := x.Rows

	// Linear projections
	Q := MatMul(x, mha.WQ)
	K := MatMul(x, mha.WK)
	V := MatMul(x, mha.WV)

	// Compute attention scores
	scores := MatMul(Q, Transpose(K))
	scale := math.Sqrt(float64(mha.HeadDim))
	for i := range scores.Data {
		scores.Data[i] /= scale
	}

	// Apply causal mask (for autoregressive generation)
	for i := 0; i < seqLen; i++ {
		for j := i + 1; j < seqLen; j++ {
			scores.Set(i, j, -1e10)
		}
	}

	// Apply softmax
	attnWeights := Softmax(scores)

	// Apply attention to values
	output := MatMul(attnWeights, V)

	// Output projection
	return MatMul(output, mha.WO)
}

// Transpose transposes a matrix
func Transpose(m *Matrix) *Matrix {
	result := NewMatrix(m.Cols, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Set(j, i, m.Get(i, j))
		}
	}
	return result
}

// FeedForward implements the position-wise feed-forward network
type FeedForward struct {
	W1 *Matrix
	W2 *Matrix
}

// NewFeedForward creates a new feed-forward layer
func NewFeedForward(embedDim, hiddenDim int) *FeedForward {
	scale := math.Sqrt(float64(embedDim))
	return &FeedForward{
		W1: RandomMatrix(embedDim, hiddenDim, 1.0/scale),
		W2: RandomMatrix(hiddenDim, embedDim, 1.0/math.Sqrt(float64(hiddenDim))),
	}
}

// Forward performs the forward pass
func (ff *FeedForward) Forward(x *Matrix) *Matrix {
	hidden := MatMul(x, ff.W1)
	activated := ApplyGELU(hidden)
	return MatMul(activated, ff.W2)
}

// TransformerBlock represents a single transformer layer
type TransformerBlock struct {
	Attention  *MultiHeadAttention
	FeedForward *FeedForward
	Eps        float64
}

// NewTransformerBlock creates a new transformer block
func NewTransformerBlock(embedDim, numHeads, hiddenDim int) *TransformerBlock {
	return &TransformerBlock{
		Attention:   NewMultiHeadAttention(embedDim, numHeads),
		FeedForward: NewFeedForward(embedDim, hiddenDim),
		Eps:         1e-5,
	}
}

// Forward performs the forward pass
func (tb *TransformerBlock) Forward(x *Matrix) *Matrix {
	// Self-attention with residual connection
	normalized := LayerNorm(x, tb.Eps)
	attnOutput := tb.Attention.Forward(normalized)
	x = Add(x, attnOutput)

	// Feed-forward with residual connection
	normalized = LayerNorm(x, tb.Eps)
	ffOutput := tb.FeedForward.Forward(normalized)
	return Add(x, ffOutput)
}

// GPTModel represents the full GPT model
type GPTModel struct {
	VocabSize    int
	EmbedDim     int
	NumLayers    int
	NumHeads     int
	MaxSeqLen    int
	TokenEmbed   *Matrix
	PosEmbed     *Matrix
	Blocks       []*TransformerBlock
	OutputLayer  *Matrix
}

// NewGPTModel creates a new GPT model
func NewGPTModel(vocabSize, embedDim, numLayers, numHeads, maxSeqLen int) *GPTModel {
	hiddenDim := embedDim * 4
	blocks := make([]*TransformerBlock, numLayers)
	for i := 0; i < numLayers; i++ {
		blocks[i] = NewTransformerBlock(embedDim, numHeads, hiddenDim)
	}

	scale := math.Sqrt(float64(embedDim))
	return &GPTModel{
		VocabSize:   vocabSize,
		EmbedDim:    embedDim,
		NumLayers:   numLayers,
		NumHeads:    numHeads,
		MaxSeqLen:   maxSeqLen,
		TokenEmbed:  RandomMatrix(vocabSize, embedDim, 1.0/scale),
		PosEmbed:    RandomMatrix(maxSeqLen, embedDim, 1.0/scale),
		Blocks:      blocks,
		OutputLayer: RandomMatrix(embedDim, vocabSize, 1.0/scale),
	}
}

// Embed converts token IDs to embeddings
func (gpt *GPTModel) Embed(tokenIDs []int) *Matrix {
	seqLen := len(tokenIDs)
	embeddings := NewMatrix(seqLen, gpt.EmbedDim)

	for i, tokenID := range tokenIDs {
		// Token embedding
		for j := 0; j < gpt.EmbedDim; j++ {
			embeddings.Set(i, j, gpt.TokenEmbed.Get(tokenID, j))
		}
		// Add positional embedding
		for j := 0; j < gpt.EmbedDim; j++ {
			embeddings.Set(i, j, embeddings.Get(i, j)+gpt.PosEmbed.Get(i, j))
		}
	}

	return embeddings
}

// Forward performs the forward pass through the model
func (gpt *GPTModel) Forward(tokenIDs []int) *Matrix {
	// Get embeddings
	x := gpt.Embed(tokenIDs)

	// Pass through transformer blocks
	for _, block := range gpt.Blocks {
		x = block.Forward(x)
	}

	// Final layer norm
	x = LayerNorm(x, 1e-5)

	// Project to vocabulary
	return MatMul(x, gpt.OutputLayer)
}

// Predict generates the next token probabilities
func (gpt *GPTModel) Predict(tokenIDs []int) []float64 {
	logits := gpt.Forward(tokenIDs)
	lastLogits := NewMatrix(1, gpt.VocabSize)
	lastIdx := logits.Rows - 1
	for j := 0; j < gpt.VocabSize; j++ {
		lastLogits.Set(0, j, logits.Get(lastIdx, j))
	}
	probs := Softmax(lastLogits)
	return probs.Data
}

// Sample samples a token from the probability distribution
func Sample(probs []float64, temperature float64) int {
	// Apply temperature
	if temperature != 1.0 {
		sum := 0.0
		for i := range probs {
			probs[i] = math.Pow(probs[i], 1.0/temperature)
			sum += probs[i]
		}
		for i := range probs {
			probs[i] /= sum
		}
	}

	// Sample
	r := rand.Float64()
	cumulative := 0.0
	for i, p := range probs {
		cumulative += p
		if r < cumulative {
			return i
		}
	}
	return len(probs) - 1
}

// Generate generates text autoregressively
func (gpt *GPTModel) Generate(seedTokens []int, maxNewTokens int, temperature float64) []int {
	tokens := make([]int, len(seedTokens))
	copy(tokens, seedTokens)

	for i := 0; i < maxNewTokens; i++ {
		// Get probabilities for next token
		probs := gpt.Predict(tokens)

		// Sample next token
		nextToken := Sample(probs, temperature)

		// Add to sequence
		tokens = append(tokens, nextToken)

		// Prevent sequence from exceeding max length
		if len(tokens) >= gpt.MaxSeqLen {
			break
		}
	}

	return tokens
}
