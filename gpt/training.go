package main

import (
	"fmt"
	"math"
)

// Trainer handles model training
type Trainer struct {
	Model        *GPTModel
	LearningRate float64
}

// NewTrainer creates a new trainer
func NewTrainer(model *GPTModel, lr float64) *Trainer {
	return &Trainer{
		Model:        model,
		LearningRate: lr,
	}
}

// CrossEntropyLoss computes the cross-entropy loss
func CrossEntropyLoss(predictions *Matrix, targets []int) float64 {
	loss := 0.0
	for i, target := range targets {
		prob := predictions.Get(i, target)
		if prob < 1e-10 {
			prob = 1e-10 // Prevent log(0)
		}
		loss -= math.Log(prob)
	}
	return loss / float64(len(targets))
}

// TrainStep performs a single training step
func (tr *Trainer) TrainStep(tokenIDs []int) float64 {
	if len(tokenIDs) < 2 {
		return 0.0
	}

	// Prepare input and target sequences
	inputSeq := tokenIDs[:len(tokenIDs)-1]
	targetSeq := tokenIDs[1:]

	// Forward pass
	logits := tr.Model.Forward(inputSeq)

	// Convert logits to probabilities
	probs := NewMatrix(logits.Rows, logits.Cols)
	for i := 0; i < logits.Rows; i++ {
		rowLogits := NewMatrix(1, logits.Cols)
		for j := 0; j < logits.Cols; j++ {
			rowLogits.Set(0, j, logits.Get(i, j))
		}
		rowProbs := Softmax(rowLogits)
		for j := 0; j < logits.Cols; j++ {
			probs.Set(i, j, rowProbs.Get(0, j))
		}
	}

	// Calculate loss
	loss := CrossEntropyLoss(probs, targetSeq)

	// Simplified gradient descent (in practice, you'd compute proper gradients)
	// This is a placeholder for demonstration - real implementation would need backpropagation
	tr.simpleUpdate(inputSeq, targetSeq, probs)

	return loss
}

// simpleUpdate performs a simplified parameter update
// Note: This is a simplified version. Real implementation needs proper backpropagation
func (tr *Trainer) simpleUpdate(inputs []int, targets []int, probs *Matrix) {
	// Gradient computation would go here
	// For demonstration, we'll do a simplified update

	// In a real implementation, you would:
	// 1. Compute gradients via backpropagation
	// 2. Update all parameters (embeddings, attention weights, etc.)
	// 3. Apply optimization algorithm (Adam, SGD, etc.)

	// This is a placeholder showing the structure
	eps := tr.LearningRate * 0.01

	// Compute error signal
	for i := 0; i < probs.Rows; i++ {
		target := targets[i]
		error := probs.Get(i, target) - 1.0

		// Update output layer (simplified)
		for j := 0; j < tr.Model.OutputLayer.Rows; j++ {
			grad := error * eps
			current := tr.Model.OutputLayer.Get(j, target)
			tr.Model.OutputLayer.Set(j, target, current-grad)
		}
	}
}

// Train trains the model on a dataset
func (tr *Trainer) Train(sequences [][]int, epochs int) {
	fmt.Println("Training model...")

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		count := 0

		for _, seq := range sequences {
			if len(seq) < 2 {
				continue
			}
			loss := tr.TrainStep(seq)
			totalLoss += loss
			count++
		}

		avgLoss := totalLoss / float64(count)
		if epoch%10 == 0 {
			fmt.Printf("Epoch %d, Average Loss: %.4f\n", epoch, avgLoss)
		}
	}

	fmt.Println("Training complete!")
}

// PrepareSequences creates training sequences from text
func PrepareSequences(texts []string, tokenizer *Tokenizer, maxLen int) [][]int {
	var sequences [][]int

	for _, text := range texts {
		tokens := tokenizer.Encode(text)

		// Create overlapping sequences
		for i := 0; i+1 < len(tokens) && i < maxLen; i++ {
			end := i + maxLen
			if end > len(tokens) {
				end = len(tokens)
			}
			if end-i >= 2 { // At least 2 tokens (input + target)
				sequences = append(sequences, tokens[i:end])
			}
		}
	}

	return sequences
}
