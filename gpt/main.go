package main

import (
	"fmt"
	"math/rand"
	"time"
)

func main() {
	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	fmt.Println("=== Custom GPT Model in Go ===\n")

	// Training data (simple examples)
	trainingTexts := []string{
		"hello world, how are you today?",
		"the weather is nice today.",
		"artificial intelligence is fascinating.",
		"go is a great programming language.",
		"machine learning models learn from data.",
		"natural language processing is important.",
		"transformers changed everything in AI.",
		"attention is all you need for modern models.",
		"deep learning requires lots of training data.",
		"neural networks can learn complex patterns.",
	}

	// Initialize tokenizer
	fmt.Println("Building vocabulary...")
	tokenizer := NewTokenizer()
	tokenizer.BuildVocab(trainingTexts)
	fmt.Printf("Vocabulary size: %d\n\n", tokenizer.VocabSize)

	// Model hyperparameters
	embedDim := 64      // Embedding dimension
	numLayers := 2      // Number of transformer layers
	numHeads := 4       // Number of attention heads
	maxSeqLen := 20     // Maximum sequence length

	// Create model
	fmt.Println("Initializing GPT model...")
	model := NewGPTModel(tokenizer.VocabSize, embedDim, numLayers, numHeads, maxSeqLen)
	fmt.Printf("Model parameters:\n")
	fmt.Printf("  - Embedding dimension: %d\n", embedDim)
	fmt.Printf("  - Number of layers: %d\n", numLayers)
	fmt.Printf("  - Number of attention heads: %d\n", numHeads)
	fmt.Printf("  - Max sequence length: %d\n\n", maxSeqLen)

	// Prepare training sequences
	fmt.Println("Preparing training sequences...")
	sequences := PrepareSequences(trainingTexts, tokenizer, maxSeqLen)
	fmt.Printf("Number of training sequences: %d\n\n", len(sequences))

	// Train model
	trainer := NewTrainer(model, 0.01)
	epochs := 100
	trainer.Train(sequences, epochs)
	fmt.Println()

	// Generate text
	fmt.Println("=== Text Generation Examples ===\n")

	// Example 1
	prompt1 := "hello"
	fmt.Printf("Prompt: \"%s\"\n", prompt1)
	seedTokens1 := tokenizer.Encode(prompt1)
	generated1 := model.Generate(seedTokens1, 10, 0.8)
	output1 := tokenizer.Decode(generated1)
	fmt.Printf("Generated: \"%s\"\n\n", output1)

	// Example 2
	prompt2 := "the weather"
	fmt.Printf("Prompt: \"%s\"\n", prompt2)
	seedTokens2 := tokenizer.Encode(prompt2)
	generated2 := model.Generate(seedTokens2, 10, 0.8)
	output2 := tokenizer.Decode(generated2)
	fmt.Printf("Generated: \"%s\"\n\n", output2)

	// Example 3
	prompt3 := "machine learning"
	fmt.Printf("Prompt: \"%s\"\n", prompt3)
	seedTokens3 := tokenizer.Encode(prompt3)
	generated3 := model.Generate(seedTokens3, 10, 0.8)
	output3 := tokenizer.Decode(generated3)
	fmt.Printf("Generated: \"%s\"\n\n", output3)

	// Show model architecture summary
	fmt.Println("=== Model Architecture Summary ===")
	fmt.Printf("Total transformer blocks: %d\n", len(model.Blocks))
	fmt.Printf("Each block contains:\n")
	fmt.Printf("  - Multi-head attention (%d heads)\n", model.NumHeads)
	fmt.Printf("  - Feed-forward network\n")
	fmt.Printf("  - Layer normalization (x2)\n")
	fmt.Printf("  - Residual connections (x2)\n")
	fmt.Printf("\nToken embedding matrix: %dx%d\n", model.TokenEmbed.Rows, model.TokenEmbed.Cols)
	fmt.Printf("Position embedding matrix: %dx%d\n", model.PosEmbed.Rows, model.PosEmbed.Cols)
	fmt.Printf("Output projection matrix: %dx%d\n", model.OutputLayer.Rows, model.OutputLayer.Cols)

	fmt.Println("\n=== Note ===")
	fmt.Println("This is a minimal implementation for educational purposes.")
	fmt.Println("For production use, consider:")
	fmt.Println("  - Proper gradient computation and backpropagation")
	fmt.Println("  - Advanced optimizers (Adam, AdamW)")
	fmt.Println("  - Better tokenization (BPE, WordPiece)")
	fmt.Println("  - GPU acceleration")
	fmt.Println("  - Gradient clipping and learning rate scheduling")
	fmt.Println("  - Proper weight initialization schemes")
	fmt.Println("  - Dropout and regularization")
	fmt.Println("  - Larger datasets and longer training")
}
