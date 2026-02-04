package main

import (
	"strings"
	"unicode"
)

// Tokenizer handles text tokenization
type Tokenizer struct {
	Vocab       map[string]int
	InverseVocab map[int]string
	VocabSize   int
}

// NewTokenizer creates a new tokenizer
func NewTokenizer() *Tokenizer {
	return &Tokenizer{
		Vocab:       make(map[string]int),
		InverseVocab: make(map[int]string),
		VocabSize:   0,
	}
}

// BuildVocab builds vocabulary from text
func (t *Tokenizer) BuildVocab(texts []string) {
	// Add special tokens
	t.addToken("<PAD>")
	t.addToken("<UNK>")
	t.addToken("<START>")
	t.addToken("<END>")

	// Collect all unique words
	wordSet := make(map[string]bool)
	for _, text := range texts {
		words := t.tokenize(text)
		for _, word := range words {
			wordSet[word] = true
		}
	}

	// Add to vocabulary
	for word := range wordSet {
		t.addToken(word)
	}
}

// addToken adds a token to the vocabulary
func (t *Tokenizer) addToken(token string) {
	if _, exists := t.Vocab[token]; !exists {
		t.Vocab[token] = t.VocabSize
		t.InverseVocab[t.VocabSize] = token
		t.VocabSize++
	}
}

// tokenize splits text into tokens
func (t *Tokenizer) tokenize(text string) []string {
	// Simple word-level tokenization
	text = strings.ToLower(text)

	// Split by whitespace and punctuation
	var tokens []string
	var currentWord strings.Builder

	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			currentWord.WriteRune(r)
		} else {
			if currentWord.Len() > 0 {
				tokens = append(tokens, currentWord.String())
				currentWord.Reset()
			}
			if !unicode.IsSpace(r) {
				tokens = append(tokens, string(r))
			}
		}
	}

	if currentWord.Len() > 0 {
		tokens = append(tokens, currentWord.String())
	}

	return tokens
}

// Encode converts text to token IDs
func (t *Tokenizer) Encode(text string) []int {
	words := t.tokenize(text)
	ids := make([]int, 0, len(words)+1)

	// Add start token
	ids = append(ids, t.Vocab["<START>"])

	for _, word := range words {
		if id, exists := t.Vocab[word]; exists {
			ids = append(ids, id)
		} else {
			ids = append(ids, t.Vocab["<UNK>"])
		}
	}

	return ids
}

// Decode converts token IDs back to text
func (t *Tokenizer) Decode(ids []int) string {
	var words []string

	for _, id := range ids {
		if word, exists := t.InverseVocab[id]; exists {
			if word != "<START>" && word != "<PAD>" && word != "<UNK>" {
				words = append(words, word)
			}
		}
	}

	// Simple reconstruction (doesn't perfectly preserve original spacing/punctuation)
	result := strings.Join(words, " ")

	// Fix spacing around punctuation
	result = strings.ReplaceAll(result, " ,", ",")
	result = strings.ReplaceAll(result, " .", ".")
	result = strings.ReplaceAll(result, " !", "!")
	result = strings.ReplaceAll(result, " ?", "?")
	result = strings.ReplaceAll(result, " '", "'")

	return result
}
