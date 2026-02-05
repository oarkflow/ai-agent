package processor

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/oarkflow/ai-agent/pkg/content"
	"github.com/oarkflow/ai-agent/pkg/llm"
)

// ------------------------------
// ContentAnalyzer
// ------------------------------

type ContentAnalyzer struct {
	Router *llm.SmartRouter
}

func NewContentAnalyzer(router *llm.SmartRouter) *ContentAnalyzer {
	return &ContentAnalyzer{Router: router}
}

// AnalyzeImage analyzes an image and returns an AnalysisResult
func (a *ContentAnalyzer) AnalyzeImage(ctx context.Context, imagePath string, prompt string) (*AnalysisResult, error) {
	_, err := content.NewImageFromFile(imagePath)
	if err != nil {
		return nil, fmt.Errorf("failed to load image: %w", err)
	}

	if prompt == "" {
		prompt = "Analyze this image in detail. Describe what you see, any text visible, objects, people, colors, composition, and any notable features."
	}

	msg := content.NewUserMessage(prompt)
	_, err = msg.AddImageFromFile(imagePath)
	if err != nil {
		return nil, fmt.Errorf("failed to add image %s: %w", imagePath, err)
	}

	resp, err := a.Router.Route(ctx, []*content.Message{msg}, nil, &llm.ModelRequirements{
		Capabilities: []llm.Capability{llm.CapVision},
	})
	if err != nil {
		return nil, err
	}

	return &AnalysisResult{
		Description: resp.Message.GetText(),
		Type:        content.TypeImage,
		Usage:       resp.Usage,
		Model:       resp.Model,
	}, nil
}

// ------------------------------
// AnalyzeDocument / Video / Audio
// ------------------------------

func (a *ContentAnalyzer) AnalyzeDocument(ctx context.Context, docPath, prompt string) (*AnalysisResult, error) {
	docContent, err := content.NewDocumentFromFile(docPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load document: %w", err)
	}

	if prompt == "" {
		prompt = "Analyze this document. Provide a summary of the content, key points, and any important information."
	}

	msg := content.NewUserMessage(prompt)
	msg.AddContent(docContent)

	resp, err := a.Router.Route(ctx, []*content.Message{msg}, nil, &llm.ModelRequirements{
		Capabilities: []llm.Capability{llm.CapDocument},
	})
	if err != nil {
		return nil, err
	}

	return &AnalysisResult{
		Description: resp.Message.GetText(),
		Type:        content.TypeDocument,
		Usage:       resp.Usage,
		Model:       resp.Model,
		FileName:    filepath.Base(docPath),
	}, nil
}

func (a *ContentAnalyzer) AnalyzeVideo(ctx context.Context, videoPath, prompt string) (*AnalysisResult, error) {
	videoContent, err := content.NewVideoFromFile(videoPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load video: %w", err)
	}

	if prompt == "" {
		prompt = "Analyze this video in detail. Describe the content, key moments, any speech or text, and provide a summary of what happens."
	}

	msg := content.NewUserMessage(prompt)
	msg.AddContent(videoContent)

	resp, err := a.Router.Route(ctx, []*content.Message{msg}, nil, &llm.ModelRequirements{
		Capabilities: []llm.Capability{llm.CapVideo},
	})
	if err != nil {
		return nil, err
	}

	return &AnalysisResult{
		Description: resp.Message.GetText(),
		Type:        content.TypeVideo,
		Usage:       resp.Usage,
		Model:       resp.Model,
	}, nil
}

func (a *ContentAnalyzer) AnalyzeAudio(ctx context.Context, audioPath, prompt string) (*AnalysisResult, error) {
	audioContent, err := content.NewAudioFromFile(audioPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load audio: %w", err)
	}

	if prompt == "" {
		prompt = "Transcribe and analyze this audio."
	}

	transcription, err := a.Router.RouteTranscribe(ctx, &llm.TranscriptionRequest{
		Audio:      audioContent,
		Timestamps: true,
	})
	if err != nil {
		return nil, fmt.Errorf("transcription failed: %w", err)
	}

	msg := content.NewUserMessage(fmt.Sprintf("%s\n\nTranscription:\n%s", prompt, transcription.Text))

	resp, err := a.Router.Route(ctx, []*content.Message{msg}, nil, &llm.ModelRequirements{
		Capabilities: []llm.Capability{llm.CapAudio},
	})
	if err != nil {
		return nil, err
	}

	return &AnalysisResult{
		Description:   resp.Message.GetText(),
		Type:          content.TypeAudio,
		Usage:         resp.Usage,
		Model:         resp.Model,
		Language:      transcription.Language,
		Duration:      transcription.Duration,
		Transcription: transcription.Text,
	}, nil
}

// ------------------------------
// Content Type Detection
// ------------------------------

func detectContentType(filePath string) content.ContentType {
	ext := strings.ToLower(filepath.Ext(filePath))

	codeExts := map[string]bool{".go": true, ".py": true, ".js": true, ".ts": true, ".java": true, ".rs": true}
	docExts := map[string]bool{".pdf": true, ".docx": true, ".xlsx": true, ".pptx": true, ".txt": true, ".md": true}
	videoExts := map[string]bool{".mp4": true, ".mov": true, ".avi": true, ".webm": true, ".mkv": true}
	audioExts := map[string]bool{".mp3": true, ".wav": true, ".ogg": true, ".flac": true, ".m4a": true}
	imageExts := map[string]bool{".jpg": true, ".jpeg": true, ".png": true, ".gif": true, ".webp": true, ".svg": true}

	switch {
	case codeExts[ext]:
		return content.TypeCode
	case docExts[ext]:
		return content.TypeDocument
	case videoExts[ext]:
		return content.TypeVideo
	case audioExts[ext]:
		return content.TypeAudio
	case imageExts[ext]:
		return content.TypeImage
	default:
		return content.TypeFile
	}
}

// ------------------------------
// Batch Processing
// ------------------------------

type BatchProcessor struct {
	Analyzer    *ContentAnalyzer
	Concurrency int
}

func NewBatchProcessor(analyzer *ContentAnalyzer, concurrency int) *BatchProcessor {
	if concurrency <= 0 {
		concurrency = 5
	}
	return &BatchProcessor{
		Analyzer:    analyzer,
		Concurrency: concurrency,
	}
}

type BatchResult struct {
	FilePath string      `json:"file_path"`
	Result   interface{} `json:"result,omitempty"`
	Error    string      `json:"error,omitempty"`
}

func (bp *BatchProcessor) ProcessFiles(ctx context.Context, files []string) ([]BatchResult, error) {
	results := make([]BatchResult, len(files))

	for i, file := range files {
		contentType := detectContentType(file)
		var result *AnalysisResult
		var err error

		switch contentType {
		case content.TypeImage:
			result, err = bp.Analyzer.AnalyzeImage(ctx, file, "")
		case content.TypeAudio:
			result, err = bp.Analyzer.AnalyzeAudio(ctx, file, "")
		case content.TypeVideo:
			result, err = bp.Analyzer.AnalyzeVideo(ctx, file, "")
		case content.TypeDocument:
			result, err = bp.Analyzer.AnalyzeDocument(ctx, file, "")
		default:
			results[i] = BatchResult{FilePath: file, Error: "unsupported content type"}
			continue
		}

		if err != nil {
			results[i] = BatchResult{FilePath: file, Error: err.Error()}
			continue
		}

		results[i] = BatchResult{FilePath: file, Result: result}
	}

	return results, nil
}

func (bp *BatchProcessor) ProcessDirectory(ctx context.Context, dirPath string, recursive bool) ([]BatchResult, error) {
	var files []string
	walkFn := func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() && path != dirPath && !recursive {
			return filepath.SkipDir
		}
		if !info.IsDir() {
			files = append(files, path)
		}
		return nil
	}

	if err := filepath.Walk(dirPath, walkFn); err != nil {
		return nil, fmt.Errorf("failed to walk directory: %w", err)
	}

	return bp.ProcessFiles(ctx, files)
}

// ------------------------------
// Analysis Result
// ------------------------------

type AnalysisResult struct {
	Type          content.ContentType `json:"type"`
	Description   string              `json:"description"`
	Analysis      string              `json:"analysis,omitempty"`
	Usage         *llm.Usage          `json:"usage,omitempty"`
	Model         string              `json:"model"`
	FileName      string              `json:"file_name,omitempty"`
	Language      string              `json:"language,omitempty"`
	Duration      float64             `json:"duration,omitempty"`
	Transcription string              `json:"transcription,omitempty"`
}

// ------------------------------
// Code Analysis
// ------------------------------

// CodeAnalysisType defines the type of code analysis to perform.
type CodeAnalysisType string

const (
	CodeAnalysisReview   CodeAnalysisType = "review"
	CodeAnalysisOptimize CodeAnalysisType = "optimize"
	CodeAnalysisExplain  CodeAnalysisType = "explain"
	CodeAnalysisRefactor CodeAnalysisType = "refactor"
	CodeAnalysisSecurity CodeAnalysisType = "security"
	CodeAnalysisTest     CodeAnalysisType = "test"
)

// AnalyzeCode analyzes code and returns insights based on analysis type.
func (a *ContentAnalyzer) AnalyzeCode(ctx context.Context, code, language string, analysisType CodeAnalysisType) (*AnalysisResult, error) {
	var prompt string

	switch analysisType {
	case CodeAnalysisReview:
		prompt = fmt.Sprintf(`Review the following %s code. Identify:
1. Potential bugs or errors
2. Code quality issues
3. Performance concerns
4. Security vulnerabilities
5. Best practice violations

Code:
%s`, language, code)

	case CodeAnalysisOptimize:
		prompt = fmt.Sprintf(`Analyze the following %s code for optimization opportunities:
1. Performance improvements
2. Memory efficiency
3. Algorithmic optimizations
4. Code simplification

Provide the optimized version with explanations.

Code:
%s`, language, code)

	case CodeAnalysisExplain:
		prompt = fmt.Sprintf(`Explain the following %s code in detail:
1. What the code does
2. How it works step by step
3. Key concepts and patterns used
4. Input/output behavior

Code:
%s`, language, code)

	case CodeAnalysisRefactor:
		prompt = fmt.Sprintf(`Refactor the following %s code to improve:
1. Readability
2. Maintainability
3. Testability
4. Code structure

Provide the refactored version with explanations.

Code:
%s`, language, code)

	case CodeAnalysisSecurity:
		prompt = fmt.Sprintf(`Perform a security analysis of the following %s code:
1. Identify security vulnerabilities
2. Check for common attack vectors (SQL injection, XSS, etc.)
3. Review authentication/authorization issues
4. Assess data handling practices

Provide severity ratings and remediation suggestions.

Code:
%s`, language, code)

	case CodeAnalysisTest:
		prompt = fmt.Sprintf(`Generate comprehensive tests for the following %s code:
1. Unit tests for each function
2. Edge cases
3. Error handling tests
4. Integration test suggestions

Code:
%s`, language, code)

	default:
		prompt = fmt.Sprintf(`Analyze the following %s code:
%s`, language, code)
	}

	msg := content.NewUserMessage(prompt)

	resp, err := a.Router.Route(ctx, []*content.Message{msg}, nil, &llm.ModelRequirements{
		Capabilities: []llm.Capability{llm.CapCodeGeneration},
	})
	if err != nil {
		return nil, err
	}

	return &AnalysisResult{
		Type:     content.TypeCode,
		Analysis: resp.Message.GetText(),
		Usage:    resp.Usage,
		Model:    resp.Model,
		Language: language,
	}, nil
}
