package domains

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/oarkflow/ai-agent/pkg/content"
	"github.com/oarkflow/ai-agent/pkg/llm"
	"github.com/oarkflow/ai-agent/pkg/storage"
)

// MultimodalDomain handles multimodal analysis (images, documents, files).
type MultimodalDomain struct {
	*BaseDomain
}

// AnalysisResult represents the result of multimodal analysis.
type AnalysisResult struct {
	Type           string         `json:"type"`        // image, document, file
	Format         string         `json:"format"`      // jpg, png, pdf, etc.
	Description    string         `json:"description"` // Human-readable description
	ExtractedText  string         `json:"extracted_text,omitempty"`
	StructuredData map[string]any `json:"structured_data,omitempty"`
	Entities       []Entity       `json:"entities,omitempty"`
	Tags           []string       `json:"tags,omitempty"`
	Confidence     float64        `json:"confidence,omitempty"`
	Metadata       map[string]any `json:"metadata,omitempty"`
}

// Entity represents an extracted entity from content.
type Entity struct {
	Type       string          `json:"type"` // person, organization, location, date, amount, etc.
	Value      string          `json:"value"`
	Label      string          `json:"label,omitempty"`
	Confidence float64         `json:"confidence,omitempty"`
	Position   *EntityPosition `json:"position,omitempty"`
	Metadata   map[string]any  `json:"metadata,omitempty"`
}

// EntityPosition represents location in document/image.
type EntityPosition struct {
	Page       int     `json:"page,omitempty"`
	X          float64 `json:"x,omitempty"`
	Y          float64 `json:"y,omitempty"`
	Width      float64 `json:"width,omitempty"`
	Height     float64 `json:"height,omitempty"`
	StartIndex int     `json:"start_index,omitempty"`
	EndIndex   int     `json:"end_index,omitempty"`
}

// ExtractionSchema defines what to extract from content.
type ExtractionSchema struct {
	Name         string        `json:"name"`
	Description  string        `json:"description"`
	Fields       []SchemaField `json:"fields"`
	OutputFormat string        `json:"output_format"` // json, text, markdown, csv
}

// SchemaField defines a field to extract.
type SchemaField struct {
	Name        string `json:"name"`
	Type        string `json:"type"` // string, number, boolean, array, object, date
	Description string `json:"description"`
	Required    bool   `json:"required"`
	Format      string `json:"format,omitempty"` // email, phone, currency, etc.
	Default     any    `json:"default,omitempty"`
}

// NewMultimodalDomain creates a new multimodal domain trainer.
func NewMultimodalDomain(provider llm.MultimodalProvider, storage *storage.Storage) *MultimodalDomain {
	return &MultimodalDomain{
		BaseDomain: NewBaseDomain("multimodal", "Multimodal Analysis", provider, storage),
	}
}

// GetSystemPrompt returns the system prompt for multimodal analysis.
func (d *MultimodalDomain) GetSystemPrompt() string {
	return `You are an expert at analyzing multimodal content including images, documents, and files.

Your capabilities include:
- Image Analysis: Object detection, text extraction (OCR), scene understanding, visual QA
- Document Analysis: Text extraction, layout understanding, table extraction, form processing
- Data Extraction: Entity recognition, structured data extraction, JSON generation
- Content Classification: Categorization, tagging, sentiment analysis

When analyzing content:
1. Describe what you observe in detail
2. Extract relevant text and data
3. Identify key entities and their relationships
4. Structure the output according to the requested format
5. Provide confidence scores for extractions

Output formats supported:
- JSON: Structured data with nested objects and arrays
- Text: Human-readable descriptions
- Markdown: Formatted text with tables and lists
- CSV: Tabular data

Always be accurate and acknowledge uncertainty when appropriate.`
}

// AnalyzeImage analyzes an image and returns structured results.
func (d *MultimodalDomain) AnalyzeImage(ctx context.Context, image *content.Content, prompt string) (*AnalysisResult, error) {
	examples, _ := d.GetFewShotExamples("image_analysis", 2)

	messages := []*content.Message{
		content.NewTextMessage(content.RoleSystem, d.GetSystemPrompt()),
	}

	// Add few-shot examples (text only for context)
	for _, ex := range examples {
		messages = append(messages,
			content.NewTextMessage(content.RoleUser, ex.Input),
			content.NewTextMessage(content.RoleAssistant, ex.Output),
		)
	}

	// Create multimodal message with image
	userMsg := content.NewUserMessage(prompt)
	userMsg.Contents = append(userMsg.Contents, image)

	analysisPrompt := prompt
	if analysisPrompt == "" {
		analysisPrompt = `Analyze this image and provide:
1. A detailed description
2. Any text visible in the image
3. Key entities identified
4. Relevant tags

Respond with a JSON object.`
	}
	userMsg.Contents[0] = &content.Content{Type: content.TypeText, Text: analysisPrompt}

	messages = append(messages, userMsg)

	resp, err := d.provider.Generate(ctx, messages, &llm.GenerationConfig{
		Temperature: 0.3,
		MaxTokens:   2048,
	})
	if err != nil {
		return nil, err
	}

	responseText := resp.Message.GetText()
	jsonStr := extractMultimodalJSON(responseText)

	var result AnalysisResult
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		// If JSON parsing fails, return as description
		result = AnalysisResult{
			Type:        "image",
			Description: responseText,
		}
	}
	result.Type = "image"

	return &result, nil
}

// AnalyzeDocument analyzes a document and extracts data.
func (d *MultimodalDomain) AnalyzeDocument(ctx context.Context, document *content.Content, schema *ExtractionSchema) (*AnalysisResult, error) {
	examples, _ := d.GetFewShotExamples("document_analysis", 2)

	messages := []*content.Message{
		content.NewTextMessage(content.RoleSystem, d.GetSystemPrompt()),
	}

	// Add few-shot examples
	for _, ex := range examples {
		messages = append(messages,
			content.NewTextMessage(content.RoleUser, ex.Input),
			content.NewTextMessage(content.RoleAssistant, ex.Output),
		)
	}

	// Build extraction prompt
	var prompt string
	if schema != nil {
		schemaJSON, _ := json.MarshalIndent(schema, "", "  ")
		prompt = fmt.Sprintf(`Analyze this document and extract data according to this schema:

%s

Extract all matching data and return as JSON.`, string(schemaJSON))
	} else {
		prompt = `Analyze this document and extract:
1. All text content
2. Key entities (names, dates, amounts, etc.)
3. Any tables or structured data
4. Document classification/tags

Return as structured JSON.`
	}

	userMsg := content.NewUserMessage(prompt)
	userMsg.Contents = append(userMsg.Contents, document)

	messages = append(messages, userMsg)

	resp, err := d.provider.Generate(ctx, messages, &llm.GenerationConfig{
		Temperature: 0.2,
		MaxTokens:   4096,
	})
	if err != nil {
		return nil, err
	}

	responseText := resp.Message.GetText()
	jsonStr := extractMultimodalJSON(responseText)

	var result AnalysisResult
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		result = AnalysisResult{
			Type:          "document",
			ExtractedText: responseText,
		}
	}
	result.Type = "document"

	return &result, nil
}

// ExtractToJSON extracts data from content according to a schema.
func (d *MultimodalDomain) ExtractToJSON(ctx context.Context, contentItem *content.Content, schema *ExtractionSchema) (map[string]any, error) {
	schemaJSON, _ := json.MarshalIndent(schema, "", "  ")

	prompt := fmt.Sprintf(`Extract data from the provided content according to this schema:

%s

Return ONLY a valid JSON object with the extracted data. Use null for missing required fields.`, string(schemaJSON))

	messages := []*content.Message{
		content.NewTextMessage(content.RoleSystem, d.GetSystemPrompt()),
	}

	userMsg := content.NewUserMessage(prompt)
	userMsg.Contents = append(userMsg.Contents, contentItem)
	messages = append(messages, userMsg)

	resp, err := d.provider.Generate(ctx, messages, &llm.GenerationConfig{
		Temperature: 0.1,
		MaxTokens:   2048,
	})
	if err != nil {
		return nil, err
	}

	responseText := resp.Message.GetText()
	jsonStr := extractMultimodalJSON(responseText)

	var result map[string]any
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		return nil, fmt.Errorf("failed to parse extracted JSON: %w", err)
	}

	return result, nil
}

// GenerateFromPrompt generates output based on multimodal input and text prompt.
func (d *MultimodalDomain) GenerateFromPrompt(ctx context.Context, contents []*content.Content, prompt string, outputFormat string) (string, error) {
	messages := []*content.Message{
		content.NewTextMessage(content.RoleSystem, d.GetSystemPrompt()),
	}

	// Build the user message with all content
	formatInstruction := ""
	switch outputFormat {
	case "json":
		formatInstruction = "\n\nRespond with valid JSON only."
	case "markdown":
		formatInstruction = "\n\nFormat your response as Markdown."
	case "csv":
		formatInstruction = "\n\nFormat your response as CSV."
	}

	userMsg := content.NewUserMessage(prompt + formatInstruction)
	userMsg.Contents = append(userMsg.Contents, contents...)
	messages = append(messages, userMsg)

	resp, err := d.provider.Generate(ctx, messages, &llm.GenerationConfig{
		Temperature: 0.3,
		MaxTokens:   4096,
	})
	if err != nil {
		return "", err
	}

	return resp.Message.GetText(), nil
}

// BatchAnalyze analyzes multiple content items.
func (d *MultimodalDomain) BatchAnalyze(ctx context.Context, items []*content.Content, schema *ExtractionSchema) ([]*AnalysisResult, error) {
	var results []*AnalysisResult

	for _, item := range items {
		var result *AnalysisResult
		var err error

		switch item.Type {
		case content.TypeImage:
			result, err = d.AnalyzeImage(ctx, item, "")
		case content.TypeDocument:
			result, err = d.AnalyzeDocument(ctx, item, schema)
		default:
			result, err = d.AnalyzeDocument(ctx, item, schema)
		}

		if err != nil {
			results = append(results, &AnalysisResult{
				Type:        string(item.Type),
				Description: fmt.Sprintf("Error: %v", err),
			})
			continue
		}
		results = append(results, result)
	}

	return results, nil
}

// CompareContents compares multiple content items.
func (d *MultimodalDomain) CompareContents(ctx context.Context, contents []*content.Content, prompt string) (string, error) {
	basePrompt := "Compare the following content items and describe the similarities and differences."
	if prompt != "" {
		basePrompt = prompt
	}

	messages := []*content.Message{
		content.NewTextMessage(content.RoleSystem, d.GetSystemPrompt()),
	}

	userMsg := content.NewUserMessage(basePrompt)
	userMsg.Contents = append(userMsg.Contents, contents...)
	messages = append(messages, userMsg)

	resp, err := d.provider.Generate(ctx, messages, &llm.GenerationConfig{
		Temperature: 0.4,
		MaxTokens:   2048,
	})
	if err != nil {
		return "", err
	}

	return resp.Message.GetText(), nil
}

// Common extraction schemas

// InvoiceSchema returns a schema for invoice extraction.
func InvoiceSchema() *ExtractionSchema {
	return &ExtractionSchema{
		Name:         "invoice",
		Description:  "Extract invoice data",
		OutputFormat: "json",
		Fields: []SchemaField{
			{Name: "invoice_number", Type: "string", Description: "Invoice number/ID", Required: true},
			{Name: "invoice_date", Type: "date", Description: "Invoice date", Required: true},
			{Name: "due_date", Type: "date", Description: "Payment due date", Required: false},
			{Name: "vendor_name", Type: "string", Description: "Vendor/seller name", Required: true},
			{Name: "vendor_address", Type: "string", Description: "Vendor address", Required: false},
			{Name: "customer_name", Type: "string", Description: "Customer/buyer name", Required: false},
			{Name: "line_items", Type: "array", Description: "List of line items with description, quantity, unit_price, amount", Required: false},
			{Name: "subtotal", Type: "number", Description: "Subtotal before tax", Format: "currency", Required: false},
			{Name: "tax", Type: "number", Description: "Tax amount", Format: "currency", Required: false},
			{Name: "total", Type: "number", Description: "Total amount due", Format: "currency", Required: true},
		},
	}
}

// ReceiptSchema returns a schema for receipt extraction.
func ReceiptSchema() *ExtractionSchema {
	return &ExtractionSchema{
		Name:         "receipt",
		Description:  "Extract receipt data",
		OutputFormat: "json",
		Fields: []SchemaField{
			{Name: "merchant_name", Type: "string", Description: "Store/merchant name", Required: true},
			{Name: "date", Type: "date", Description: "Transaction date", Required: true},
			{Name: "time", Type: "string", Description: "Transaction time", Required: false},
			{Name: "items", Type: "array", Description: "List of purchased items", Required: false},
			{Name: "subtotal", Type: "number", Description: "Subtotal", Format: "currency", Required: false},
			{Name: "tax", Type: "number", Description: "Tax", Format: "currency", Required: false},
			{Name: "total", Type: "number", Description: "Total paid", Format: "currency", Required: true},
			{Name: "payment_method", Type: "string", Description: "Payment method used", Required: false},
		},
	}
}

// BusinessCardSchema returns a schema for business card extraction.
func BusinessCardSchema() *ExtractionSchema {
	return &ExtractionSchema{
		Name:         "business_card",
		Description:  "Extract business card data",
		OutputFormat: "json",
		Fields: []SchemaField{
			{Name: "name", Type: "string", Description: "Person's full name", Required: true},
			{Name: "title", Type: "string", Description: "Job title", Required: false},
			{Name: "company", Type: "string", Description: "Company name", Required: false},
			{Name: "email", Type: "string", Description: "Email address", Format: "email", Required: false},
			{Name: "phone", Type: "string", Description: "Phone number", Format: "phone", Required: false},
			{Name: "mobile", Type: "string", Description: "Mobile number", Format: "phone", Required: false},
			{Name: "address", Type: "string", Description: "Address", Required: false},
			{Name: "website", Type: "string", Description: "Website URL", Required: false},
			{Name: "linkedin", Type: "string", Description: "LinkedIn profile", Required: false},
		},
	}
}

func extractMultimodalJSON(text string) string {
	if start := strings.Index(text, "```json"); start != -1 {
		start += 7
		if end := strings.Index(text[start:], "```"); end != -1 {
			return strings.TrimSpace(text[start : start+end])
		}
	}
	if start := strings.Index(text, "```"); start != -1 {
		start += 3
		if end := strings.Index(text[start:], "```"); end != -1 {
			return strings.TrimSpace(text[start : start+end])
		}
	}
	if start := strings.Index(text, "{"); start != -1 {
		depth := 0
		for i := start; i < len(text); i++ {
			if text[i] == '{' {
				depth++
			} else if text[i] == '}' {
				depth--
				if depth == 0 {
					return text[start : i+1]
				}
			}
		}
	}
	return text
}
