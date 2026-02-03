package domains

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/sujit/ai-agent/pkg/content"
	"github.com/sujit/ai-agent/pkg/llm"
	"github.com/sujit/ai-agent/pkg/storage"
)

// HealthcareDomain handles healthcare-related generation including medical coding and billing.
type HealthcareDomain struct {
	*BaseDomain
}

// ICD10Code represents an ICD-10 diagnostic code.
type ICD10Code struct {
	Code        string   `json:"code"`
	Description string   `json:"description"`
	Category    string   `json:"category"`
	Subcategory string   `json:"subcategory,omitempty"`
	Specificity string   `json:"specificity,omitempty"` // unspecified, initial, subsequent, sequela
	Billable    bool     `json:"billable"`
	Chapter     string   `json:"chapter,omitempty"`
	Synonyms    []string `json:"synonyms,omitempty"`
}

// CPTCode represents a CPT procedure code.
type CPTCode struct {
	Code           string   `json:"code"`
	Description    string   `json:"description"`
	Category       string   `json:"category"`
	Subcategory    string   `json:"subcategory,omitempty"`
	WorkRVU        float64  `json:"work_rvu,omitempty"`
	PracticeRVU    float64  `json:"practice_expense_rvu,omitempty"`
	MalpracticeRVU float64  `json:"malpractice_rvu,omitempty"`
	TotalRVU       float64  `json:"total_rvu,omitempty"`
	GlobalPeriod   string   `json:"global_period,omitempty"`
	Modifiers      []string `json:"modifiers,omitempty"`
}

// MedicalCodingResult represents the result of medical coding analysis.
type MedicalCodingResult struct {
	DiagnosisCodes     []ICD10Code    `json:"diagnosis_codes"`
	ProcedureCodes     []CPTCode      `json:"procedure_codes"`
	Modifiers          []string       `json:"modifiers,omitempty"`
	PrimaryDiagnosis   string         `json:"primary_diagnosis,omitempty"`
	SecondaryDiagnoses []string       `json:"secondary_diagnoses,omitempty"`
	PlaceOfService     string         `json:"place_of_service,omitempty"`
	Notes              []string       `json:"notes,omitempty"`
	Confidence         float64        `json:"confidence,omitempty"`
	Metadata           map[string]any `json:"metadata,omitempty"`
}

// ClaimData represents billing claim data.
type ClaimData struct {
	ClaimID           string           `json:"claim_id"`
	PatientID         string           `json:"patient_id"`
	ProviderNPI       string           `json:"provider_npi"`
	FacilityNPI       string           `json:"facility_npi,omitempty"`
	DateOfService     time.Time        `json:"date_of_service"`
	PlaceOfService    string           `json:"place_of_service"`
	DiagnosisCodes    []string         `json:"diagnosis_codes"`
	ProcedureCodes    []ClaimProcedure `json:"procedure_codes"`
	TotalCharges      float64          `json:"total_charges"`
	ExpectedReimburse float64          `json:"expected_reimbursement,omitempty"`
	PayerInfo         *PayerInfo       `json:"payer_info,omitempty"`
	Status            string           `json:"status"` // pending, submitted, approved, denied, partial
	DenialReason      string           `json:"denial_reason,omitempty"`
}

// ClaimProcedure represents a procedure on a claim.
type ClaimProcedure struct {
	CPTCode      string   `json:"cpt_code"`
	Modifiers    []string `json:"modifiers,omitempty"`
	Units        int      `json:"units"`
	ChargeAmount float64  `json:"charge_amount"`
	DiagPointers []int    `json:"diagnosis_pointers"` // 1-based index to diagnosis codes
}

// PayerInfo represents insurance payer information.
type PayerInfo struct {
	PayerID       string `json:"payer_id"`
	PayerName     string `json:"payer_name"`
	PlanType      string `json:"plan_type"` // commercial, medicare, medicaid, tricare
	MemberID      string `json:"member_id"`
	GroupNumber   string `json:"group_number,omitempty"`
	Authorization string `json:"authorization,omitempty"`
}

// ClinicalNote represents clinical documentation.
type ClinicalNote struct {
	NoteType       string            `json:"note_type"` // HPI, ROS, PE, A/P, progress
	Provider       string            `json:"provider"`
	DateOfService  time.Time         `json:"date_of_service"`
	ChiefComplaint string            `json:"chief_complaint,omitempty"`
	HPI            string            `json:"hpi,omitempty"`
	ROS            map[string]string `json:"ros,omitempty"`
	PhysicalExam   map[string]string `json:"physical_exam,omitempty"`
	Assessment     string            `json:"assessment,omitempty"`
	Plan           string            `json:"plan,omitempty"`
	RawText        string            `json:"raw_text,omitempty"`
}

// NewHealthcareDomain creates a new healthcare domain trainer.
func NewHealthcareDomain(provider llm.MultimodalProvider, storage *storage.Storage) *HealthcareDomain {
	return &HealthcareDomain{
		BaseDomain: NewBaseDomain("healthcare", "Medical Coding & Billing", provider, storage),
	}
}

// GetSystemPrompt returns the system prompt for healthcare coding.
func (d *HealthcareDomain) GetSystemPrompt() string {
	return `You are an expert medical coding and billing specialist with deep knowledge of:

ICD-10-CM/PCS Coding:
- Diagnostic coding with proper specificity
- Laterality, episode of care, and sequela
- External cause codes and manifestation codes
- Coding conventions and guidelines

CPT/HCPCS Coding:
- Evaluation and Management (E/M) levels
- Surgical procedures and modifiers
- Medicine, radiology, pathology codes
- Appropriate modifier usage (25, 59, 76, 77, etc.)

Medical Billing:
- Clean claim submission requirements
- Diagnosis pointer assignment
- Units of service calculation
- Bundling and unbundling rules

IMPORTANT DISCLAIMERS:
- This is for educational and reference purposes only
- All coding should be verified by certified coders
- Clinical documentation must support all codes assigned
- Compliance with payer-specific rules is essential

When analyzing clinical documentation:
1. Extract relevant diagnoses and procedures
2. Assign appropriate ICD-10 and CPT codes
3. Apply relevant modifiers
4. Note any documentation gaps
5. Provide confidence levels for each code`
}

// AnalyzeClinicalNote extracts medical codes from clinical documentation.
func (d *HealthcareDomain) AnalyzeClinicalNote(ctx context.Context, note *ClinicalNote) (*MedicalCodingResult, error) {
	examples, _ := d.GetFewShotExamples("coding", 3)

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

	noteText := note.RawText
	if noteText == "" {
		noteText = d.formatClinicalNote(note)
	}

	prompt := fmt.Sprintf(`Analyze this clinical documentation and extract appropriate medical codes:

%s

Respond with a JSON object containing:
- diagnosis_codes: Array of ICD-10 codes with descriptions
- procedure_codes: Array of CPT codes with descriptions
- primary_diagnosis: The principal diagnosis code
- modifiers: Any applicable modifiers
- notes: Coding rationale and documentation observations
- confidence: Overall confidence score (0.0 to 1.0)`, noteText)

	messages = append(messages, content.NewTextMessage(content.RoleUser, prompt))

	resp, err := d.provider.Generate(ctx, messages, &llm.GenerationConfig{
		Temperature: 0.2, // Lower temperature for accuracy
		MaxTokens:   2048,
	})
	if err != nil {
		return nil, err
	}

	responseText := resp.Message.GetText()
	jsonStr := extractHealthcareJSON(responseText)

	var result MedicalCodingResult
	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		return nil, fmt.Errorf("failed to parse coding result: %w", err)
	}

	return &result, nil
}

// GenerateClaimData generates billing claim data from coding results.
func (d *HealthcareDomain) GenerateClaimData(ctx context.Context, coding *MedicalCodingResult, patientID, providerNPI string, dateOfService time.Time) (*ClaimData, error) {
	claim := &ClaimData{
		ClaimID:        fmt.Sprintf("CLM%d", time.Now().UnixNano()),
		PatientID:      patientID,
		ProviderNPI:    providerNPI,
		DateOfService:  dateOfService,
		PlaceOfService: coding.PlaceOfService,
		Status:         "pending",
	}

	// Add diagnosis codes
	for _, dx := range coding.DiagnosisCodes {
		claim.DiagnosisCodes = append(claim.DiagnosisCodes, dx.Code)
	}

	// Add procedure codes with charge amounts
	for i, proc := range coding.ProcedureCodes {
		claimProc := ClaimProcedure{
			CPTCode:      proc.Code,
			Modifiers:    proc.Modifiers,
			Units:        1,
			ChargeAmount: d.estimateChargeAmount(proc),
			DiagPointers: []int{1}, // Default to first diagnosis
		}
		if i < len(coding.DiagnosisCodes) {
			claimProc.DiagPointers = []int{i + 1}
		}
		claim.ProcedureCodes = append(claim.ProcedureCodes, claimProc)
	}

	// Calculate total charges
	for _, proc := range claim.ProcedureCodes {
		claim.TotalCharges += proc.ChargeAmount * float64(proc.Units)
	}

	return claim, nil
}

// ValidateClaim validates a claim for common errors.
func (d *HealthcareDomain) ValidateClaim(claim *ClaimData) []string {
	var errors []string

	if claim.PatientID == "" {
		errors = append(errors, "Patient ID is required")
	}
	if claim.ProviderNPI == "" {
		errors = append(errors, "Provider NPI is required")
	}
	if len(claim.DiagnosisCodes) == 0 {
		errors = append(errors, "At least one diagnosis code is required")
	}
	if len(claim.ProcedureCodes) == 0 {
		errors = append(errors, "At least one procedure code is required")
	}

	// Validate diagnosis pointer references
	for i, proc := range claim.ProcedureCodes {
		for _, ptr := range proc.DiagPointers {
			if ptr < 1 || ptr > len(claim.DiagnosisCodes) {
				errors = append(errors, fmt.Sprintf("Procedure %d has invalid diagnosis pointer %d", i+1, ptr))
			}
		}
	}

	return errors
}

// SuggestModifiers suggests appropriate modifiers for a procedure.
func (d *HealthcareDomain) SuggestModifiers(ctx context.Context, cptCode string, clinicalContext string) ([]string, error) {
	prompt := fmt.Sprintf(`For CPT code %s with the following clinical context:

%s

Suggest appropriate modifiers and explain their use. Return as JSON array of modifier objects with code and rationale.`, cptCode, clinicalContext)

	messages := []*content.Message{
		content.NewTextMessage(content.RoleSystem, d.GetSystemPrompt()),
		content.NewTextMessage(content.RoleUser, prompt),
	}

	resp, err := d.provider.Generate(ctx, messages, &llm.GenerationConfig{
		Temperature: 0.2,
		MaxTokens:   1024,
	})
	if err != nil {
		return nil, err
	}

	// Parse modifiers from response
	var modifiers []string
	responseText := resp.Message.GetText()

	// Simple extraction of common modifiers mentioned
	commonModifiers := []string{"25", "26", "59", "76", "77", "LT", "RT", "50"}
	for _, mod := range commonModifiers {
		if strings.Contains(responseText, mod) {
			modifiers = append(modifiers, mod)
		}
	}

	return modifiers, nil
}

func (d *HealthcareDomain) formatClinicalNote(note *ClinicalNote) string {
	var parts []string

	if note.ChiefComplaint != "" {
		parts = append(parts, fmt.Sprintf("Chief Complaint: %s", note.ChiefComplaint))
	}
	if note.HPI != "" {
		parts = append(parts, fmt.Sprintf("History of Present Illness: %s", note.HPI))
	}
	if len(note.ROS) > 0 {
		parts = append(parts, "Review of Systems:")
		for system, finding := range note.ROS {
			parts = append(parts, fmt.Sprintf("  %s: %s", system, finding))
		}
	}
	if len(note.PhysicalExam) > 0 {
		parts = append(parts, "Physical Examination:")
		for system, finding := range note.PhysicalExam {
			parts = append(parts, fmt.Sprintf("  %s: %s", system, finding))
		}
	}
	if note.Assessment != "" {
		parts = append(parts, fmt.Sprintf("Assessment: %s", note.Assessment))
	}
	if note.Plan != "" {
		parts = append(parts, fmt.Sprintf("Plan: %s", note.Plan))
	}

	return strings.Join(parts, "\n\n")
}

func (d *HealthcareDomain) estimateChargeAmount(proc CPTCode) float64 {
	// Simple estimation based on RVU if available
	if proc.TotalRVU > 0 {
		return proc.TotalRVU * 35.0 // Approximate conversion factor
	}
	// Default estimates by category
	switch proc.Category {
	case "E/M":
		return 150.0
	case "Surgery":
		return 500.0
	case "Radiology":
		return 200.0
	case "Pathology":
		return 100.0
	default:
		return 100.0
	}
}

func extractHealthcareJSON(text string) string {
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
