package main

import (
	"context"
	"fmt"
	"log"
	"path/filepath"
	"strings"

	"github.com/sujit/ai-agent/pkg/agent"
	"github.com/sujit/ai-agent/pkg/config"
	"github.com/sujit/ai-agent/pkg/content"
	"github.com/sujit/ai-agent/pkg/llm"
	"github.com/sujit/ai-agent/pkg/training"
)

func main() {
	ctx := context.Background()
	configDir := "./config"
	absPath, _ := filepath.Abs(configDir)

	fmt.Println("🚀 Advanced Domain Verification: Medical & Workflow")

	// 1. Initialize
	loader := config.NewConfigLoader(absPath)
	cfg, err := loader.Load()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	registry, _ := llm.NewProviderRegistryFromConfig(cfg)
	multimodalAgent := agent.NewMultimodalAgent("DomainTester", registry,
		agent.WithConfig(&agent.AgentConfig{
			DefaultModel:    "mistral",
			EnableStreaming: true,
			AutoPreprocess:  true,
			EnableRAG:       true,
		}),
	)

	provider, _ := registry.GetProvider(llm.ProviderOllama)
	trainer := training.NewDomainTrainer(provider.(llm.MultimodalProvider), nil)
	multimodalAgent.DomainTrainer = trainer

	fmt.Println("\n--- 1. Medical Coding Domain Verification ---")
	medDomain, _ := trainer.CreateDomain("Medical", "Specialized medical coding knowledge.")
	trainer.AddTerminology(medDomain.ID, "Acute Myocardial Infarction", "ICD-10-CM I21.9")
	trainer.AddGuideline(medDomain.ID, "Always output diagnosis in a structured table with Code and Description.")
	trainer.AddGuideline(medDomain.ID, "Use professional clinical terminology.")

	multimodalAgent.SetDomain(medDomain.ID)
	fmt.Println("👉 Querying Medical Domain (Diagnosis)...")
	respMed := streamChat(ctx, multimodalAgent, "Patient presents with chest pain and final diagnosis is Acute Myocardial Infarction. Provide the coding report.")

	if contains(respMed, "I21.9") {
		fmt.Println("\n✅ Correct ICD-10 code (I21.9) used!")
	}
	if contains(respMed, "|") {
		fmt.Println("✅ Table format detected!")
	}

	// Reset for next domain
	multimodalAgent.ClearConversation()
	fmt.Println("\n--- 2. Workflow Automation Domain Verification ---")
	wfDomain, _ := trainer.CreateDomain("Workflow", "Enterprise workflow and BPMN logic.")
	trainer.AddGuideline(wfDomain.ID, "Represent all workflows using Mermaid.js graph syntax (graph TD).")
	trainer.AddTerminology(wfDomain.ID, "Approval", "State: PENDING_REVIEW")

	multimodalAgent.SetDomain(wfDomain.ID)
	fmt.Println("👉 Querying Workflow Domain (Onboarding Flow)...")
	respWF := streamChat(ctx, multimodalAgent, "Design an employee onboarding workflow that includes a document sign-off and manager approval step.")

	if contains(respWF, "graph TD") || contains(respWF, "mermaid") {
		fmt.Println("\n✅ Mermaid.js workflow syntax detected!")
	}
	if contains(respWF, "PENDING_REVIEW") {
		fmt.Println("✅ Domain terminology (PENDING_REVIEW) correctly applied!")
	}

	fmt.Println("\n--- 3. Multi-turn Knowledge Retention (Format Check) ---")
	fmt.Println("👉 Testing Turn 2 within Medical Context...")
	multimodalAgent.SetDomain(medDomain.ID)
	multimodalAgent.ClearConversation()
	streamChat(ctx, multimodalAgent, "The patient also has history of Hypertension.")
	fmt.Println("\n👉 Turn 2: Recalling combined coding...")
	respFinal := streamChat(ctx, multimodalAgent, "What was the initial primary diagnosis code mentioned earlier for the chest pain?")

	if contains(respFinal, "I21.9") || contains(respFinal, "myocardial") {
		fmt.Println("\n✅ History and Domain terminology retained across turns!")
	}

	fmt.Println("\n✅ Domain Training Verification Complete!")
}

func streamChat(ctx context.Context, a *agent.MultimodalAgent, input string) string {
	msg := content.NewUserMessage(input)
	stream, err := a.Stream(ctx, msg)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return ""
	}

	var fullText strings.Builder
	for chunk := range stream {
		if chunk.Error != nil {
			fmt.Printf("\n[Stream Error]: %v\n", chunk.Error)
			break
		}
		fmt.Print(chunk.Delta)
		fullText.WriteString(chunk.Delta)
	}
	fmt.Println()
	return fullText.String()
}

func contains(s, substr string) bool {
	return strings.Contains(strings.ToLower(s), strings.ToLower(substr))
}
