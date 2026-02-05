#!/bin/bash

# 1. Uncached Request (Stateless)
echo "Request 1 (Mistral/Ollama - Stateless - Expect Slow)..."
time curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain the pathophysiology of acute asthma exacerbation briefly.",
    "domain": "healthcare",
    "stateless": true
  '

echo -e "\n\nwaiting 2 seconds...\n"
sleep 2

# 2. Cached Request (Stateless)
echo "Request 2 (mistral/ollama - Stateless - Expect INSTANT)..."
time curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain the pathophysiology of acute asthma exacerbation briefly.",
    "domain": "healthcare",
    "stateless": true
  '
