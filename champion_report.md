# Champion Model Selection Report

## Summary of Best Model: mistral:latest

The system tested all available models across 9 hyperparameter configurations and 3 different weight profiles. The champion was selected based on the **Accuracy Focused** profile (100% Accuracy, 0% Latency) to ensure maximum reliability for the 'medical-coding' domain.

### Evaluation Context

- **Domain**: medical-coding
- **Dataset**: data/examples/healthcare/examples.json (3 benchmarking samples)
- **Validation Logic**: Validated on 3 data points using grid search across 2 models.
- **Selection Logic**: Automated grid search across all model/hyperparameter combinations.

## Model Leaderboard (Primary Weights)

| Rank | Model | Accuracy | Avg Latency (ms) | Score | Best Params |
|---|---|---|---|---|---|
| 1 | mistral:latest 🏆 | 100.00% | 118076.90 | 100.0000 | T=0.0, P=1.0 |
| 2 | qwen2.5:14b | 100.00% | 102018.86 | 100.0000 | T=0.0, P=1.0 |

## Selection Reasoning & Validation Details

### Champion: mistral:latest

**Detailed Reasoning:**
Model mistral:latest achieved 100.00% accuracy with 118076.90ms average latency. Best hyperparameters: Temperature=0.0, TopP=1.00, TopK=1, MaxTokens=1024. Low temperature ensures deterministic and consistent outputs. Tested across 9 configurations. Accuracy was consistent (100.0%-100.0%) across configurations. 

**Validation Methodology:**
- Tested across all 9 hyperparameter configurations (deterministic through creative).
- Each configuration processed the full benchmark dataset.
- Results were compared against ground truth in the provided dataset.
- Latency was measured as end-to-end response time divided by sample count.

**Optimal Profile Found:**
- **Temperature**: 0.0
- **TopP**: 1.00
- **TopK**: 1
- **MaxTokens**: 1024

**All Test Results:**

| Trial | Temp | TopP | TopK | Tokens | Accuracy | Latency (ms) | Score |
|---|---|---|---|---|---|---|---|
| 1 | 0.0 | 1.00 | 1 | 1024 | 100.00% | 118076.90 | 100.0000 |
| 2 | 0.0 | 1.00 | 1 | 2048 | 100.00% | 91723.76 | 100.0000 |
| 3 | 0.1 | 0.90 | 40 | 1024 | 100.00% | 136034.00 | 100.0000 |
| 4 | 0.2 | 0.90 | 40 | 2048 | 100.00% | 115209.18 | 100.0000 |
| 5 | 0.3 | 0.95 | 50 | 1024 | 100.00% | 120475.29 | 100.0000 |
| 6 | 0.5 | 0.90 | 50 | 1024 | 100.00% | 95198.64 | 100.0000 |
| 7 | 0.7 | 0.90 | 50 | 2048 | 100.00% | 136376.62 | 100.0000 |
| 8 | 0.9 | 0.95 | 100 | 1024 | 100.00% | 140388.94 | 100.0000 |
| 9 | 1.0 | 1.00 | 100 | 2048 | 100.00% | 196753.27 | 100.0000 |

## Other Models Tested

### qwen2.5:14b

- Best Config: Temp=0.0, TopP=1.00, TopK=1, MaxTokens=1024
- Accuracy: 100.00%
- Latency: 102018.86 ms
- Score: 100.0000
- Tests Run: 9

**Analysis:** Model qwen2.5:14b achieved 100.00% accuracy with 102018.86ms average latency. Best hyperparameters: Temperature=0.0, TopP=1.00, TopK=1, MaxTokens=1024. Low temperature ensures deterministic and consistent outputs. Tested across 9 configurations. Accuracy was consistent (100.0%-100.0%) across configurations. 

---

*Report generated on Wed, 04 Feb 2026 21:14:08 +0545*
*Configuration saved to: config/domains.json*
