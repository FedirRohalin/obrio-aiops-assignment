# LLM Evaluation Report: GPT-4o vs GPT-4o-mini

## 1. Executive Summary
This evaluation compares `gpt-4o` and `gpt-4o-mini` across a dataset of 4 customer support tickets to determine the optimal model for the Nebula AI Support Assistant. 

**Conclusion:** Both models achieved 100% accuracy in retrieving and citing the correct KB articles. However, `gpt-4o-mini` is recommended for production deployment as it delivers identical logical performance at **~6% of the cost** of `gpt-4o`, with negligible differences in response length and acceptable latency.

## 2. Aggregate Metrics
| Metric | `gpt-4o` | `gpt-4o-mini` | Comparison |
| :--- | :--- | :--- | :--- |
| **Accuracy (Pass Rate)** | 100% (4/4) | 100% (4/4) | Tie |
| **Avg Latency (seconds)** | 2.48 s | 2.95 s | `gpt-4o` is ~0.47s faster |
| **Total Prompt Tokens** | 4,610 | 4,610 | Identical (same prompts) |
| **Total Completion Tokens** | 805 | 872 | `gpt-4o-mini` is slightly more verbose |
| **Total Cost (USD)** | **$0.019575** | **$0.001215** | `gpt-4o-mini` is ~16x cheaper |

## 3. Per-Ticket Breakdown

| Ticket Index | Expected Article | `gpt-4o` Accuracy | `gpt-4o` Latency | `gpt-4o-mini` Accuracy | `gpt-4o-mini` Latency |
| :---: | :--- | :---: | :--- | :---: | :--- |
| **0** (Password Reset) | `KB-101` | ✅ Pass | 3.43 s | ✅ Pass | 2.94 s |
| **1** (Duplicate Charge) | `KB-201` | ✅ Pass | 2.55 s | ✅ Pass | 3.08 s |
| **2** (App Crash) | `KB-301` | ✅ Pass | 1.85 s | ✅ Pass | 3.20 s |
| **3** (Delete Account) | `KB-401` | ✅ Pass | 2.09 s | ✅ Pass | 2.57 s |

*Note: Latency values include network overhead. Tokens and pricing are based on standard OpenAI API JSON mode usage.*