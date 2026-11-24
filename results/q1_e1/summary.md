# E1 MetaTool

Seeds 0–19. Accuracy and exchange rates have pointwise 95% seed-bootstrap intervals. Exchange rates invert the monotone verified-only curve. Infinite values exceed the reference curve. See comparisons.csv for paired differences, effect sizes, and Holm-adjusted p-values.

MetaTool Similar Tool Selection has 995 items. The BFCL protocol uses seven fixed annotators, uniform source draws, eight candidate-tool context clusters, and a frozen names-only bge-small learner. Qwen3-0.6B is the BFCL reference model and is excluded from the feedback mix. Each seed uses 300 test, 64 clean seed, and 631 adaptation items. The method roster matches BFCL, including its omission of the three full-confusion baselines for item-local candidate positions.

- **Context dependence.** The uniform-source feedback error is 46.6%. Source accuracy varies by 10.0–31.1 points across clusters. The interaction beyond both main effects has LR = 62.9, df = 42, nominal p = .0199. The independence approximation ignores repeated predictions per item. [Context audit](source_context.md)
- **Exchange rates.** The Ctx-EB point rates are 4.06× and 9.00× at 2% and 5%, and 1.80× at 25%. The CA-EB rates are 1.68×, 7.20× and 0.70×. Each of these CIs includes 1. This supports the qualitative decline toward parity at higher verification, with substantial uncertainty. The reference accuracy changes by 1.40 points from zero to full verification, so its inverse is sensitive to small accuracy changes.
- **Noisy-label baselines.** MACE and RGPO-style lead Ctx-EB in mean accuracy at 2% and 5%. Co-teaching-style trails Ctx-EB at all four budgets, and at 25% Ctx-EB has the highest mean among these methods and small-loss. None of the 80 proposed-method comparisons reaches 5% after Holm correction. The relative strength of some baselines replicates descriptively, and a blanket ordering is unsupported. [Paired comparisons](comparisons.csv)

| learner   | view   |   rho | method             | accuracy                | exchange rate       |
|:----------|:-------|------:|:-------------------|:------------------------|:--------------------|
| bge_small | names  |  0.02 | ca_eb              | 45.90% [44.77%, 47.15%] | 1.68 [0.00, 20.87]  |
| bge_small | names  |  0.02 | conf_filter        | 45.95% [44.92%, 47.13%] | 2.09 [0.00, 19.14]  |
| bge_small | names  |  0.02 | coteach            | 45.88% [44.88%, 46.92%] | 1.55 [0.00, 18.30]  |
| bge_small | names  |  0.02 | ctx_eb             | 46.08% [45.08%, 47.22%] | 4.06 [0.00, 22.14]  |
| bge_small | names  |  0.02 | mace               | 46.65% [45.53%, 47.92%] | 38.51 [3.13, inf]   |
| bge_small | names  |  0.02 | naive              | 45.85% [44.80%, 47.03%] | 1.27 [0.00, 19.20]  |
| bge_small | names  |  0.02 | oracle_clean       | 46.83% [45.68%, 47.98%] | 42.23 [4.30, inf]   |
| bge_small | names  |  0.02 | rgpo_style         | 46.42% [45.32%, 47.60%] | 20.50 [2.08, 39.91] |
| bge_small | names  |  0.02 | self_train         | 45.77% [44.68%, 46.97%] | 0.00 [0.00, 8.93]   |
| bge_small | names  |  0.02 | small_loss         | 46.13% [45.00%, 47.33%] | 5.00 [0.00, 39.52]  |
| bge_small | names  |  0.02 | source_prior       | 46.13% [44.97%, 47.33%] | 5.00 [0.00, 38.71]  |
| bge_small | names  |  0.02 | verified_only      | 45.70% [44.73%, 46.82%] | reference           |
| bge_small | names  |  0.02 | verifier_corrected | 46.37% [45.33%, 47.52%] | 19.00 [0.00, 47.32] |
| bge_small | names  |  0.05 | ca_eb              | 46.33% [45.22%, 47.52%] | 7.20 [0.00, 15.90]  |
| bge_small | names  |  0.05 | conf_filter        | 45.83% [44.80%, 46.98%] | 0.45 [0.00, 3.76]   |
| bge_small | names  |  0.05 | coteach            | 45.87% [44.85%, 46.92%] | 0.56 [0.00, 5.45]   |
| bge_small | names  |  0.05 | ctx_eb             | 46.48% [45.25%, 47.75%] | 9.00 [0.77, 19.00]  |
| bge_small | names  |  0.05 | mace               | 46.70% [45.40%, 48.07%] | 15.81 [1.50, inf]   |
| bge_small | names  |  0.05 | naive              | 45.85% [44.75%, 47.07%] | 0.51 [0.00, 8.29]   |
| bge_small | names  |  0.05 | oracle_clean       | 46.83% [45.68%, 48.03%] | 16.89 [1.88, inf]   |
| bge_small | names  |  0.05 | rgpo_style         | 46.60% [45.37%, 48.02%] | 15.00 [1.21, inf]   |
| bge_small | names  |  0.05 | self_train         | 46.08% [44.80%, 47.52%] | 1.63 [0.00, 15.10]  |
| bge_small | names  |  0.05 | small_loss         | 46.37% [45.18%, 47.72%] | 7.60 [0.00, 17.46]  |
| bge_small | names  |  0.05 | source_prior       | 46.22% [44.93%, 47.50%] | 2.71 [0.00, 17.22]  |
| bge_small | names  |  0.05 | verified_only      | 46.00% [44.82%, 47.28%] | reference           |
| bge_small | names  |  0.05 | verifier_corrected | 46.37% [45.37%, 47.48%] | 7.60 [0.00, 18.87]  |
| bge_small | names  |  0.1  | ca_eb              | 46.23% [45.10%, 47.47%] | 1.43 [0.00, 8.36]   |
| bge_small | names  |  0.1  | conf_filter        | 46.23% [45.10%, 47.45%] | 1.43 [0.00, 6.38]   |
| bge_small | names  |  0.1  | coteach            | 46.23% [45.00%, 47.53%] | 1.43 [0.00, 8.00]   |
| bge_small | names  |  0.1  | ctx_eb             | 46.40% [45.12%, 47.72%] | 4.00 [0.00, 8.15]   |
| bge_small | names  |  0.1  | mace               | 46.57% [45.52%, 47.75%] | 5.00 [0.46, 9.75]   |
| bge_small | names  |  0.1  | naive              | 45.85% [44.78%, 47.03%] | 0.25 [0.00, 3.81]   |
| bge_small | names  |  0.1  | oracle_clean       | 46.83% [45.70%, 48.02%] | 8.45 [0.90, inf]    |
| bge_small | names  |  0.1  | rgpo_style         | 46.25% [45.08%, 47.52%] | 1.50 [0.00, 6.15]   |
| bge_small | names  |  0.1  | self_train         | 46.25% [45.02%, 47.50%] | 1.50 [0.00, 7.83]   |
| bge_small | names  |  0.1  | small_loss         | 46.18% [45.08%, 47.33%] | 1.21 [0.00, 8.67]   |
| bge_small | names  |  0.1  | source_prior       | 46.53% [45.33%, 47.83%] | 4.80 [0.49, inf]    |
| bge_small | names  |  0.1  | verified_only      | 46.13% [45.00%, 47.27%] | reference           |
| bge_small | names  |  0.1  | verifier_corrected | 46.58% [45.53%, 47.68%] | 6.25 [0.32, inf]    |
| bge_small | names  |  0.25 | ca_eb              | 46.27% [45.15%, 47.47%] | 0.70 [0.00, 3.02]   |
| bge_small | names  |  0.25 | conf_filter        | 46.25% [45.17%, 47.48%] | 0.60 [0.00, 3.04]   |
| bge_small | names  |  0.25 | coteach            | 46.25% [45.17%, 47.40%] | 0.60 [0.00, 3.14]   |
| bge_small | names  |  0.25 | ctx_eb             | 46.48% [45.33%, 47.68%] | 1.80 [0.00, 3.70]   |
| bge_small | names  |  0.25 | mace               | 46.35% [45.30%, 47.45%] | 1.48 [0.00, 3.33]   |
| bge_small | names  |  0.25 | naive              | 45.85% [44.80%, 47.02%] | 0.10 [0.00, 1.65]   |
| bge_small | names  |  0.25 | oracle_clean       | 46.83% [45.68%, 47.97%] | 3.38 [0.35, inf]    |
| bge_small | names  |  0.25 | rgpo_style         | 46.40% [45.37%, 47.53%] | 1.60 [0.00, 3.26]   |
| bge_small | names  |  0.25 | self_train         | 46.12% [45.00%, 47.35%] | 0.37 [0.00, 1.72]   |
| bge_small | names  |  0.25 | small_loss         | 46.18% [45.13%, 47.25%] | 0.49 [0.00, 3.34]   |
| bge_small | names  |  0.25 | source_prior       | 46.22% [45.08%, 47.37%] | 0.54 [0.00, 3.22]   |
| bge_small | names  |  0.25 | verified_only      | 46.32% [45.30%, 47.42%] | reference           |
| bge_small | names  |  0.25 | verifier_corrected | 46.20% [45.05%, 47.35%] | 0.51 [0.00, 3.26]   |
