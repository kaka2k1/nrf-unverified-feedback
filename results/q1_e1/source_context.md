# Real LLM feedback on MetaTool — go/no-go analysis (auto-generated)

## Accuracy of each feedback source by context (%)

|                                        |   Phi-3.5-mini-instruct |   Qwen2.5-0.5B-Instruct |   Qwen2.5-1.5B-Instruct |   Qwen3-0.6B |   Qwen3-1.7B |   SmolLM2-1.7B-Instruct |   SmolLM2-360M-Instruct |   granite-3.1-2b-instruct |
|:---------------------------------------|------------------------:|------------------------:|------------------------:|-------------:|-------------:|------------------------:|------------------------:|--------------------------:|
| 0: keyword, seo, website (n=80)        |                    57.5 |                    31.2 |                    47.5 |         47.5 |         52.5 |                    42.5 |                    27.5 |                      58.8 |
| 1: text, audio, scenexplain (n=115)    |                    73.9 |                    55.7 |                    69.6 |         62.6 |         72.2 |                    56.5 |                    27   |                      74.8 |
| 2: hotel, travel, restaurants (n=80)   |                    71.2 |                    38.8 |                    57.5 |         48.8 |         63.7 |                    45   |                    18.8 |                      75   |
| 3: books, music, api (n=175)           |                    53.7 |                    36   |                    50.3 |         48.6 |         52.6 |                    32.6 |                    26.3 |                      59.4 |
| 4: chess, play, game (n=80)            |                    75   |                    36.2 |                    53.8 |         52.5 |         70   |                    57.5 |                    28.7 |                      73.8 |
| 5: global, world, job (n=195)          |                    84.1 |                    44.6 |                    71.8 |         68.7 |         71.3 |                    49.2 |                    27.7 |                      80   |
| 6: user, form, pdf (n=160)             |                    78.1 |                    43.1 |                    58.1 |         60.6 |         70.6 |                    46.9 |                    23.8 |                      71.9 |
| 7: assistant, council, planfit (n=110) |                    64.5 |                    24.5 |                    46.4 |         44.5 |         63.6 |                    37.3 |                    23.6 |                      59.1 |
| ALL                                    |                    70.6 |                    39.7 |                    58.2 |         55.9 |         64.9 |                    45.2 |                    25.6 |                      69.5 |

## Is reliability context-dependent? (sources = ['Phi-3.5-mini-instruct', 'Qwen2.5-0.5B-Instruct', 'Qwen2.5-1.5B-Instruct', 'Qwen3-1.7B', 'SmolLM2-1.7B-Instruct', 'SmolLM2-360M-Instruct', 'granite-3.1-2b-instruct'])

Likelihood-ratio test, correct ~ source vs ~ source + context + source×context: LR = 211.2, df ≈ 64, p = 1.26e-17. Overall feedback error (uniform source mix) = 0.466.
Per-source accuracy range across contexts (max−min, pts): Phi-3.5-mini-instruct: 30.4, Qwen2.5-0.5B-Instruct: 31.1, Qwen2.5-1.5B-Instruct: 25.4, Qwen3-1.7B: 19.7, SmolLM2-1.7B-Instruct: 24.9, SmolLM2-360M-Instruct: 10.0, granite-3.1-2b-instruct: 21.3
Interaction conditional on both main effects: LR = 62.9, df = 42, nominal p = 1.99e-02. Chi-square p-values are approximate and treat repeated annotations of an item as independent.

## Weighted label precision of the training set (50 random source assignments)

|                              |   label_precision |   lp_sd |   accepted_noisy |   accepted_clean |   n_eff |    brier |
|:-----------------------------|------------------:|--------:|-----------------:|-----------------:|--------:|---------:|
| (0.1, 'ctx_eb')              |            0.7519 |  0.0655 |           0.5333 |           0.715  | 320.475 |   0.2389 |
| (0.1, 'ctx_prior')           |            0.7468 |  0.0323 |           0.4619 |           0.646  | 339.028 |   0.2451 |
| (0.1, 'instance_rel')        |            0.7509 |  0.0359 |           0.4597 |           0.6492 | 333.161 |   0.2424 |
| (0.1, 'naive')               |            0.5311 |  0.0116 |           1      |           1      | 995     | nan      |
| (0.1, 'source_prior')        |            0.757  |  0.0358 |           0.4738 |           0.6993 | 318.992 |   0.2353 |
| (0.1, 'verifier_corrected')  |            0.5795 |  0.0127 |           1      |           1      | 995     | nan      |
| (0.25, 'ctx_eb')             |            0.8484 |  0.022  |           0.4878 |           0.739  | 437.589 |   0.2298 |
| (0.25, 'ctx_prior')          |            0.8435 |  0.0194 |           0.4492 |           0.6793 | 453.982 |   0.2369 |
| (0.25, 'instance_rel')       |            0.8451 |  0.0199 |           0.4487 |           0.6898 | 456.36  |   0.2334 |
| (0.25, 'naive')              |            0.5311 |  0.0116 |           1      |           1      | 995     | nan      |
| (0.25, 'source_prior')       |            0.8444 |  0.0235 |           0.4975 |           0.7455 | 441.938 |   0.2301 |
| (0.25, 'verifier_corrected') |            0.6477 |  0.0129 |           1      |           1      | 995     | nan      |
verify 0.1: ctx_eb − source_prior label precision = -0.51 ± 1.36 pts; median EB m_ctx = 4096.0
verify 0.25: ctx_eb − source_prior label precision = +0.40 ± 0.27 pts; median EB m_ctx = 4096.0
