# Real LLM feedback on BFCL — go/no-go analysis (auto-generated)

## Accuracy of each feedback source by context (%)

|                                                |   Qwen2.5-0.5B-Instruct |   Qwen3-0.6B |   SmolLM2-1.7B-Instruct |   SmolLM2-360M-Instruct |   Phi-3.5-mini-instruct |   Qwen2.5-1.5B-Instruct |   Qwen3-1.7B |   granite-3.1-2b-instruct |
|:-----------------------------------------------|------------------------:|-------------:|------------------------:|------------------------:|------------------------:|------------------------:|-------------:|--------------------------:|
| 0: restaurants, restaurant, reservation (n=69) |                    76.8 |         88.4 |                    92.8 |                    94.2 |                    97.1 |                    92.8 |         89.9 |                      95.7 |
| 1: payment, receiver, initiates (n=45)         |                    88.9 |         68.9 |                    71.1 |                    73.3 |                    86.7 |                    51.1 |         66.7 |                      80   |
| 2: music, genre, media (n=156)                 |                    49.4 |         89.7 |                    91   |                    64.1 |                    94.9 |                    96.8 |         95.5 |                      94.9 |
| 3: service, profile, providers (n=76)          |                    92.1 |         93.4 |                    85.5 |                    72.4 |                    88.2 |                    92.1 |         93.4 |                      96.1 |
| 4: hotels, flights, class (n=142)              |                    64.8 |         97.2 |                    81   |                    53.5 |                    97.2 |                    95.1 |         95.8 |                      95.8 |
| 5: calculate, current, api (n=547)             |                    84.5 |         91   |                    89.4 |                    70.9 |                    93.2 |                    94   |         93.2 |                      94.9 |
| 6: events, cultural, tickets (n=122)           |                    78.7 |         91   |                    88.5 |                    50   |                    92.6 |                    93.4 |         95.9 |                      93.4 |
| 7: alarm, property, homes (n=93)               |                    79.6 |        100   |                    98.9 |                    71   |                   100   |                   100   |        100   |                     100   |
| ALL                                            |                    77.1 |         91.4 |                    88.6 |                    67.5 |                    94   |                    93.1 |         93.4 |                      94.8 |

## Is reliability context-dependent? (sources = ['Qwen2.5-0.5B-Instruct', 'SmolLM2-1.7B-Instruct', 'SmolLM2-360M-Instruct', 'Phi-3.5-mini-instruct', 'Qwen2.5-1.5B-Instruct', 'Qwen3-1.7B', 'granite-3.1-2b-instruct'])

Likelihood-ratio test, correct ~ source vs ~ source + context + source×context: LR = 371.6, df ≈ 64, p = 6.47e-45. Overall feedback error (uniform source mix) = 0.131.
Per-source accuracy range across contexts (max−min, pts): Qwen2.5-0.5B-Instruct: 42.7, SmolLM2-1.7B-Instruct: 27.8, SmolLM2-360M-Instruct: 44.2, Phi-3.5-mini-instruct: 13.3, Qwen2.5-1.5B-Instruct: 48.9, Qwen3-1.7B: 33.3, granite-3.1-2b-instruct: 20.0

## Weighted label precision of the training set (50 random source assignments)

|                              |   label_precision |   lp_sd |   accepted_noisy |   accepted_clean |    n_eff |    brier |
|:-----------------------------|------------------:|--------:|-----------------:|-----------------:|---------:|---------:|
| (0.1, 'ctx_eb')              |            0.8996 |  0.0143 |           0.9597 |           0.9862 |  968.623 |   0.1109 |
| (0.1, 'ctx_prior')           |            0.9048 |  0.0117 |           0.913  |           0.9733 |  983.245 |   0.1133 |
| (0.1, 'instance_rel')        |            0.901  |  0.0114 |           0.9579 |           0.9895 |  999.697 |   0.1094 |
| (0.1, 'naive')               |            0.8689 |  0.0084 |           1      |           1      | 1250     | nan      |
| (0.1, 'source_prior')        |            0.9015 |  0.0114 |           0.9864 |           0.9954 |  971.402 |   0.1089 |
| (0.1, 'verifier_corrected')  |            0.8829 |  0.0087 |           1      |           1      | 1250     | nan      |
| (0.25, 'ctx_eb')             |            0.9209 |  0.0084 |           0.9985 |           0.9999 | 1026.37  |   0.1062 |
| (0.25, 'ctx_prior')          |            0.9264 |  0.0083 |           0.9325 |           0.9829 | 1030.26  |   0.1076 |
| (0.25, 'instance_rel')       |            0.9246 |  0.0083 |           0.9582 |           0.9911 | 1043.02  |   0.1044 |
| (0.25, 'naive')              |            0.8689 |  0.0084 |           1      |           1      | 1250     | nan      |
| (0.25, 'source_prior')       |            0.9222 |  0.0078 |           1      |           1      | 1024.6   |   0.106  |
| (0.25, 'verifier_corrected') |            0.9023 |  0.0082 |           1      |           1      | 1250     | nan      |
verify 0.1: ctx_eb − source_prior label precision = -0.18 ± 0.15 pts; median EB m_ctx = 4096.0
verify 0.25: ctx_eb − source_prior label precision = -0.14 ± 0.05 pts; median EB m_ctx = 4096.0
