#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
py=../.venv/bin/python
free -g
case "${1:-}" in
  metatool)
    out=results/q1_e1/annotations
    mkdir -p "$out"
    exec > >(tee -a "$out/run.log") 2>&1
    for model in Qwen/Qwen3-0.6B HuggingFaceTB/SmolLM2-360M-Instruct \
      Qwen/Qwen2.5-0.5B-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct \
      Qwen/Qwen2.5-1.5B-Instruct ibm-granite/granite-3.1-2b-instruct \
      Qwen/Qwen3-1.7B microsoft/Phi-3.5-mini-instruct; do
      quantize=()
      case "$model" in
        *1.5B*|*1.7B*|*2b-instruct|*mini-instruct) quantize=(--quantize "$model");;
      esac
      /usr/bin/time -f 'Peak host RSS %M KiB' nice -n 10 "$py" annotate_bfcl.py \
        --items data/metatool_items.jsonl --models "$model" --bs 2 --out "$out" "${quantize[@]}"
      test -s "$out/annot_${model##*/}.jsonl"
    done
    ;;
  qlora)
    out=results/q1_gpu
    mkdir -p "$out"
    exec > >(tee -a "$out/run.log") 2>&1
    for noise in context source; do
      for rho in 0 .02 .05 .1 .25 .5 1; do
        dir="$out/$noise/$rho"
        if [[ -e "$dir/results.jsonl" ]]; then
          echo "Existing result at $dir/results.jsonl"
          exit 1
        fi
        methods=(verified_only)
        if [[ "$rho" == .02 || "$rho" == .05 ]]; then
          methods+=(source_prior ctx_eb ca_eb)
        fi
        /usr/bin/time -f 'Peak host RSS %M KiB' nice -n 10 "$py" kaggle_qlora.py \
          --noise-type "$noise" --noise .4 --verify-rate "$rho" --seeds 0 1 2 3 4 \
          --methods "${methods[@]}" --batch 2 --eval-bs 4 --grad-ckpt --amp fp16 \
          --skip-frozen --n-gen 0 --out "$dir"
      done
    done
    ;;
  *) echo "Usage: bash scripts/q1_local_gpu.sh {metatool|qlora}" >&2; exit 2;;
esac
