#!/bin/bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$script_dir/../.." && pwd)"
cd "$root"
if [[ ! -d rl-env || ! -d fast-downward ]]; then
    echo 'Install this folder at rl-plan/experiments/full100-training first.' >&2
    exit 1
fi
mode="${1:-run}"
if [[ "$mode" != run && "$mode" != --prepare-only ]]; then
    echo 'Usage: bash experiments/full100-training/train-all.sh [--prepare-only]' >&2
    exit 1
fi
# Validate the environment before starting any expensive training.
python -c 'import chainer, chainerrl, numpy; print("Python environment imports OK")'
mkdir -p results-full100-v1-logs
for seed in 1 2 3; do
    for domain in barman blocksworld rovers childsnack sokoban visitall; do
        for model in original progress; do
            python "$script_dir/train-$model.py" --root "$root" --domain "$domain" \
                --seed "$seed" --split-seed 20260926 --steps 1000000 \
                --episode-limit 7500 --wall-seconds 300 --port 12631
        done
    done
done
echo 'Preflight complete for all 36 configurations. No evaluation will be launched.'
if [[ "$mode" == --prepare-only ]]; then exit 0; fi
for seed in 1 2 3; do
    for domain in barman blocksworld rovers childsnack sokoban visitall; do
        for model in original progress; do
            python -u "$script_dir/train-$model.py" --root "$root" --domain "$domain" \
                --seed "$seed" --split-seed 20260926 --steps 1000000 \
                --episode-limit 7500 --wall-seconds 300 --port 12631 --run \
                2>&1 | tee -a "results-full100-v1-logs/$domain-$model-seed$seed.log"
        done
    done
done
echo 'ALL 36 TRAINING RUNS COMPLETE. Final test evaluation has not been performed.'
