#!/usr/bin/env bash

set -euo pipefail

DATASET="${1:-retailrocket}"
GPU_ID="${GPU_ID:-0}"

# Coarse grid first. Keep this small enough to finish in a practical time.
TAUS=(0.1 0.2 0.5)
LMD_SEMS=(0.2 0.3 0.5)
DROPOUTS=(0.2 0.3)
CONTRASTS=(us_x us)

OUT_DIR="../Log/DuoRec/${DATASET}/$(date +%d-%m-%Y)-sweep"
mkdir -p "${OUT_DIR}"
SUMMARY_FILE="${OUT_DIR}/summary.tsv"

echo -e "run_id\tcontrast\ttau\tlmd_sem\tdropout\tseed\tnote" > "${SUMMARY_FILE}"

run_id=0
for contrast in "${CONTRASTS[@]}"; do
  for tau in "${TAUS[@]}"; do
    for lmd_sem in "${LMD_SEMS[@]}"; do
      for dropout in "${DROPOUTS[@]}"; do
        run_id=$((run_id + 1))
        seed=$((2020 + run_id))
        run_log="${OUT_DIR}/run_${run_id}.log"

        echo "[Run ${run_id}] contrast=${contrast} tau=${tau} lmd_sem=${lmd_sem} dropout=${dropout} seed=${seed}"

        CUDA_VISIBLE_DEVICES="${GPU_ID}" python run_seq.py \
          --model=DuoRec \
          --dataset="${DATASET}" \
          --config_files="seq.yaml configs/duorec_retailrocket_tuned.yaml" \
          --contrast="${contrast}" \
          --tau="${tau}" \
          --lmd=0.1 \
          --lmd_sem="${lmd_sem}" \
          --hidden_dropout_prob="${dropout}" \
          --attn_dropout_prob="${dropout}" \
          --seed="${seed}" \
          > "${run_log}" 2>&1

        best_line=$(grep "best valid" "${run_log}" | tail -n 1 || true)
        test_line=$(grep "test result" "${run_log}" | tail -n 1 || true)

        note="${best_line} || ${test_line}"
        echo -e "${run_id}\t${contrast}\t${tau}\t${lmd_sem}\t${dropout}\t${seed}\t${note}" >> "${SUMMARY_FILE}"
      done
    done
  done
done

echo "Sweep completed. Summary: ${SUMMARY_FILE}"
