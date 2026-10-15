#!/usr/bin/env bash
# Benchmark FlashMap drafts in sglang against the target alone and DFlash, with
# sglang.bench_serving on chat-templated prompts in its "custom" format (one
# {"conversations": [{"role": "user", "content": <templated prompt>}, ...]} per
# line, e.g. the FlashMap release's bench_prompts/<task>.jsonl).
#
# One server at a time; each is started, checked to run its method, warmed up,
# then timed on every cell, then stopped. Servers are listed in SERVERS as
# name=method[:draft dir][:env assignments separated by +][:extra server
# arguments separated by +]:
#
#   target                    the target alone
#   flashmap:<export>         FlashMap via DSPARK (python -m specforge.export.flashmap);
#                             flashmap@<g>:<export> verifies only the first g drafts
#   dflash:<draft>            sglang's DFlash (e.g. z-lab/Qwen3-8B-DFlash-b16)
#   dspark:<draft>            another DSPARK draft; extra modules via SGLANG_EXTERNAL_MODEL_PACKAGE in its env
#
#   TARGET=Qwen/Qwen3-8B PROMPTS=<dir of task.jsonl> OUT=logs/fm8b \
#   SERVERS="target=target fm1=flashmap:/exports/fm8b fm2=flashmap:/exports/fm8b:FLASHMAP_NUM_STEPS=2" \
#       benchmarks/flashmap/bench_sglang.sh
#
# Cells (CELLS, "task:concurrency:prompts:temperature" each) default to
# gsm8k/humaneval/mt-bench at concurrency 1, 4 and 16, greedy, plus gsm8k at
# temperature 1. Outputs stop at EOS or MAX_TOKENS (1024). Results: one JSON
# line per cell in $OUT/results-<server>.jsonl; summary.py prints the table.
# OUTPUT_DETAILS=1 also records every generated text (to compare servers' greedy
# outputs with compare_outputs.py); GREEDY_IDS=<n> saves the token ids of greedy
# outputs for the first n prompts of each task (for check_greedy.py). GPU_LOCK=<file> holds an exclusive flock on it for each server's lifetime
# (for GPUs shared under that protocol).
set -uo pipefail

TARGET=${TARGET:?set TARGET (target model path or id)}
PROMPTS=${PROMPTS:?set PROMPTS (directory of <task>.jsonl in custom format)}
OUT=${OUT:-logs/flashmap_sglang}
OUTPUT_DETAILS=${OUTPUT_DETAILS:-}
GREEDY_IDS=${GREEDY_IDS:-}
SERVERS=${SERVERS:-target=target}
CELLS=${CELLS:-"gsm8k:1:16:0 humaneval:1:16:0 mt-bench:1:16:0 gsm8k:4:32:0 humaneval:4:32:0 mt-bench:4:32:0 gsm8k:16:64:0 humaneval:16:64:0 mt-bench:16:64:0 gsm8k:1:16:1"}
MAX_TOKENS=${MAX_TOKENS:-1024}
MEM_FRACTION=${MEM_FRACTION:-0.8}
MAX_RUNNING=${MAX_RUNNING:-32}
PORT=${PORT:-31000}
GPU_LOCK=${GPU_LOCK:-}
GPU_TURNSTILE=${GPU_TURNSTILE:-}
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}

mkdir -p "${OUT}"
log() { echo "[$(date '+%F %T')] $*"; }

server_cmd() {  # method draft [extra server args, '+'-separated]
    local method=$1 draft=$2 extra=${3:-}
    local cmd=(python -m sglang.launch_server --model-path "${TARGET}" --host 127.0.0.1
               --port "${PORT}" --dtype bfloat16 --mem-fraction-static "${MEM_FRACTION}"
               --max-running-requests "${MAX_RUNNING}" --cuda-graph-max-bs-decode "${MAX_RUNNING}")
    case ${method} in
        target) ;;
        flashmap*|dspark*)  # flashmap@<g>: verify only the first g of the 16 drafts
            local gamma=16
            [[ ${method} == *@* ]] && gamma=${method#*@}
            cmd+=(--speculative-algorithm DSPARK --speculative-draft-model-path "${draft}"
                  --speculative-dspark-block-size "${gamma}") ;;
        dflash)
            cmd+=(--speculative-algorithm DFLASH --speculative-draft-model-path "${draft}") ;;
        *) echo "unknown method ${method}" >&2; return 1 ;;
    esac
    [[ -n ${extra} ]] && cmd+=(${extra//+/ })
    echo "${cmd[@]}"
}

run_server() {  # name=method[:draft][:env][:extra server args]
    local name=${1%%=*} spec=${1#*=}
    local method draft environment extra
    IFS=: read -r method draft environment extra <<<"${spec}"
    local cmd
    cmd=$(server_cmd "${method}" "${draft}" "${extra}") || return 1
    log "${name}: ${cmd}"
    (
        export PYTHONPATH=${REPO}${PYTHONPATH:+:${PYTHONPATH}}
        export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=${SGLANG_ENABLE_OVERLAP_PLAN_STREAM:-1}
        [[ ${method} == flashmap* ]] && export SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang
        for assignment in ${environment//+/ }; do export "${assignment}"; done
        exec ${cmd}
    ) > "${OUT}/server-${name}.log" 2>&1 &
    local pid=$!
    local deadline=$(( SECONDS + 1200 ))
    until curl -sf -m 5 "http://127.0.0.1:${PORT}/health" > /dev/null; do
        if ! kill -0 "${pid}" 2> /dev/null || (( SECONDS > deadline )); then
            log "${name}: server did not come up; see ${OUT}/server-${name}.log"
            tail -n 30 "${OUT}/server-${name}.log"
            kill "${pid}" 2> /dev/null; wait "${pid}" 2> /dev/null
            return 1
        fi
        sleep 5
    done
    if [[ ${method} == flashmap* ]] && ! grep -q "type=FlashMapDraftModel" "${OUT}/server-${name}.log"; then
        log "${name}: the FlashMap draft did not load (sglang logged no type=FlashMapDraftModel)"
        kill "${pid}"; wait "${pid}" 2> /dev/null
        return 1
    fi
    log "${name}: healthy; benchmarking"
    python "${REPO}/benchmarks/flashmap/run_cells.py" --name "${name}" --target "${TARGET}" \
        --prompts "${PROMPTS}" --out "${OUT}" --cells "${CELLS}" --port "${PORT}" \
        --max-tokens "${MAX_TOKENS}" ${OUTPUT_DETAILS:+--output-details} \
        ${GREEDY_IDS:+--greedy-ids "${GREEDY_IDS}"} > "${OUT}/bench-${name}.log" 2>&1 \
        || log "${name}: benchmark failed; see ${OUT}/bench-${name}.log"
    kill "${pid}"; wait "${pid}" 2> /dev/null
    log "${name}: done"
}

for server in ${SERVERS}; do
    if [[ -n ${GPU_LOCK} ]]; then
        export -f run_server server_cmd log
        export TARGET PROMPTS OUT CELLS MAX_TOKENS MEM_FRACTION MAX_RUNNING PORT REPO OUTPUT_DETAILS GREEDY_IDS
        # GPU_TURNSTILE: taken first, so new shared holders queue behind us.
        ${GPU_TURNSTILE:+flock -x "${GPU_TURNSTILE}"} \
            flock -x "${GPU_LOCK}" bash -c "run_server '${server}'"
    else
        run_server "${server}"
    fi
done
python "${REPO}/benchmarks/flashmap/summary.py" "${OUT}" | tee "${OUT}/summary.txt"
