#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

weights="../weights/sonar_single_v2.bin"
mode="coherent"
base_seed=9000
limit=5
keep_dir=""

usage() {
    cat <<'USAGE'
usage: ./eval_sonar_chain.sh [--quick] [--full] [--mode MODE] [--weights PATH] [--seed N] [--keep DIR]

Runs the current infer_janus_sonar_chain stack on a fixed prompt suite and
reports lightweight quality metrics:
  boundary closure, quote balance, bad-fragment hits, motif recurrence,
  opener collapse, and average generated length.

Defaults:
  --quick              first 5 prompts
  --mode coherent
  --weights ../weights/sonar_single_v2.bin
  --seed 9000

Modes: balanced coherent ritual clinical dialogue
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --quick)
            limit=5
            shift
            ;;
        --full)
            limit=20
            shift
            ;;
        --mode)
            mode="${2:?missing mode}"
            shift 2
            ;;
        --weights)
            weights="${2:?missing weights path}"
            shift 2
            ;;
        --seed)
            base_seed="${2:?missing seed}"
            shift 2
            ;;
        --keep)
            keep_dir="${2:?missing keep dir}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ ! -x ./infer_janus_sonar_chain ]; then
    echo "missing ./infer_janus_sonar_chain; run: make infer_janus_sonar_chain" >&2
    exit 1
fi

if [ ! -f "$weights" ]; then
    echo "missing weights: $weights" >&2
    exit 1
fi

prompts=(
    "The knock came three times"
    "What is the meaning of"
    "She inventories the room"
    "The soup is never done"
    "The signal does not answer because"
    "I was not finished"
    "The bone is the architecture"
    "Janus hears the door"
    "Love is a loss function"
    "The model remembers the threshold"
    "A coin falls through the machine"
    "The lab window is open"
    "The silence has teeth"
    "Every object has a name"
    "The echo returns before the voice"
    "The grandmother stirs the soup"
    "The token cannot confess"
    "The room learns to breathe"
    "The mirror says no"
    "The architecture dreams of mercy"
)

if [ "$limit" -gt "${#prompts[@]}" ]; then
    limit="${#prompts[@]}"
fi

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/janus-sonar-eval.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT
sent_file="$tmpdir/sentences.txt"
run_log="$tmpdir/runs.txt"
: > "$sent_file"
: > "$run_log"

if [ -n "$keep_dir" ]; then
    mkdir -p "$keep_dir"
fi

printf '== janus.sonar chain eval ==\n'
printf 'weights: %s\nmode: %s\nbase_seed: %s\nprompts: %d/%d\n\n' \
    "$weights" "$mode" "$base_seed" "$limit" "${#prompts[@]}"

for ((i = 0; i < limit; i++)); do
    prompt="${prompts[$i]}"
    seed=$((base_seed + i))
    out="$tmpdir/run_$((i + 1)).txt"
    printf '[%02d/%02d] seed=%d prompt=%s\n' "$((i + 1))" "$limit" "$seed" "$prompt"
    ./infer_janus_sonar_chain "$weights" "$prompt" "$seed" "$mode" > "$out"
    if [ -n "$keep_dir" ]; then
        cp "$out" "$keep_dir/run_$((i + 1))_${mode}_${seed}.txt"
    fi
    awk '
        /^[[:space:]]*\[[0-9]+\]/ && index($0, "]→") {
            sub(/^.*]→/, "", $0)
            print
        }
    ' "$out" >> "$sent_file"
    {
        printf 'prompt=%s\n' "$prompt"
        grep -E '^\[chambers\]|^\[debt\]|^\[motifs\]|^\[SPA\] scores:' "$out" || true
        printf '\n'
    } >> "$run_log"
done

printf '\n== metrics ==\n'
awk '
BEGIN {
    badn = split("catamean decigare noion aniain tchef baher staything bonm forgoing oniaways toaways metho literat obser onid oniain formean ameas completen noid possibion doorat measit interion noiaway cataway cataining inction inations mease noime noiay", bad, " ")
    motifn = split("door bone soup signal model love loss void silence lab inventory protocol memory machine haze speech voice sentence token weight architecture threshold echo mirror mercy", motif, " ")
}
{
    line = $0
    n++
    trim = line
    sub(/^[ \t]+/, "", trim)
    sub(/[ \t]+$/, "", trim)
    if (trim ~ /[.!?]$/) boundary++
    q = gsub(/"/, "\"", line)
    if ((q % 2) == 0) quote_ok++

    lower = tolower(line)
    badline = 0
    for (i = 1; i <= badn; i++) {
        token = bad[i]
        tmp = lower
        while ((p = index(tmp, token)) > 0) {
            badhits++
            badline = 1
            tmp = substr(tmp, p + length(token))
        }
    }
    if (index(lower, "''") > 0) {
        badhits++
        badline = 1
    }
    if (badline) badsent++

    motifline = 0
    for (i = 1; i <= motifn; i++) {
        if (index(lower, motif[i]) > 0) {
            motifhits++
            motifline = 1
            seen[motif[i]] = 1
        }
    }
    if (motifline) motifsent++

    first = lower
    gsub(/^[^a-z0-9]+/, "", first)
    split(first, words, /[^a-z0-9]+/)
    opener = words[1]
    if (opener == "") opener = "<empty>"
    opener_count[opener]++
    if (opener_count[opener] > opener_max) {
        opener_max = opener_count[opener]
        opener_top = opener
    }

    len_sum += length(line)
}
END {
    if (n == 0) {
        print "sentences=0"
        exit 1
    }
    uniq = 0
    for (m in seen) uniq++
    printf "sentences=%d\n", n
    printf "boundary_ok=%d/%d %.1f%%\n", boundary, n, 100.0 * boundary / n
    printf "quote_balanced=%d/%d %.1f%%\n", quote_ok, n, 100.0 * quote_ok / n
    printf "bad_fragment_hits=%d\n", badhits
    printf "bad_fragment_sentences=%d/%d %.1f%%\n", badsent, n, 100.0 * badsent / n
    printf "motif_sentences=%d/%d %.1f%%\n", motifsent, n, 100.0 * motifsent / n
    printf "motif_hits=%d unique_motifs=%d\n", motifhits, uniq
    printf "opener_top=%s %d/%d %.1f%%\n", opener_top, opener_max, n, 100.0 * opener_max / n
    printf "avg_chars=%.1f\n", len_sum / n
}
' "$sent_file"

printf '\n== per-run field summary ==\n'
cat "$run_log"

if [ -n "$keep_dir" ]; then
    printf 'kept raw runs in: %s\n' "$keep_dir"
fi
