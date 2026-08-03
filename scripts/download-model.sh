#!/usr/bin/env bash
# download-model.sh, fetch the Kimi K3 checkpoint and verify it byte-exactly.
#
#   scripts/download-model.sh <dest_dir>
#
# The checkpoint is 1.56 TB across 96 safetensors shards. A partial or corrupt download
# does not fail loudly, it produces wrong tokens, so the byte total is checked against
# the published figure before anything else is allowed to use it.
#
# Requires a HuggingFace token in $HF_TOKEN or ~/.cache/huggingface/token.

set -euo pipefail

DEST="${1:?usage: download-model.sh <dest_dir>}"
REPO="moonshotai/Kimi-K3"

# Published totals for the released checkpoint. Verified, not assumed.
EXPECT_SHARDS=96
EXPECT_BYTES=1560936091448

command -v python3 >/dev/null || { echo "python3 required"; exit 1; }

# `hf download` is the supported CLI entry point. Check for checksum verification too:
# an older installation may provide `hf` but not `hf cache verify`.
if ! command -v hf >/dev/null 2>&1 || ! hf cache verify --help >/dev/null 2>&1; then
    echo "installing/upgrading huggingface_hub CLI…"
    python3 -m pip install --quiet --upgrade "huggingface_hub[cli]"
    hash -r
fi

command -v hf >/dev/null 2>&1 || {
    echo "hf CLI required; install or upgrade huggingface_hub"
    exit 1
}
hf cache verify --help >/dev/null 2>&1 || {
    echo "hf cache verify unavailable; upgrade huggingface_hub"
    exit 1
}

# Resolve `main` once, before downloading. Both download and verification then target the
# same immutable Hub commit rather than allowing `main` to move between the two steps.
REVISION="${K3_REVISION:-}"
if [ -z "$REVISION" ]; then
    REVISION="$(
        python3 - "$REPO" <<'PY'
import sys
from huggingface_hub import HfApi

info = HfApi().model_info(sys.argv[1])
if not info.sha:
    raise SystemExit("Hub did not return an immutable revision")
print(info.sha)
PY
    )"
fi

[ -n "$REVISION" ] || {
    echo "could not resolve model revision"
    exit 1
}

mkdir -p "$DEST"

echo "downloading $REPO@$REVISION -> $DEST"
echo "  1.56 TB across $EXPECT_SHARDS shards; expect ~30 min at 1 GB/s"
echo

# The token is read from the environment or the saved Hugging Face login and never
# echoed. Pinning the revision keeps download and verification on the same snapshot.
HF_HUB_ENABLE_HF_TRANSFER=1 \
hf download "$REPO" \
    --revision "$REVISION" \
    --local-dir "$DEST" \
    --max-workers 16

echo
echo "verifying…"
# find, not `ls | wc -l`: under `set -euo pipefail` a glob that matches nothing makes
# ls exit non-zero and the script dies HERE, so the "download is incomplete" message
# below -- the entire point of this verification block -- would never be reached.
N=$(find "$DEST" -maxdepth 1 -name '*.safetensors' | wc -l)
B=$(find "$DEST" -maxdepth 1 -name '*.safetensors' -printf '%s\n' | awk '{s+=$1} END{print s+0}')

printf '  shards : %s (expect %s)\n' "$N" "$EXPECT_SHARDS"
printf '  bytes  : %s (expect %s)\n' "$B" "$EXPECT_BYTES"

if [ "$N" -ne "$EXPECT_SHARDS" ]; then
    echo "FAIL: wrong shard count, the download is incomplete."
    exit 1
fi
if [ "$B" -ne "$EXPECT_BYTES" ]; then
    echo "FAIL: byte total mismatch (delta $((B - EXPECT_BYTES)))."
    echo "      A partial checkpoint yields wrong output silently. Re-run to resume."
    exit 1
fi

# Per-shard sizes. The total above already proves the download is complete; this proves
# WHICH shard is wrong when it is not, turning "re-download 1.56 TB" into "re-download
# 17 GB". It also catches the one case a total cannot: two shards wrong in opposite
# directions by the same amount.
SIZES="$(dirname "$0")/shard_sizes.txt"
if [ -f "$SIZES" ]; then
    bad=0
    while read -r name want; do
        [ -n "$name" ] || continue
        got=$(stat -c%s "$DEST/$name" 2>/dev/null || echo 0)
        if [ "$got" != "$want" ]; then
            printf '  BAD  %s: %s bytes, expected %s\n' "$name" "$got" "$want"
            bad=$((bad + 1))
        fi
    done < "$SIZES"
    if [ "$bad" -ne 0 ]; then
        echo "FAIL: $bad shard(s) do not match their published sizes."
        echo "      Delete just those files and re-run; the download resumes."
        exit 1
    fi
    printf '  shards : all %s match their published sizes individually\n' "$EXPECT_SHARDS"
fi

echo
echo "verifying file checksums against Hub metadata for revision $REVISION…"
hf cache verify "$REPO" \
    --revision "$REVISION" \
    --local-dir "$DEST" \
    --fail-on-missing-files

echo
echo "  RESULT : checksum-verified snapshot"
echo
echo "next: scripts/pack-trunk.sh $DEST <trunk_dir>"
echo "      packing the trunk is what lets the engine stream it, which is what makes"
echo "      the memory budget a dial instead of a floor."
