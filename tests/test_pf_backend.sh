#!/bin/sh
# Regression test for scripts/pf_backend.sh (POSIX sh, no framework, no GPU).
#
# Stubs `nvidia-smi` on PATH to replay REAL outputs from lcluster13 and asserts the
# backend token. The bug this pins: r610 renamed the header fields, the old header
# scrape came back empty and the script answered `cpu` for a healthy H100 -- so the
# engine built a CPU-only torch and cached it under a `cpu` fingerprint.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
script="$here/../scripts/pf_backend.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export PATH="$tmp:$PATH"
export PF_CUDA_COMPAT_DIR=""            # no compat dir: exercises the warning branch too
fails=0

stub_nvidia_smi() {  # $1 = header line, $2 = driver_version for --query-gpu
	cat > "$tmp/nvidia-smi" <<STUB
#!/bin/sh
case "\$*" in
*--query-gpu=driver_version*) printf '%s\n' '$2' ;;
*) printf '%s\n%s\n' 'Fri Aug 28 21:51:23 2026' '$1' ;;
esac
STUB
	chmod +x "$tmp/nvidia-smi"
}

check() {  # $1 = label, $2 = expected token
	got=$(sh "$script" 2>/dev/null)
	if [ "$got" = "$2" ]; then
		echo "ok   $1 -> $got"
	else
		echo "FAIL $1 -> got '$got', want '$2'"
		fails=$((fails + 1))
	fi
}

# r580 (V100 nodes): the old header format.
stub_nvidia_smi '| NVIDIA-SMI 580.173.02  Driver Version: 580.173.02  CUDA Version: 13.0 |' '580.173.02'
check "r580 old header" cu130

# r610 (H100 nodes): renamed fields -- THE REGRESSION. No "CUDA Version:" anywhere.
stub_nvidia_smi '| NVIDIA-SMI 610.43.02  KMD Version: 610.43.02  CUDA UMD Version: 13.3 |' '610.43.02'
check "r610 renamed header" cu130

# A future driver whose header we cannot parse AND whose query fails: still CUDA.
# Downgrading a working GPU to `cpu` is never the right answer.
stub_nvidia_smi '| totally unknown future format |' ''
check "unparseable driver" cu130

# Pre-r580 driver: still cu130, but must warn about forward-compat.
stub_nvidia_smi '| NVIDIA-SMI 550.54.15  Driver Version: 550.54.15  CUDA Version: 12.4 |' '550.54.15'
check "pre-r580 driver" cu130
if sh "$script" 2>&1 >/dev/null | grep -q "predates r580"; then
	echo "ok   pre-r580 warns"
else
	echo "FAIL pre-r580 did not warn"
	fails=$((fails + 1))
fi

# No nvidia-smi at all -> cpu is correct. PATH is narrowed to the (now empty) stub dir
# rather than just deleting the stub: a dev box with real NVIDIA tools installed would
# otherwise find the system one and this case would silently test nothing.
rm -f "$tmp/nvidia-smi"
got=$(PATH="$tmp" /bin/sh "$script" 2>/dev/null)   # absolute: $tmp has no `sh`
if [ "$got" = "cpu" ]; then
	echo "ok   no nvidia-smi -> cpu"
else
	echo "FAIL no nvidia-smi -> got '$got', want 'cpu'"
	fails=$((fails + 1))
fi

[ "$fails" -eq 0 ] || exit 1
echo "all pf_backend checks passed"
