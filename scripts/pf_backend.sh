#!/bin/sh
# Print THIS node's torch backend token: cu130 | rocm7.0 | cpu.
# `sh scripts/pf_backend.sh compat-dir` instead prints the CUDA forward-compat lib dir
# when one is resolvable (else nothing) — a diagnostic for setting up old-driver clusters.
#
# Single source of truth shared by `make install-torch` (which picks the PyTorch index
# URL from it) and the env-cache fingerprint (scripts/pf_env_fp.sh / the Makefile `ensure`
# target), so the cache KEY and the actual torch INSTALL can never disagree.
#
# WHY EVERY NVIDIA NODE GETS cu130: torchtitan tracks PyTorch main, published only for
# CUDA 13 (cu128 wheels froze at torch 2.12 nightly, whose fused kernels are broken with
# current torchtitan — tested 2026-07-28, unfixable without forking the engine). CUDA 13
# needs an r580+ driver OR NVIDIA's forward-compat userspace driver (datacenter GPUs):
# put its lib dir on LD_LIBRARY_PATH before torch starts. On PanoFabric that is the
# cluster's "Worker environment" setting (compute page → Slurm settings → worker_envs);
# baking venv symlinks does NOT work — the loader resolves the system libcuda first.
#
# POSIX sh only (no bashisms): it runs on whatever /bin/sh the compute node ships, and is
# invoked as `sh scripts/pf_backend.sh` so a dropped exec bit (server-mode zip upload) is
# harmless.
compat_dir="${PF_CUDA_COMPAT_DIR:-}"
if [ -z "$compat_dir" ]; then
	IFS=:
	for d in ${LD_LIBRARY_PATH:-}; do
		if [ -e "$d/libcuda.so.1" ]; then compat_dir="$d"; break; fi
	done
	unset IFS
fi
if [ -z "$compat_dir" ]; then
	compat_base="$HOME"
	case "$HOME" in */.sky_clusters/*) compat_base="${HOME%/.sky_clusters/*}" ;; esac
	compat_dir="$compat_base/cuda-compat"
fi
[ -e "$compat_dir/libcuda.so.1" ] || compat_dir=""

if [ "${1:-}" = "compat-dir" ]; then
	printf '%s' "$compat_dir"
	exit 0
fi

if command -v rocminfo >/dev/null 2>&1 || [ -x /opt/rocm/bin/rocminfo ]; then
	printf 'rocm7.0'
elif command -v nvidia-smi >/dev/null 2>&1; then
	# A working nvidia-smi means CUDA, FULL STOP: nothing below may downgrade that to
	# `cpu`. It used to, and it cost a live debug (lcluster13, 2026-08-28). The driver
	# version came from scraping nvidia-smi's human-readable header, which is not a
	# stable interface -- r610 renamed its fields:
	#   r580: | NVIDIA-SMI 580.173.02  Driver Version: 580.173.02  CUDA Version: 13.0 |
	#   r610: | NVIDIA-SMI 610.43.02   KMD Version: 610.43.02  CUDA UMD Version: 13.3 |
	# so `grep "CUDA Version"` matched NOTHING on r610, the old code fell through to
	# `cpu`, and the engine built a CPU-only torch on an H100. Worse, `ensure` keys its
	# cache on this token, so the junk env was published under a `cpu` fingerprint and
	# the training run died with no hint that a STRING PARSE was the cause. The cluster
	# was mixed (V100s on r580, H100s on r610), so the prewarm looked healthy and only
	# runs landing on the newer nodes broke.
	#
	# So: query the driver version through the machine-readable interface, and use it
	# ONLY to decide whether to warn about forward-compat.
	driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null |
		head -1 | tr -d ' \r')
	major=${driver%%.*}
	case "$major" in
	'' | *[!0-9]*) : ;; # unparseable/unavailable -> no claim about the driver
	*)
		# CUDA 13 wheels need an r580+ driver, or NVIDIA's forward-compat userspace.
		if [ "$major" -lt 580 ] && [ -z "$compat_dir" ]; then
			echo "[pf_backend] WARNING: NVIDIA driver $driver predates r580, but the engine" \
				"needs torch>=2.13 (CUDA 13 only). Update the driver to r580+, or install" \
				"NVIDIA's cuda-compat libs and put their dir on LD_LIBRARY_PATH" \
				"(on PanoFabric: compute page -> Slurm settings -> Worker environment)." \
				"Without one of these, torch will fail to initialize CUDA." >&2
		fi
		;;
	esac
	printf 'cu130'
else
	printf 'cpu'
fi
