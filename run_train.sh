#!/usr/bin/bash

set -ex

# use envs as local overwrites for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_train.sh

NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}

# Keep the repo root off sys.path so the installed packages always win. The forks
# are sibling clones now (not in-tree submodules), so the old shadowing hazard is
# gone, but `panoengine` / `models` must still resolve from the install rather
# than from cwd.
export PYTHONSAFEPATH=${PYTHONSAFEPATH:-1}
MODULE=${MODULE:-"models.llama3"} # module containing config_registry.py
CONFIG_NAME=${CONFIG_NAME:-"llama3_debugmodel"} # function in config_registry.py returning the run config
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.train"} # entry point module passed to torchrun -m; points directly to torchtitan's trainer

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

# Fault-tolerant (DiLoCo / HeLoCo) launch. FT_ENABLE=true turns the torchrun below
# into one replica of an FT group by appending torchtitan's own
# --fault_tolerance.* flags; the rendezvous wiring is already identical, so this
# only contributes flags rather than duplicating the invocation.
#
# This lived in the torchtitan fork's run_train.sh. It moved here so that file can
# stay byte-identical to upstream — nothing in it is fork-specific.
#
# You can also pass the same flags directly (skypilot/worker.yaml does); do one or
# the other, not both, or tyro sees each flag twice.
FT_ENABLE=${FT_ENABLE:-"false"}
FT_FLAGS=()
if [ "${FT_ENABLE,,}" == "true" ]; then
    : "${FT_REPLICA_ID:?FT_REPLICA_ID must be set when FT_ENABLE=true}"
    : "${FT_GROUP_SIZE:?FT_GROUP_SIZE must be set when FT_ENABLE=true}"
    : "${FT_SYNC_STEPS:?FT_SYNC_STEPS must be set when FT_ENABLE=true}"

    FT_RANK_0_SYNC=${FT_RANK_0_SYNC:-"false"}
    FT_FLAGS=(
        --fault_tolerance.enable
        --fault_tolerance.replica_id="${FT_REPLICA_ID}"
        --fault_tolerance.group_size="${FT_GROUP_SIZE}"
        --fault_tolerance.sync_steps="${FT_SYNC_STEPS}"
        --fault_tolerance.num_fragments="${FT_NUM_FRAGMENTS:-1}"
        --fault_tolerance.process_group="${FT_PROCESS_GROUP:-gloo}"
        --fault_tolerance.process_group_timeout_ms="${FT_PROCESS_GROUP_TIMEOUT_MS:-10000}"
    )
    if [ "${FT_RANK_0_SYNC,,}" == "true" ]; then
        FT_FLAGS+=(--fault_tolerance.rank0_synchronization_only)
    fi
fi

MASTER_ADDR=${MASTER_ADDR:-"localhost"} # Set to the public IP of the master node (e.g. tailscale ip)
MASTER_PORT=${MASTER_PORT:-"29500"}
LOCAL_ADDR=${LOCAL_ADDR:-"localhost"} # Set to the public IP of the local node (e.g. tailscale ip)


NNODES=${NNODES:-"1"} # Total number of nodes within an island
ISHOST=${ISHOST:-"true"} # Set to true for the master node, false for other nodes in the same island

export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-"tailscale0"} # Hint Gloo to use desired network interface, in this case tailscale
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-"tailscale0"} # Hint NCCL to use desired network interface, in this case tailscale


PYTORCH_ALLOC_CONF="expandable_segments:True" \
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
LOCAL_ADDR=${LOCAL_ADDR} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
NNODES=${NNODES} \
ISHOST=${ISHOST} \
torchrun --nproc_per_node=${NGPU} --nnodes ${NNODES} --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
--local_addr=${LOCAL_ADDR} --rdzv-conf is_host=${ISHOST} --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
-m ${TRAIN_FILE} --module ${MODULE} --config ${CONFIG_NAME} "${FT_FLAGS[@]}" "$@"
