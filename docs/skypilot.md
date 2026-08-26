# 🚀 Deployment on Cloud using SkyPilot
This guide explains how to launch and run decentralized model training using SkyPilot on supported cloud providers.
The setup uses a Lighthouse node to coordinate the training and one or more Worker clusters with GPUs for distributed training.
Our tests were conducted on Nebius Cloud, but SkyPilot supports many major providers (AWS, GCP, Azure, OCI, etc.).

## 🧩 Requirements
Before proceeding, make sure you have:

- Installed Python 3.8+
- Installed and configured [SkyPilot](https://docs.skypilot.co/en/latest/getting-started/quickstart.html)
- Access to a supported cloud provider (Nebius, AWS, etc.)
- Proper credentials set up for the cloud provider
- Access to this repository (contains configuration files under panofabric-engine/skypilot/)

## ⚙️ Step 1: Launch the Lighthouse Node
First, launch a Lighthouse node that acts as the central coordinator for TorchFT. A normal CPU VM is sufficient (e.g., 4 vCPUs).

```bash
sky launch -c lighthouse ../panofabric-engine/skypilot/lighthouse.yaml
```

Once launched, note the public IP address of the lighthouse node — this will be used by all worker nodes to connect.
You may also setup tailscale VPN, but that requires manual intervention.

## ⚙️ Step 2: Launch Worker Clusters
Next, start one or more Worker clusters. Each cluster should contain at least one GPU node, depending on your training requirements.

```bash
sky exec worker_cluster \
  --env TORCHFT_LIGHTHOUSE="<ip_lighthouse>" \
  --env SOCKET_IFNAME="network-interface" \
  --env CONFIG_FILE="./models/llama3/train_configs/debug_model.toml" \
  skypilot/worker.yaml
```

### 🧠 Environment Variables Explained
| Variable	|   Description |   Example |
| --------  |   ----------- |   ------- |
| TORCHFT_LIGHTHOUSE    | 	The IP address of the lighthouse node for all workers to communicate with.  |	10.0.5.18   |
| SOCKET_IFNAME         |	The name of the network interface used for inter-node communication. Use the interface that connects to other worker nodes within the cloud.    |	eth0, network-interfa, tailscale0, etc.    |
| CONFIG_FILE	|   Path to the training configuration file (e.g., model, hyperparameters). |	./models/llama3/train_configs/debug_model.toml |
| FT_REPLICA_ID |	Unique ID for the cluster (e.g., 0, 1, 2, …). Each cluster must have a unique ID. | 0 |
| FT_GROUP_SIZE	|   Total number of participating clusters. |   2 |
### ⚙️ Worker Resource Customization
In the worker.yaml file, configure your desired GPU type and cluster size.
For example:

```yaml
resources:
  accelerators: {H100: 8, H200: 8}
  num_nodes: 2
```
Set `--num_nodes` and `--accelerators` accordingly when running the sky exec command if needed.

## 🔁 Step 3: Restart or Manually Run Training
SkyPilot automatically executes the training script once the cluster is ready.
However, you can re-run the training manually if needed:

```bash
sky exec <sky_cluster_name> \
  --env TORCHFT_LIGHTHOUSE="<ip_lighthouse>" \
  --env SOCKET_IFNAME="network-interfa" \
  --env CONFIG_FILE="./models/llama3/train_configs/debug_model.toml" \
  skypilot/worker.yaml
```

## 🌍 Step 4: Multi-Region Setup (Optional)
You can launch additional clusters in different regions for cross-regional decentralized training.

Make sure you set the right value for the environment variables `FT_REPLICA_ID` and `FT_GROUP_SIZE`.

```bash
sky exec worker_cluster_eu \
  --env TORCHFT_LIGHTHOUSE="<ip_lighthouse>" \
  --env SOCKET_IFNAME="eth0" \
  --env CONFIG_FILE="./models/llama3/train_configs/debug_model.toml" \
  --env FT_REPLICA_ID=1 \
  --env FT_GROUP_SIZE=2 \
  skypilot/worker.yaml
```


## ✅ Summary
- Launch the Lighthouse node (CPU-only)
- Launch Worker clusters (GPU instances) using worker.yaml
- Define required environment variables (TORCHFT_LIGHTHOUSE, SOCKET_IFNAME, CONFIG_FILE)
- Customize GPU types and cluster size as needed
- (Optional) Add multi-region clusters with FT_REPLICA_ID and FT_GROUP_SIZE
- With these steps, your decentralized training setup should be ready to run fully across supported clouds using SkyPilot!