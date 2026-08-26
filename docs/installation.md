# Installation guideline
In case running the Makefile did not work for your setup, you may run the following steps to install the framework.
We assume that you have a debian-based linux OS, preferably Ubuntu 24+.

## 1. Setting up the environment
Here is the process of setting up the environment, including installing core utilities, dependencies, and important tools such as Protocol Buffers, Tailscale, Rust, and uv.

1. Make sure you have latest necessary packages.
```bash
    sudo apt-get update -y
	sudo apt-get install -y unzip curl ca-certificates gnupg wget build-essential pkg-config libssl-dev
```

2. Installing protoc > v30.0 can be either done via `apt install protobuf-compiler` if you have the latest update or manually: 
```bash
    wget https://github.com/protocolbuffers/protobuf/releases/download/v32.0/protoc-32.0-linux-x86_64.zip
    unzip protoc-32.0-linux-x86_64.zip -d $HOME/.local
    echo 'export PATH=$PATH:$HOME/.local/bin' >> ~/.bashrc
    source ~/.bashrc
```

3. Installing tailscale for decentralized training in different networks. This is mandatory for connecting different HPC/cloud resources running on different networks with no public routable IP address.
``` bash
    curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.noarmor.gpg | sudo tee /usr/share/keyrings/tailscale-archive-keyring.gpg >/dev/null
	curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.tailscale-keyring.list | sudo tee /etc/apt/sources.list.d/tailscale.list >/dev/null
	sudo apt-get update -y
	sudo apt-get install -y tailscale
```

In case you do not have root permission to install tailscale, please follow the instruction [here](https://blog.papermatch.me/html/How_to_run_Tailscale_without_root).

4. Installing UV package manager:
```bash
    curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh -s -- -y
	curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Installing the framework
1. First clone the repository with:
```
git clone --recursive https://github.com/PanocularAI/panofabric-engine.git
```
Make sure that you pull all submodules using `--recursive` flag.

2. Change directory into the cloned repository and create uv venv using:
```bash
    cd panofabric-engine
    uv sync
```

3. Installing PyTorch:
You need to install the correct version of PyTorch nightly according to your hardware. On Nvidia GPUs the engine requires the **CUDA 13** wheels:
```bash
    uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130 --force-reinstall
```

The older `cu128`/`cu129` indexes will NOT work: torchtitan calls
`create_block_mask(separate_full_blocks=...)`, added in torch 2.13, and PyTorch stopped
publishing CUDA 12 wheels before that (cu128 caps at 2.11 stable / 2.12.0.dev20260408).
CUDA 13 wheels need an r580+ driver. On an older driver (datacenter GPUs only), install
NVIDIA's forward-compat package (`conda install -c nvidia cuda-compat`) and put its lib
dir on `LD_LIBRARY_PATH` before starting training — on PanoFabric, set it once per cluster
under compute page → Slurm settings → Worker environment. For AMD GPUs use the rocm
backend instead:
```bash
    uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm7.0 --force-reinstall
```

4. Installing torchtitan and torchft dependencies:
```bash
    uv pip install -r torchtitan/requirements.txt
    uv pip install ./torchtitan
    uv pip install ./torchft
```

Once you are done with all these steps, you should have all the required dependencies set up.

<a id="tailscale-setup"></a>
## 3. (Optional) Configuring Tailscale VPN
To establish communication between different compute islands, each master node within an island must have a routable public IP address.
If public IPs are not available, it is recommended to use Tailscale, which was installed in the previous steps.

To configure Tailscale, execute the following command:
```bash
    sudo tailscale up
```

A link will be shown and upon clicking on the link and potential registration, you should be able to sign in and get a unique ip for your node via:
```bash
    tailscale ip
```
