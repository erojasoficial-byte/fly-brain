# wsl --unregister Ubuntu-22.04
wsl --install -d Ubuntu-22.04

sudo apt update
sudo apt upgrade -y
sudo apt install software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
gcc --version

# # Source 1: https://developer.nvidia.com/cuda-12-5-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
# # Source 2: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#post-installation-actions
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.5.0/local_installers/cuda-repo-wsl-ubuntu-12-5-local_12.5.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-5-local_12.5.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-5-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-5 # NOTE: we had an older CUDA installed (I believe v11.5), we may need to revert back to it
echo '' >> ~/.bashrc # Make CUDA paths permanent by adding to ~/.bashrc
echo '# CUDA paths' >> ~/.bashrc
echo 'export PATH=${PATH}:/usr/local/cuda-12.5/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-12.5/lib64' >> ~/.bashrc
nvcc --version

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda --version

# cd /mnt/c/Users/YOUR_USERNAME/GitHub/fly-brain
conda env create -f environment.yml
conda activate brian2cuda
python -c "import brian2cuda; brian2cuda.example_run()"
# python fly-brain/code/example-benchmark.py

conda deactivate
conda create --name flybody -c conda-forge -y python=3.10 pip cudatoolkit=11.8.0 ipython ipykernel jupyterlab notebook ffmpeg
conda activate flybody
pip install -e fly-body[ray]
mujoco fly-body/flybody/fruitfly/assets/floor.xml

# Optionally, test training a simple DMPO policy for the flybody MuJoCo environment
# CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib # NOTE: this is temporary, run it at least once when starting a new terminal
# python fly-body/flybody/train_dmpo_ray.py --test