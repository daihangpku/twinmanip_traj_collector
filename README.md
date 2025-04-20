# twinmanip_traj_collector

## Installation
```bash
git submodule update --init --recursive
conda create -n traj_collector python=3.10 -y
conda activate traj_collector
# choosing your cuda version, this is for cuda 11.8
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```
### Genesis
```bash
pip install genesis-world
```
### Curobo
```bash
sudo apt install git-lfs
cd motion_planner/curobo
pip install -e . --no-build-isolation
cd ../..
```
### Recorder
```bash
cd twinmanip_realsense_recorder
pip install -r requirements
cd ..
```

## Installation(Using Frankapy)

```bash
conda activate frankapy
source ~/workspace/frankapy/catkin_ws/devel/setup.bash
```

### Curobo
```bash
sudo apt install git-lfs
cd motion_planner/curobo
pip install -e . --no-build-isolation
cd ../..
```
### Recorder
```bash
cd twinmanip_realsense_recorder
pip install -r requirements.txt
cd ..
```