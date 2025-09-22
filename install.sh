conda create -n graspnet python==3.12

conda activate graspnet

pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

pip install setuptools==78.0.1

# libfranka 0.15
pip install franky-control

pip install pyrealsense2

# install graspnet
cd pointnet2
python setup.py install
cd ..

cd knn
python setup.py install
cd ..

cd graspnetAPI
pip install .

cd ..

# install foundation stereo
pip install -r requirements_fs.txt