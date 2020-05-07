# Install envs

# 1.  due to torch upgrade. if install with cuda10.1 and pip:
python3 -m pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

#python3 -m pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html
python3 -m pip install cython pyyaml==5.1
python3 -m pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# 2. upgrade to latest detectron2.
python3 -m pip install -U detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html -i https://pypi.douban.com/simple
