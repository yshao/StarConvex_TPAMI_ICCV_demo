from .pointnet import PointNet
from .pointconv import create_pointconv
from .dgcnn import DGCNN
from .ppfnet import PPFNet
from .pooling import Pooling

from .classifier import Classifier
from .segmentation import Segmentation

from .dcp import DCP
from .prnet import PRNet
from .pcrnet import iPCRNet
from .pointnetlk import PointNetLK
from .rpmnet import RPMNet
from .pcn import PCN
from .deepgmr import DeepGMR

# +
### Additional
# from .prnet import PRNet_mod
# print('success')
# -

try:
	from .flownet3d import FlowNet3D
	print('success model')    
except Exception as e:
	print(e)
	print("Error raised in pointnet2 module for FlowNet3D Network!\nEither don't use pointnet2_utils or retry it's setup.")
