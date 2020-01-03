import os,sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.insert(0, '..')

from model import unet
from model import attention_unet