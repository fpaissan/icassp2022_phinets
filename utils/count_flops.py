import sys
sys.path.append("../")

from _modules.VGGish.vggish import VGGish
from torch.autograd import Variable
from torchinfo import summary
import numpy as np
import torch

model = VGGish(device="cpu") #.to("cpu")
model.eval()
rand_x = torch.Tensor(np.random.rand(1, 1, 96, 64)).to("cpu")
# print(model(rand_x).shape)

summary(model, input_data=rand_x, col_names = ["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=1, device="cpu")