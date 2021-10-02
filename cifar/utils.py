import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

count_ops = 0
num_ids = 0
def get_feature_hook(self, _input, _output):
	global count_ops, num_ids 
	# print('------>>>>>>')
	# print('{}th node, input shape: {}, output shape: {}, input channel: {}, output channel {}'.format(
	# 	num_ids, _input[0].size(2), _output.size(2), _input[0].size(1), _output.size(1)))
	# print(self)
	delta_ops = self.in_channels * self.out_channels * self.kernel_size[0] * self.kernel_size[1] * _output.size(2) * _output.size(3) / self.groups
	count_ops += delta_ops
	# print('ops is {:.6f}M'.format(delta_ops / 1024.  /1024.))
	num_ids += 1
	# print('')

def measure_model(net, H_in, W_in):
	import torch
	import torch.nn as nn
	_input = torch.randn((1, 3, H_in, W_in))
	#_input, net = _input.cpu(), net.cpu()
	hooks = []
	for module in net.named_modules():
		if isinstance(module[1], nn.Conv2d) or isinstance(module[1], nn.ConvTranspose2d):
			# print(module)
			hooks.append(module[1].register_forward_hook(get_feature_hook))

	_out = net(_input)
	global count_ops
	print('count_ops: {:.6f}M'.format(count_ops / 1024. /1024.)) # in Million






def get_pruned_feature_hook(self, _input, _output):
	global count_ops, num_ids
	print('------>>>>>>')
	print('{}th node, input shape: {}, output shape: {}, input channel: {}, output channel {}'.format(
		num_ids, _input[0].size(2), _output.size(2), _input[0].size(1), _output.size(1)))
	print("self:", self)
	# delta_ops = self.in_channels * self.out_channels * self.kernel_size[0] * self.kernel_size[1] * _output.size(2) * _output.size(3) / self.groups
	weight = self.weight
	print("weight gpu:", weight.size())
	weight = weight.data.cpu().numpy()
	print("weight cpu:", weight.shape)
	non_zero_num = np.sum(weight!=0)
	print("none zero ratio: %d/%d" % (non_zero_num, weight.size))
	delta_ops = non_zero_num * _output.size(2) * _output.size(3)
	count_ops += delta_ops
	print('ops is {:.6f}M'.format(delta_ops / 1024.  /1024.))
	num_ids += 1
	print('')

def measure_pruned_model(net, H_in, W_in):
	'''
	Args:
	   net: pytorch network, father class is nn.Module
	   H_in: int, input image height
	   W_in: int, input image weight
	'''
	_input = Variable(torch.randn((1, 3, H_in, W_in)))
	#_input, net = _input.cpu(), net.cpu()
	hooks = []
	for module in net.named_modules():
		if isinstance(module[1], nn.Conv2d) or isinstance(module[1], nn.ConvTranspose2d) :
			print(module)
			hooks.append(module[1].register_forward_hook(get_pruned_feature_hook))
	_out = net(_input)
	global count_ops
	print('count_ops: {:.6f}M'.format(count_ops / 1024. /1024.)) # in Million (edited) 



if __name__ == '__main__':
	import torchvision.models as models
	model = models.resnet34(pretrained=True)
	
	# measure_pruned_model(model, 224, 224)
	measure_model(model, 224, 224)
