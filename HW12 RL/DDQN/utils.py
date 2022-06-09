# 一些接口工具,方法,硬替换,直接隔多少步把eval_net赋给targ_net
import numpy as np
import torch

def hard_update(target, source):
	# 更新target net
	for target_param, source_param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(source_param.data)

def soft_update(target, source, tau):
	for target_param, source_param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(tau*source_param.data + (1-tau)*target_param)


