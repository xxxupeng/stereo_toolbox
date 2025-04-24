import torch

from .PSMNet.stackhourglass import PSMNet # change `.cuda()` to `.to(x.device)` and optimize the cost volume building
from .GwcNet.gwcnet import GwcNet_G, GwcNet_GC
from .CFNet.cfnet import CFNet # mish avtivation function only, return pred1_s2 only when evaluation
from .PCWNet.pcwnet import PCWNet_G, PCWNet_GC # rename class as PCWNet, mish avtivation function only, return disp_finetune only when evaluation
from .RAFTStereo.raft_stereo import RAFTStereo # init self.args, negate all outputs as disparity is positive when traversing to the left by default.
from .IGEVStereo.igev_stereo import IGEVStereo # init self.args, add imagenet_norm para. (true for imagenet's mean and std, false for all 0.5 to rescale to [-1,1])
from .MonSter.monster import Monster
from .DEFOMStereo.defom_stereo import DEFOMStereo # init self.args, rescale the inputs (1/255 of the orginal), note that the used depthanythingv2 has additional interpolation step
from .depth_anything_v2.dpt import DepthAnythingV2

def load_checkpoint_flexible(model, checkpoint_path, state_dict_key=None):
    if state_dict_key is not None:
        state_dict = torch.load(checkpoint_path, map_location='cpu')[state_dict_key]
    else:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    new_state_dict = {}
    model_dict = model.state_dict()

    for k, v in state_dict.items():
        if k in model_dict:
            name = k
        elif k.startswith('module.') and k not in model_dict:
            name = k[7:]
        elif not k.startswith('module.') and k not in model_dict:
            name = f'module.{k}'    

        new_state_dict[name] = v
    
    matched_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
    unmatched_keys = [k for k in new_state_dict if k not in model_dict]
    if unmatched_keys:
        print(f"Warning! Keys below are not macthed: {unmatched_keys}")
        
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict)
    return model