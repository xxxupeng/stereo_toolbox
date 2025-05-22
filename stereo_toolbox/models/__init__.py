import torch

from .PSMNet.stackhourglass import PSMNet
from .GwcNet.gwcnet import GwcNet_G, GwcNet_GC
from .CFNet.cfnet import CFNet
from .PCWNet.pcwnet import PCWNet_G, PCWNet_GC
from .RAFTStereo.raft_stereo import RAFTStereo
from .IGEVStereo.igev_stereo import IGEVStereo
from .MonSter.monster import Monster
from .DEFOMStereo.defom_stereo import DEFOMStereo
from .depth_anything_v2.dpt import DepthAnythingV2
from .STTR.sttr import STTR
from .ACVNet.acv import ACVNet
from .SelectiveStereo.SelectiveIGEV.igev_stereo import IGEVStereo as SelectiveIGEV
from .SelectiveStereo.SelectiveRAFT.raft import RAFT as SelectiveRAFT
from .FoundationStereo.foundation_stereo import FoundationStereo
from .StereoAnywhere import StereoAnywhere


def load_checkpoint_flexible(model, checkpoint_path, state_dict_key=None):
    if state_dict_key is not None:
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)[state_dict_key]
    else:
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
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
    # unmatched_keys = [k for k in new_state_dict if k not in model_dict]
    # if unmatched_keys:
    #     print(f"Warning! Keys below are not macthed: {unmatched_keys}")
        
    model_dict.update(matched_dict)
    missing, unexpected = model.load_state_dict(model_dict)
    if missing:
        print("Missing keys: ", ','.join(missing))
    if unexpected:
        print("Unexpected keys: ", ','.join(unexpected))

    return model