import torch
from .PSMNet.stackhourglass import PSMNet


def load_checkpoint_flexible(model, checkpoint_path, state_dict_key=None):
    if state_dict_key is not None:
        state_dict = torch.load(checkpoint_path, map_location='cpu')[state_dict_key]
    else:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.') and not hasattr(model, 'module'):
            name = k[7:]
        else:
            name = k
            
        if not k.startswith('module.') and hasattr(model, 'module'):
            name = f'module.{k}'
            
        new_state_dict[name] = v
    
    model_dict = model.state_dict()
    matched_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
    unmatched_keys = [k for k in new_state_dict if k not in model_dict]
    
    if unmatched_keys:
        print(f"Warning! Keys below are not macthed: {unmatched_keys}")
        
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict)
    return model