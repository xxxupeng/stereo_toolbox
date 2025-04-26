import torch

def find_unused_parameters(model):
    """手动找出未在前向传播中使用的参数"""
    used_params = set()
    
    # 注册前向钩子
    hooks = []
    for name, module in model.named_modules():
        def hook_fn(module, input, output, name=name):
            for param_name, param in module.named_parameters(recurse=False):
                used_params.add(f"{name}.{param_name}" if name else param_name)
        hooks.append(module.register_forward_hook(hook_fn))
    
    # 执行一次前向传播 (使用示例输入)
    # 这里需要根据你的模型创建适当的输入
    sample_input = (
        torch.randn(1, 3, 384, 512).cuda(),  # 左图
        torch.randn(1, 3, 384, 512).cuda()   # 右图
    )
    model(*sample_input)
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 找出所有参数以及未使用的参数
    all_params = set()
    for name, _ in model.named_parameters():
        all_params.add(name)
    
    unused_params = all_params - used_params
    return unused_params 