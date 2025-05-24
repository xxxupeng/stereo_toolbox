import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
torch.backends.cudnn.benchmark = True


def speed_and_memory_test(model, resolution=None, batch_size=1, num_iterations=100, device='cuda:0'):
    """
    Test the speed and memory usage of a model.

    Parameters:
        model (torch.nn.Module): The model to test.
        resolution (tuple): The input resolution (height, width).
        batch_size (int): The batch size.
        num_iterations (int): The number of iterations to run.

    Returns:
        list: The resolutions tested.
        list: The average time per iteration in seconds for each resolution.
        list: The average memory usage in MB for each resolution.
    """

    model = model.to(device).eval()

    # Count the total number of parameters in the model (including both learnable and non-learnable)
    total_params = sum(p.numel() for p in model.parameters())
    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of parameters: {total_params / 1e6:.2f}M')
    print(f'Learnable parameters: {learnable_params / 1e6:.2f}M')

    resolutions = [(480, 640), (736,1280), (1088,1920)]
    if resolution is not None:
        resolutions.append(resolution)

    avg_times = []
    avg_memories = []

    for resolution in resolutions:
        # Create a random input tensor
        input_tensor = torch.randn(batch_size, 3, *resolution).to(device)
        input_tensor = [input_tensor, input_tensor]

        # Warm up the GPU
        for _ in range(20):
            with torch.no_grad():
                model(*input_tensor)

        # Measure time and memory usage
        times = []
        mems = []

        for _ in tqdm(range(num_iterations)):
            torch.cuda.reset_peak_memory_stats(device)
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            start_time.record()
            with torch.no_grad():
                model(*input_tensor)
            end_time.record()

            torch.cuda.synchronize()

            elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB

            times.append(elapsed_time)
            mems.append(peak_memory)

        avg_times.append(np.mean(times))
        avg_memories.append(np.mean(mems))

        print(f'Resolution: {resolution}, Avg Time: {np.mean(times):.4f} s, Avg Frequency: {1/np.mean(times):.4f} Hz,Avg Memory: {np.mean(mems):.2f} MB')

    return resolutions, avg_times, avg_memories
