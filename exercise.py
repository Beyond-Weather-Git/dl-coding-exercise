"""
Weather Model Training Assignment
=====================================
Time: 20 minutes

Your task is to build a minimal pipeline for training a weather prediction model
using ERA5-like data stored in Zarr format.

Setup Instructions:
conda create --name dl-exercise python==3.12
conda activate dl-exercise
pip install uv
uv pip install torch xarray zarr dask netcdf4 decouple cartopy matplotlib
"""

import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

# ============================================================================
# TASK 1: Data Loading (5 minutes)
# ============================================================================

def load_and_inspect_data(zarr_path: str = "weather_data.zarr") -> xr.Dataset:
    """
    TODO: 
    1. Load the Zarr dataset
    2. Print basic information (dimensions, variables, time range)
    3. Extract a subset for training (e.g., specific region or time range)
    4. Return the processed dataset
    """
    
    # YOUR CODE HERE
    ds = xr.open_zarr(zarr_path, zarr_format=3, consolidated=False)
    
    # - Consider memory / IO efficiency
    # - Think about train/val split
    
    return ds



# ============================================================================
# TASK 2: PyTorch DataLoader (7 minutes)
# ============================================================================

class ReducedGaussianDataset(Dataset):
    """
    TODO: Implement a PyTorch Dataset for weather prediction on reduced Gaussian grid.
    
    Challenge: How do you handle patch extraction on a 1D array that represents
    a sphere with varying resolution?
    
    Options to consider:
    1. Work directly with the 1D representation
    2. Interpolate to a regular grid for training
    3. Use graph neural network approaches
    4. Sample patches in spectral space
    """
    
    def __init__(
        self, 
        data: xr.Dataset, 
        lead_time: int = 1,
        normalize: bool = True,
    ):
        """
        Args:
            data: xarray Dataset with weather variables
            lead_time: number of timesteps to predict ahead
            normalize: whether to normalize the data
            approach: how to handle the reduced Gaussian grid
        """
        self.data = data
        self.lead_time = lead_time
        self.normalize = normalize
        
        # Extract grid structure
        self.points_per_lat = data.attrs.get("points_per_latitude", [])
        self.n_points = data.sizes["points"]
        self.cell_areas = data.cell_area.values
        
        # TODO: Calculate normalization statistics
        # IMPORTANT: Weight by cell area when computing statistics!
        self.mean = {}
        self.std = {}
        
    def __len__(self):
        # TODO: Calculate number of valid samples
        # This depends on your approach (full grid vs patches)
        return self.data.sizes["time"] - self.lead_time
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        TODO: Return (input, target, weights) tuple
        """
        # YOUR CODE HERE

        
        # Dummy return (replace this)
        input_tensor = torch.randn(4, self.n_points)
        target_tensor = torch.randn(4, self.n_points)
        weights = torch.tensor(self.cell_areas)
        
        return input_tensor, target_tensor, weights


# ============================================================================
# TASK 3: Model Setup (8 minutes)
# ============================================================================

class SimpleWeatherModel(nn.Module):
    """
    TODO: Implement a model that can handle reduced Gaussian grid data.
    
    Options:
    1. Fully connected network (treats grid as 1D)
    2. Graph Neural Network (build connectivity based on neighbors)
    3. Spectral methods (spherical harmonics)
    4. Interpolate to regular grid then use CNN
    
    Consider the trade-offs of each approach!
    """
    
    def __init__(
        self, 
        n_points: int,

    ):
        super().__init__()
        self.n_points = n_points
        # Fully connected approach
        # Input: [batch, n_vars, n_points]
        # Output: [batch, n_vars, n_points]
        self.model = nn.Sequential(
            nn.Linear(n_points, 512),
            nn.ReLU(),
            nn.Linear(512, n_points)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE
        # Handle different approaches
        return x


class AreaWeightedMSELoss(nn.Module):
    """
    TODO: Implement area-weighted MSE loss for reduced Gaussian grid.
    Cell areas vary significantly between equator and poles!
    """
    
    def __init__(self, normalize_by_area: bool = True):
        super().__init__()
        self.normalize_by_area = normalize_by_area
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions [B, C, N_points]
            target: Targets [B, C, N_points]
            weights: Area weights [N_points] or [B, N_points]
        """
        # YOUR CODE HERE
        # Remember: proper area weighting is crucial for physical consistency!
        
        mse = (pred - target) ** 2
        
        # Apply area weighting
        if len(weights.shape) == 1:
            weights = weights.unsqueeze(0).unsqueeze(0)  # [1, 1, N_points]
        
        weighted_mse = mse * weights
        
        if self.normalize_by_area:
            return weighted_mse.sum() / weights.sum()
        else:
            return weighted_mse.mean()


def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool = True
) -> float:
    """
    TODO: Implement training step for reduced Gaussian grid model.
    """
    model.train()
    total_loss = 0.0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    for batch_idx, (inputs, targets, weights) in enumerate(dataloader):
        # YOUR CODE HERE
        # Remember to handle the 1D nature of the data!
        pass
    
    return total_loss / len(dataloader)



# ============================================================================
# MAIN EXECUTION (Provided to test your implementation)
# ============================================================================

if __name__ == "__main__":
    
    
    # Task 1: Load data
    print("\n" + "="*50)
    print("TASK 1: Loading and inspecting data")
    print("="*50)
    ds = load_and_inspect_data("weather_data.zarr")

    # Task 2: Create dataset and dataloader
    print("\n" + "="*50)
    print("TASK 2: Creating PyTorch Dataset")
    print("="*50)
    
    # Uncomment when implemented:
    # dataset = ReducedGaussianDataset(ds, lead_time=1)
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    # print(f"Dataset size: {len(dataset)}")
    # print(f"Batch shape: {next(iter(dataloader))[0].shape}")
    
    # Task 3: Setup model and training
    print("\n" + "="*50)
    print("TASK 3: Model setup")
    print("="*50)
    
    # Uncomment when implemented:
    # model = SimpleWeatherModel()
    # loss_fn = create_loss_function()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 
    # print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    # 
    # # Run one training step
    # loss = train_step(model, dataloader, optimizer, loss_fn, device)
    # print(f"Training loss: {loss:.4f}")
    
    print("\n" + "="*50)
    print("Assignment complete! Please walk through your implementation.")