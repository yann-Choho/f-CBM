import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

class KANLinear(nn.Module):
    """
    Efficient KAN layer based on B-spline basis functions.
    Instead of w*x, applies learned univariate functions φᵢ(x) on each edge.
    More efficient than original implementation via basis function reformulation.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        grid_size: Number of grid intervals for B-spline basis (default 5)
        spline_order: Polynomial order of B-spline basis (default 3, cubic)
        dropout_rate: Dropout applied after activation (default 0.0)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 1,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.dropout = nn.Dropout(dropout_rate)
        
        # Number of basis functions = grid_size + spline_order
        self.num_basis = grid_size + spline_order
        
        # Initialize grid points [-1, 1] for normalization
        self.register_buffer(
            "grid",
            torch.linspace(-1, 1, grid_size + 1)[:-1].unsqueeze(0),
        )
        
        # Learnable coefficients for each basis function
        # Shape: (in_features, out_features, num_basis)
        self.coef = nn.Parameter(
            torch.randn(in_features, out_features, self.num_basis)
        )
        
        # Optional: learnable scales for each output neuron
        self.scale = nn.Parameter(torch.ones(out_features))
        
        # L1 regularization for sparsity (helps interpretability)
        #self.l1_regularizer = 0.0
        
        nn.init.kaiming_uniform_(self.coef, a=np.sqrt(5))
    
    def compute_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis functions for input x.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
        
        Returns:
            Basis functions of shape (batch_size, in_features, num_basis)
        """
        # Normalize input to [-1, 1]
        x = x.unsqueeze(-1)  # (batch, in_features, 1)
        
        # Compute basis functions (simplified B-spline)
        # This is a simplified version - actual B-spline computation can be more complex
        distances = torch.abs(x - self.grid)  # (batch, in_features, grid_size)
        
        # Create cubic B-spline-like basis
        basis = torch.zeros(
            x.shape[0],
            x.shape[1],
            self.num_basis,
            device=x.device,
            dtype=x.dtype
        )
        
        # Simple triangular basis functions
        for i in range(self.num_basis):
            if i == 0:
                basis[..., i] = (1 - distances[..., 0]).clamp(min=0)
            elif i < self.grid_size:
                basis[..., i] = (1 - distances[..., i]).clamp(min=0)
            else:
                # Extrapolation basis
                basis[..., i] = torch.ones_like(basis[..., 0])
        
        return basis  # (batch, in_features, num_basis)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through KAN layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
        
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Compute basis functions
        basis = self.compute_basis(x)  # (batch, in_features, num_basis)
        
        # Apply basis functions with learned coefficients
        # basis: (batch, in_features, num_basis)
        # coef: (in_features, out_features, num_basis)
        # output: (batch, out_features)
        output = torch.einsum("bim,iom->bo", basis, self.coef)
        
        # Apply learnable scales
        output = output * self.scale.unsqueeze(0)
        
        # Optional dropout for regularization
        output = self.dropout(output)
        
        return output
    
    # def get_sparsity_loss(self) -> torch.Tensor:
    #     """
    #     L1 regularization on coefficients to encourage sparsity.
    #     Helps make the model more interpretable.
    #     """
    #     return torch.mean(torch.abs(self.coef))

    @torch.no_grad()
    def forward_single_feature(
        self,
        x: torch.Tensor,
        feature_idx: int,
        apply_dropout: bool = False,
        apply_scale: bool = True,
    ) -> torch.Tensor:
        """
        Computes ONLY the contribution from a single input feature to all outputs.
        Useful for response curve visualization.

        Args:
            x: Tensor of shape (batch_size, in_features). Only x[:, feature_idx] is used.
            feature_idx: Index of the input feature to evaluate (0 <= feature_idx < in_features).
            apply_dropout: If True, applies layer's dropout to the result; default False for stable curves.
            apply_scale: If True, applies the learnable 'scale' parameters to match forward(); default True.

        Returns:
            Tensor of shape (batch_size, out_features) with the contribution from the selected feature.
        """
        if not (0 <= feature_idx < self.in_features):
            raise IndexError(f"feature_idx {feature_idx} out of range [0, {self.in_features-1}]")

        # Compute basis for all features, then slice the selected one
        basis_all = self.compute_basis(x)  # (b, in_features, num_basis)
        basis_i = basis_all[:, feature_idx, :]  # (b, num_basis)

        # Coefficients for that feature: (out_features, num_basis)
        coef_i = self.coef[feature_idx, :, :]  # (o, m)

        # Contribution from that feature: (b, o)
        contrib = torch.einsum("bm,om->bo", basis_i, coef_i)

        if apply_scale:
            contrib = contrib * self.scale.unsqueeze(0)

        if apply_dropout:
            contrib = self.dropout(contrib)

        return contrib

    @torch.no_grad()
    def response_curve(
        self,
        feature_idx: int,
        x_values: torch.Tensor,
        device: torch.device = None,
        dtype: torch.dtype = None,
        apply_scale: bool = True,
    ) -> torch.Tensor:
        """
        Convenience helper to evaluate the response of a single input feature across
        a 1D grid of values, holding other features 'unused' (they don't matter
        since we only take the single-feature contribution).

        Args:
            feature_idx: Index of the input feature to evaluate.
            x_values: 1D tensor of shape (n_points,) containing values to probe for that feature.
                      Assumed in the same scale as the layer expects (e.g., roughly [-1, 1]).
            device: Optional device to place the temporary batch on.
            dtype: Optional dtype for computation.
            apply_scale: If True, applies output scaling to match layer behavior.

        Returns:
            Tensor of shape (n_points, out_features).
        """
        if device is None:
            device = self.coef.device
        if dtype is None:
            dtype = self.coef.dtype

        n = x_values.shape[0]
        # Build a dummy input where only the selected feature is populated
        x = torch.zeros(n, self.in_features, device=device, dtype=dtype)
        x[:, feature_idx] = x_values.to(device=device, dtype=dtype)

        # Use forward_single_feature (no dropout for smooth curves)
        return self.forward_single_feature(
            x, feature_idx=feature_idx, apply_dropout=False, apply_scale=apply_scale
        )

    
class SimpleKAN(nn.Module):
    """Simple KAN network with one hidden layer"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        if(self.hidden_dim>0):
            self.layer1 = KANLinear(input_dim, hidden_dim)
            self.layer2 = KANLinear(hidden_dim, output_dim)
        else:
            self.layer1 = KANLinear(input_dim, output_dim)
    
    def forward(self, x):
        if(self.hidden_dim>0):
            x = self.layer1(x)
            x = self.layer2(x)
        else:
            x = self.layer1(x)
        return x
    
def visualize_learned_functions(model, X_train, feature_idx=0, n_points=200):
    """
    Visualize the TOTAL effect of a single input feature on the output.
    Varies one feature over its actual data range while keeping others at their mean.
    """
    model.eval()
    
    # Use actual min/max range from training data for this feature
    feature_min, feature_max = X_train[:, feature_idx].min().item(), X_train[:, feature_idx].max().item()
    x_range = torch.linspace(feature_min, feature_max, n_points)
    
    # Compute means from training data
    means = X_train.mean(dim=0)
    
    # Create inputs where we vary only the feature of interest
    inputs = means.repeat(n_points, 1)
    inputs[:, feature_idx] = x_range
    
    # Get the full model output for these inputs
    with torch.no_grad():
        outputs = model(inputs)
    
    return x_range.numpy(), outputs.squeeze().numpy()

import torch
import numpy as np
import matplotlib.pyplot as plt


import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_concept_to_class_response_curves(
    model,
    train_loader: torch.utils.data.DataLoader,
    feature_names: list,
    output_names: dict,
    n_points: int = 200,
    device: str = "cpu",
    ):
    """
    Plots a grid showing the response curve of each input feature to every output neuron.

    Args:
        model: Trained KANLinear model
        train_loader: DataLoader yielding batches that are dicts; each feature_name
                      corresponds to a key in the batch, batch[feature_name] is (batch_size,)
        feature_names: List of feature names, order defines feature indices for the model
        output_names: Optional dict mapping output index -> name
        n_points: Number of points to evaluate each response curve
        device: Device where model lives ("cpu" / "cuda")
    """

    model.eval()
    model.to(device)

    in_features = model.in_features
    in_features_list = list(range(in_features))
    out_features = model.out_features
    out_features_list = list(range(out_features))

    if feature_names is None:
        raise ValueError("feature_names must be provided when using dict-style batches.")

    if len(feature_names) != in_features:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) must match in_features ({in_features})"
        )

    # ------------------------------------------------------------------
    # 1. Compute per-feature min / max over the whole train_loader
    #    using batch[feature_name]
    # ------------------------------------------------------------------
    feature_mins = {fname: None for fname in feature_names}
    feature_maxs = {fname: None for fname in feature_names}

    with torch.no_grad():
        for batch in train_loader:
            # batch is expected to be a dict-like object
            for fname in feature_names:
                # batch[fname]: shape (batch_size,) or (batch_size, 1)
                vals = batch[fname]
                if isinstance(vals, torch.Tensor):
                    vals = vals.to(device).view(-1)
                else:
                    # if not tensor, convert
                    vals = torch.as_tensor(vals, device=device).view(-1)

                bmin = vals.min()
                bmax = vals.max()

                if feature_mins[fname] is None:
                    feature_mins[fname] = bmin
                    feature_maxs[fname] = bmax
                else:
                    feature_mins[fname] = torch.minimum(feature_mins[fname], bmin)
                    feature_maxs[fname] = torch.maximum(feature_maxs[fname], bmax)

    # Check that we saw at least one batch
    if any(v is None for v in feature_mins.values()):
        raise ValueError("train_loader appears to be empty or missing some feature keys.")

    # ------------------------------------------------------------------
    # 2. Possibly subsample outputs / inputs (as in your original code)
    # ------------------------------------------------------------------
    if len(out_features_list) > 4:
        out_features_selection = np.random.choice(out_features_list, 4, replace=False)
    else:
        out_features_selection = out_features_list

    if len(in_features_list) > 6:
        in_features_selection = np.random.choice(in_features_list, 6, replace=False)
    else:
        in_features_selection = in_features_list

    # ------------------------------------------------------------------
    # 3. Plot - Adjusted figsize to accommodate legend better
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        len(out_features_selection),
        figsize=(20, 5),  # Reduced width and height for more compact layout
        sharex=True,
        squeeze=False,
    )

    # Store handles and labels for the shared legend
    handles, labels = None, None

    for out_idx_graph, out_idx in enumerate(out_features_selection):
        ax = axes[0, out_idx_graph]

        for _, feature_idx in enumerate(in_features_selection):
            feature_name = feature_names[feature_idx]

            feature_min = feature_mins[feature_name].item()
            feature_max = feature_maxs[feature_name].item()

            x_range = torch.linspace(feature_min, feature_max, n_points, device=device)

            # Response curve: (n_points, out_features)
            with torch.no_grad():
                resp = model.response_curve(
                    feature_idx=feature_idx,
                    x_values=x_range,
                    apply_scale=True,
                )

            y_curve = resp[:, out_idx].detach().cpu().numpy()
            ax.plot(x_range.cpu().numpy(), y_curve, label=feature_name)

        ax.set_title(output_names.get(out_idx, f"Output {out_idx}"), fontsize=12, pad=10)
        ax.set_ylabel("Response", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Get handles and labels from the first subplot
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    axes[-1, 0].set_xlabel("Feature Values", fontsize=10)
    
    # Create a single legend with better positioning
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.92, 0.5), 
               fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.90)  # Less space on right - legend closer to plots
    plt.show()

