
import torch

def displacement_gradients(displacements, coords):
    """
    Calculates the gradients of displacement vectors with respect to coordinates.
    The third coordinate t is not used in the gradient calculation.
    displacements and coords must be linked in the computation graph, so torch autograd can compute gradients.
    Args:
        displacements: Tensor of shape [B, 2] containing displacement vectors (u_x, u_y)
        coords: Tensor of shape [B, 3] containing input coordinates (x, y, t)
    Returns:
        strain_tensor: Tensor of shape [B, 2, 2] containing the green-lagrange strain tensors
        deformation_gradient: Tensor of shape [B, 2, 2] containing the deformation gradient tensors
        displacement_gradient: Tensor of shape [B, 2, 2] containing the displacement gradient tensors
    """

    if not coords.requires_grad:
        coords.requires_grad_(True)
    
    batch_size = coords.shape[0]
    displacement_gradient = torch.zeros(batch_size, 2, 2, device=coords.device)
    
    for displacement_component in range(2):
        selector = torch.zeros_like(displacements)
        selector[:, displacement_component] = 1.0
        
        # Compute ∂u_i/∂(x,y,t) 
        grad_u_i = torch.autograd.grad(
            outputs=displacements,
            inputs=coords,
            grad_outputs=selector,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Store spatial gradients: ∂u_i/∂x and ∂u_i/∂y
        displacement_gradient[:, displacement_component] = grad_u_i[:, :2]
    
    identity = torch.eye(2, device=coords.device).unsqueeze(0).expand(batch_size, -1, -1)
    deformation_gradient = identity + displacement_gradient  # F = I + ∇u
    strain = 0.5 * (deformation_gradient.transpose(1, 2) @ deformation_gradient - identity)  # E = 0.5 * (F^T F - I)
    
    return strain, deformation_gradient, displacement_gradient