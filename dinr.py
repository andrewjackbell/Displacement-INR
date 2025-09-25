from images_points_dataset import IPDataset
import numpy as np
import torch
from torch.nn import MSELoss
from networks.mod_siren import ModulatedSiren
from utils.autograd import displacement_gradients
from utils.points import denormalise_coords
from utils.cropping import move_points_back
from networks.encoders import GlobalAndLocalEncoder
import os
from os.path import join
from tqdm import tqdm


np.random.seed(0) 
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class DisplacementINR:

    def configure(self, config, device):

        self.config = config
        self.device = device

        self.network = ModulatedSiren(in_features=config.in_features, hidden_features=config.hidden_dim, 
                        hidden_layers=config.hidden_layers, out_features=config.out_features, 
                        latent_dim=config.latent_dim, omega=config.omega).to(config.device)
        self.encoder = GlobalAndLocalEncoder(latent_dim=config.latent_dim, 
                                    data_shape=config.cropped_shape).to(config.device)
    
        self.params = list(self.network.parameters()) + list(self.encoder.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=config.lr)

        self.loss_fn = MSELoss()

    def __init__(self, config=None):
        if config is not None: 
            self.configure(config, config.device)

    def _prep_images(self, images):
        """
        Get reference and target image pairs for training.
        Pairs are formed by taking the first frame as the source and all subsequent frames as targets.
        Args:
            images: Image sequence of shape (B, F, H, W).
        Returns:
            images_src: Source images of shape (B, F-1, H, W).
            images_tgt: Target images of shape (B, F-1, H, W).
        """
        n_frames = images.shape[1]
        images_src = images[:, None, 0].repeat(1, n_frames-1, 1, 1)  # repeat the first frame n_frames-1 times
        images_tgt = images[:, 1:]  # all frames except the first one

        return images_src, images_tgt

    def _prep_coords(self, points):
        """
        Get source and target points for training.
        Pairs are formed by taking the first frame points as the source and all subsequent frames as targets
        Args:
            points: Tensor of shape (B, F, N, 2) containing landmark points.
        Returns:
            points_src: Source points of shape (B, F-1, N, 2) or (B, F-1, N, 3) if temporal is added.
            points_tgt: Target points of shape (B, F-1, N, 2).
        """
        n_frames = points.shape[1]
        points_src = points[:, None, 0].repeat(1, n_frames-1, 1, 1)  # repeat the first frame points for all frames
        points_tgt = points[:, 1:]  # all points except the first frame

        return points_src, points_tgt

    def _encode(self, source_images, target_images):
        """
        Predicts latent codes for pairs of frames in the sequence.
        Args:
            encoder: The encoder model, takes source and target images as input.
            source_images: Source images of shape (B, F-1, H, W).
            target_images: Target images of shape (B, F-1, H, W).
        Returns:
            latent_codes: Encoded pairwise vectors of shape (B, F-1, L), where L is the latent dimension.
        """
        batch_size, n_frames, height, width = source_images.shape

        # prepare images for encoder - flatten batch and frames, then add channel dimension
        flat_source_images = source_images.reshape(-1, height, width).unsqueeze(1)  # shape: (B*(F-1), 1, H, W)
        flat_target_images = target_images.reshape(-1, height, width).unsqueeze(1)  # shape: (B*(F-1), 1, H, W)

        flat_global_codes, flat_local_codes = self.encoder(flat_source_images, flat_target_images) # todo - utilise local codes 
        global_latent_dim = flat_global_codes.shape[-1]
        global_codes = flat_global_codes.reshape(batch_size, n_frames, global_latent_dim)

        return global_codes
 
    def _forward_step(self, source_points, source_images, target_images):
        
        """
        Forward step for training or inference.
        Args:
            source_points: Source points of shape (B, F-1, P, D).
            source_images: Source images of shape (B, F-1, H, W).
            target_images: Target images of shape (B, F-1, H, W).
        Returns:
            predicted_positions: Predicted positions of shape (B, F-1, P, D).
            latent_codes: Latent codes of shape (B, F-1, L).
            strain: Strain tensors of shape (B, F-1, P, 2, 2).
            deformation_grad: Deformation gradient tensors of shape (B, F-1, P, 2, 2).
            displacement_grad: Displacement gradient tensors of shape (B, F-1, P, 2, 2).

        """

        batch_size, n_targets, n_points, n_dims = source_points.shape
        n_frames = n_targets + 1  # since source_points has F-1 frames

        global_codes = self._encode(source_images, target_images)  # (B, F-1, L)
        global_codes = global_codes.unsqueeze(2).repeat(1, 1, n_points, 1)  # (B, F-1, P, L)
        flat_global_codes = global_codes.reshape(-1, global_codes.shape[-1]).float()  # (B*(F-1)*P, L)

        flat_source_points = source_points.reshape(-1, self.config.in_features).float()  # (B*(F-1)*P, D)
        flat_source_points.requires_grad_(True)  

        flat_pred_displacements = self.network(flat_source_points, flat_global_codes)

        # spatial derivative of displacements
        flat_strain, flat_deformation_grad, flat_displacement_grad = displacement_gradients(flat_pred_displacements, flat_source_points) 

        # reshape back to original dimensions
        predicted_displacements = flat_pred_displacements.reshape(batch_size, n_frames-1, n_points, self.config.out_features)
        strain = flat_strain.reshape(batch_size, n_frames-1, n_points, 2, 2) 
        deformation_grad = flat_deformation_grad.reshape(batch_size, n_frames-1, n_points, 2, 2)
        displacement_grad = flat_displacement_grad.reshape(batch_size, n_frames-1, n_points, 2, 2)

        return predicted_displacements, global_codes, strain, deformation_grad, displacement_grad


    def train(self):

        checkpoint_dir = os.path.join(self.config.model_dir, self.config.project_name, self.config.experiment_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        train_set = IPDataset(self.config.training_set, crop_size=self.config.cropped_shape)
        val_set = IPDataset(self.config.validation_set, crop_size=self.config.cropped_shape)
        train_set = torch.utils.data.Subset(train_set, range(self.config.training_size)) if self.config.training_size is not None else train_set
        val_set = torch.utils.data.Subset(val_set, range(self.config.validation_size)) if self.config.validation_size is not None else val_set

        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.config.batch_size, 
                                        shuffle=True, drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True)
        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.config.batch_size, 
                                        shuffle=False, drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True)

      
        def step(batch):

            images = batch['images'].to(self.device) # shape (B, F, H, W)
            points = batch['points'].to(self.device) # shape (B, F, P, 2)
            time_coords = batch['time_coords'].to(self.device) # shape (B, F)
            case_names = batch['case']
            original_shapes = batch['original_shape']

            source_images, target_images = self._prep_images(images) # shape (B, F-1, H, W)
            source_points, target_points = self._prep_coords(points) # shape (B, F-1, P, 2)
            time_coords_prep = time_coords[:, 1:, None, None].repeat(1, 1, points.shape[2], 1) # shape (B, F-1, P, 1)
            source_points_with_time = torch.cat((source_points, time_coords_prep), dim=-1)  # shape (B, F-1, P, 3)

            predicted_displacements, latent_codes, strain, deformation_grad, displacement_grad = self._forward_step(source_points_with_time, source_images, target_images)
            predicted_target_points = source_points + predicted_displacements 
            
            deformation_jac_det = torch.det(deformation_grad)  

            position_loss = self.loss_fn(predicted_target_points, target_points)
            jacobian_loss = torch.mean((deformation_jac_det - 1.0) ** 2)  
            latent_loss = torch.mean(torch.linalg.vector_norm(latent_codes, dim=1))

            return position_loss, jacobian_loss, latent_loss, deformation_jac_det
        
        training_losses = []
        validation_losses = []
        
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(self.config.epochs):

            # Epoch level loss
            train_loss_sum = 0
            train_jac_det_sum = 0
            num_train_batches = 0

            # Training step

            self.network.train()
            self.encoder.train()
            for batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}/{self.config.epochs}"):

                train_position_loss, train_jacobian_loss, train_latent_loss, train_jacobian_dets = step(batch)
                train_loss = train_position_loss + self.config.prior_weight * train_latent_loss + self.config.jacobian_weight * train_jacobian_loss

                self.optimizer.zero_grad()
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, max_norm=1.0)
                self.optimizer.step()

                train_loss_sum += train_loss.item()
                train_jac_det_sum += train_jacobian_dets.mean().item()
                num_train_batches += 1

            mean_train_loss = train_loss_sum / num_train_batches
            mean_train_det = train_jac_det_sum / num_train_batches

            # Validation step

            self.network.eval()
            self.encoder.eval()

            val_loss_sum = 0
            val_jac_det_sum = 0
            num_val_batches = 0
            
            for batch in tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}/{self.config.epochs}"):
                
                val_position_loss, val_jacobian_loss, val_latent_loss, val_jacobian_det = step(batch)

                loss = val_position_loss + self.config.prior_weight * val_latent_loss + self.config.jacobian_weight * val_jacobian_loss

                val_loss_sum += loss.item()
                val_jac_det_sum += val_jacobian_det.mean().item()
                num_val_batches += 1
            
            mean_val_loss = val_loss_sum / num_val_batches
            mean_val_det = val_jac_det_sum / num_val_batches

            # Log metrics
            print(f'Epoch {epoch+1}, Train Loss: {mean_train_loss:.6f}, Val Loss: {mean_val_loss:.6f}')
     
            training_losses.append(mean_train_loss)
            validation_losses.append(mean_val_loss)

            # Check for best model
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                best_epoch = epoch
                patience_counter = 0
                
                self._save_model(epoch, mean_train_loss, mean_val_loss, "model_best.pth")
                
                print(f"Saved best model checkpoint at epoch {epoch+1} with val_loss: {mean_val_loss:.6f}")
            else:
                patience_counter += 1
          
            # Check early stopping condition
            if patience_counter >= self.config.patience:
                print(f"Early stopping triggered! No improvement for {self.config.patience} epochs.")
                print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
                break
                    
        
        # Always save final model
        self._save_model(epoch, mean_train_loss, mean_val_loss, "model_final.pth")
        

        print("Training complete")
        print(f"Final validation loss: {validation_losses[-1]:.6f}")
        print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
        
        return training_losses, validation_losses, best_epoch

    def test(self, test_set_name, test_size=None):

        self.predictions_dir = os.path.join(self.config.results_dir, self.config.project_name, self.config.experiment_name, test_set_name)        
        os.makedirs(self.predictions_dir, exist_ok=True)

        test_set = IPDataset(test_set_name, crop_size=self.config.cropped_shape)
        test_set = torch.utils.data.Subset(test_set, range(test_size)) if test_size is not None else test_set

        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False) 

        self.network.eval()
        self.encoder.eval()

        output_summary = {}

        for batch in tqdm(self.test_loader, desc="Testing"):
            images = batch['images'].to(self.device) # shape (B, F, H, W)
            points = batch['points'].to(self.device) # shape (B, F, P, 2)
            time_coords = batch['time_coords'].to(self.device)  # shape (B, F)
            case_name = batch['case']
            slice_name = batch['slice_name']
            px_spacing = batch['px_spacing']
            original_shape = batch['original_shape']
            bounding_box = batch['bounding_box']

            source_images, target_images = self._prep_images(images)
            source_points, target_points = self._prep_coords(points)

            time_coords_prep = time_coords[:, 1:, None, None].repeat(1, 1, points.shape[2], 1) # shape (B, F-1, P, 1)
            source_points_with_time = torch.cat((source_points, time_coords_prep), dim=-1)  # shape (B, F-1, P, 3)

            predicted_displacements, latent_codes, strain, deformation_grad, displacement_grad = self._forward_step(source_points_with_time, source_images, target_images)
            predicted_target_points = source_points_with_time[:, :, :, :2] + predicted_displacements  # add displacements to source points

            # add first frame as this isn't predicted 
            predicted_points = torch.cat((points[:, 0:1], predicted_target_points), dim=1)  # shape (B, F, P, 2)

            # same for strain, but add zeros for first frame
            zero_strains = torch.zeros((strain.shape[0], 1, strain.shape[2], 2, 2), device=strain.device)
            strain = torch.cat((zero_strains, strain), dim=1)  # shape (B, F, P, 2, 2)

            predicted_points = denormalise_coords(predicted_points, self.config.cropped_shape[0])
            
            # remove batch dim and convert to numpy
            predicted_points = predicted_points.squeeze(0).detach().cpu().numpy()
            strain = strain.squeeze(0).detach().cpu().numpy()
            displacements = predicted_displacements.squeeze(0).detach().cpu().numpy()
            original_shape = original_shape.squeeze(0).cpu().numpy()
            bounding_box = bounding_box.squeeze(0).cpu().numpy()
            case_name = case_name[0]
            slice_name = slice_name[0]

            # move points back to original image size
    
            predicted_points = move_points_back(predicted_points, bounding_box, original_shape, images.shape[2:4])

            # save predictions

            points_path = join(self.predictions_dir, f"{case_name}_tracked_points.npy")
            strain_path = join(self.predictions_dir, f"{case_name}_strain.npy")
            np.save(points_path, predicted_points)
            np.save(strain_path, strain)

    def load_model(self, checkpoint_path, device):

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint['config']
        self.configure(config, device)

        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    def _save_model(self, epoch, train_loss, val_loss, filename):
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'epoch': epoch,
            'network_state_dict': self.network.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': train_loss,
            'val_loss': val_loss,
            'config': self.config
        }, checkpoint_path)