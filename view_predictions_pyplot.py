import matplotlib.pyplot as plt
import numpy as np
from utils.grad_strain import rc_strain_maps
import os
from os.path import join
from utils.diff_strain import calculate_all_strains
    
def main():
    project = 'DINR-STACOM'
    experiment = 'base'
    dataset_dir = './data/'
    dataset_name = 'example'
    predictions_dir = f'./results/predictions/{project}/{experiment}/{dataset_name}'
  
    case_files = os.listdir(join(dataset_dir, dataset_name))
    n_slices = len(case_files)
    
    gt_grs_all, gt_gcs_all = [], []
    pred_grs_all, pred_gcs_all = [], []
    pred_grs_autograd_all, pred_gcs_autograd_all = [], []
    
    for i, filename in enumerate(case_files[:10]): 
        case_name = filename.split('.npz')[0]
        
        data = np.load(f'{dataset_dir}/{dataset_name}/{case_name}.npz')
        images = data['images']
        gt_points = data['points']
        pred_points = np.load(f'{predictions_dir}/{case_name}_tracked_points.npy')
        strains_xy = np.load(f'{predictions_dir}/{case_name}_strain.npy')
        
        strains_rr, strains_cc = rc_strain_maps(
            pred_points[0].astype(np.int16), strains_xy, images.shape, point_format='xy'
        )
        
        if i < 5:  
            visualize_slice(images, gt_points, pred_points, strains_rr, strains_cc, i)
        
   
        grs_auto = np.max(np.nanmean(strains_rr, axis=(1, 2)))
        gcs_auto = np.min(np.nanmean(strains_cc, axis=(1, 2)))
        pred_grs_autograd_all.append(grs_auto)
        pred_gcs_autograd_all.append(gcs_auto)
        
        pred_prep = pred_points.transpose((0, 2, 1))[None, :, :, :]
        rr_pred, cc_pred, _, _ = calculate_all_strains(pred_prep)
        pred_grs_all.append(np.max(rr_pred[0, :, 0]))
        pred_gcs_all.append(np.min(cc_pred[0, :, 7]))
        
        gt_prep = gt_points.transpose((0, 2, 1))[None, :, :, :]
        rr_gt, cc_gt, _, _ = calculate_all_strains(gt_prep)
        gt_grs_all.append(np.max(rr_gt[0, :, 0]))
        gt_gcs_all.append(np.min(cc_gt[0, :, 7]))
    
    print_results(gt_grs_all, gt_gcs_all, pred_grs_all, pred_gcs_all, 
                  pred_grs_autograd_all, pred_gcs_autograd_all)


def visualize_slice(images, gt_points, pred_points, strains_rr, strains_cc, slice_idx):

    #Find peak strain frame
    mean_rr = np.nanmean(strains_rr, axis=(1, 2))
    peak_frame = np.argmax(mean_rr[~np.isnan(mean_rr)]) if np.any(~np.isnan(mean_rr)) else 5
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Slice {slice_idx}', fontsize=16)

    axes[0, 0].imshow(images[0], cmap='gray')
    valid = ~np.all(gt_points[0] == 0, axis=1)
    axes[0, 0].scatter(gt_points[0, valid, 0], gt_points[0, valid, 1], 
                      c='white', s=15, alpha=0.8)
    axes[0, 0].set_title('Reference Points (Frame 0)')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(images[peak_frame], cmap='gray')
    valid_pred = ~np.all(pred_points[peak_frame] == 0, axis=1)
    valid_gt = ~np.all(gt_points[peak_frame] == 0, axis=1)
    axes[1, 0].scatter(pred_points[peak_frame, valid_pred, 0], pred_points[peak_frame, valid_pred, 1], 
                      c='blue', s=10, alpha=0.7, label='Predicted')
    axes[1, 0].scatter(gt_points[peak_frame, valid_gt, 0], gt_points[peak_frame, valid_gt, 1], 
                      c='red', s=10, alpha=0.7, label='Ground Truth')
    axes[1, 0].set_title(f'Peak Frame {peak_frame}')
    axes[1, 0].legend()
    axes[1, 0].axis('off')
    
    peak_rr = np.nanmax(strains_rr, axis=0)
    im1 = axes[0, 1].imshow(peak_rr, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[0, 1].set_title('Peak Radial Strain εrr')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    peak_cc = np.nanmin(strains_cc, axis=0)
    im2 = axes[1, 1].imshow(peak_cc, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[1, 1].set_title('Peak Circumferential Strain εcc')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()


def print_results(gt_grs, gt_gcs, pred_grs, pred_gcs, pred_grs_auto, pred_gcs_auto):
    arrays = [np.array(x) for x in [gt_grs, gt_gcs, pred_grs, pred_gcs, pred_grs_auto, pred_gcs_auto]]
    gt_grs, gt_gcs, pred_grs, pred_gcs, pred_grs_auto, pred_gcs_auto = arrays
    
    print("Quantification Results:")
    print(f"GT GRS (Line-segment): {np.mean(gt_grs):.4f} ± {np.std(gt_grs):.4f}")
    print(f"Pred GRS (Line-segment): {np.mean(pred_grs):.4f} ± {np.std(pred_grs):.4f}")
    print(f"Pred GRS (Autograd): {np.mean(pred_grs_auto):.4f} ± {np.std(pred_grs_auto):.4f}")
    print()
    print(f"GT GCS (Line-segment): {np.mean(gt_gcs):.4f} ± {np.std(gt_gcs):.4f}")
    print(f"Pred GCS (Line-segment): {np.mean(pred_gcs):.4f} ± {np.std(pred_gcs):.4f}")
    print(f"Pred GCS (Autograd): {np.mean(pred_gcs_auto):.4f} ± {np.std(pred_gcs_auto):.4f}")
    print()
    print(f"GRS Error (Line-segment): {np.mean(np.abs(gt_grs - pred_grs)):.4f}")
    print(f"GRS Error (Autograd): {np.mean(np.abs(gt_grs - pred_grs_auto)):.4f}")
    print(f"GCS Error (Line-segment): {np.mean(np.abs(gt_gcs - pred_gcs)):.4f}")
    print(f"GCS Error (Autograd): {np.mean(np.abs(gt_gcs - pred_gcs_auto)):.4f}")


if __name__ == '__main__':
    main()