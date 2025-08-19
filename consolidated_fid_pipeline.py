#!/usr/bin/env python3
"""
Consolidated FID Pipeline Script

This script combines face alignment, per-image FID optimization, and FID calculation
into a single pipeline for comparing enhanced, raw, and reference datasets.

Usage:
    python consolidated_fid_pipeline.py enhanced_dataset raw_dataset reference_dataset [options]
"""

import os
import pathlib
import shutil
import time
import argparse
from pathlib import Path

import face_recognition
from PIL import Image, ImageEnhance
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
from torch.nn.functional import adaptive_avg_pool2d
from scipy import linalg
from sklearn.metrics import pairwise_distances

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from pytorch_fid.inception import InceptionV3

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp', 'gif'}

class FastImageDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = read_image(str(self.files[idx])).float()
        img = convert_image_dtype(img, torch.float32)
        mean = img.mean(dim=(1,2), keepdim=True)
        std = img.std(dim=(1,2), keepdim=True)
        img = (img - mean) / (std + 1e-8)
        return img

def convert_to_jpeg(input_folder, jpeg_folder):
    """Convert all images in input folder to JPEG format."""
    input_path = Path(input_folder)
    jpeg_path = Path(jpeg_folder)

    if not input_path.exists():
        print(f"Input folder {input_folder} does not exist!")
        return []

    jpeg_path.mkdir(exist_ok=True)
    print(f"JPEG conversion folder: {jpeg_path}")

    image_files = [f for f in input_path.iterdir()
                   if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}]

    if not image_files:
        print(f"No image files found in {input_folder}")
        return []

    print(f"Converting {len(image_files)} images to JPEG format...")

    converted_files = []
    successful = 0
    failed = 0

    for i, image_file in enumerate(image_files):
        print(f"Converting {i+1}/{len(image_files)}: {image_file.name}")

        try:
            with Image.open(image_file) as img:
                if img.mode != 'RGB':
                    if img.mode == 'RGBA':
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[-1])
                        img = background
                    else:
                        img = img.convert('RGB')

                jpeg_filename = f"{image_file.stem}.jpg"
                jpeg_filepath = jpeg_path / jpeg_filename
                img.save(jpeg_filepath, 'JPEG', quality=95, optimize=True)

                converted_files.append(str(jpeg_filepath))
                successful += 1

        except Exception as e:
            print(f"Error converting {image_file.name}: {str(e)}")
            failed += 1

    print(f"JPEG conversion complete! Successfully: {successful}, Failed: {failed}")
    return converted_files

def align_and_crop_to_celeba(image_path, output_size=(178, 218)):
    """Align and crop a face image to match CelebA format."""
    try:
        image = face_recognition.load_image_file(image_path)
        face_landmarks_list = face_recognition.face_landmarks(image)
        
        if not face_landmarks_list:
            print(f"No face detected in {image_path}")
            return None
        
        landmarks = face_landmarks_list[0]
        left_eye = np.mean(landmarks['left_eye'], axis=0)
        right_eye = np.mean(landmarks['right_eye'], axis=0)
        
        eye_distance = np.linalg.norm(right_eye - left_eye)
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        eyes_center = np.array([(left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2])
        
        pil_img = Image.fromarray(image)
        original_width, original_height = pil_img.size
        
        rotated_img = pil_img.rotate(angle, center=tuple(eyes_center), resample=Image.BICUBIC, expand=True)
        new_width, new_height = rotated_img.size
        
        angle_rad = np.radians(angle)
        expand_dx = (new_width - original_width) / 2
        expand_dy = (new_height - original_height) / 2
        
        new_eyes_center = eyes_center + np.array([expand_dx, expand_dy])
        
        target_eye_distance = 80
        scale_factor = target_eye_distance / eye_distance
        
        base_width, base_height = output_size
        crop_width = int(base_width / scale_factor)
        crop_height = int(base_height / scale_factor)
        
        eye_y_ratio = 0.51
        
        left = int(new_eyes_center[0] - crop_width // 2)
        top = int(new_eyes_center[1] - crop_height * eye_y_ratio)
        right = left + crop_width
        bottom = top + crop_height
        
        pad_left = max(0, -left)
        pad_top = max(0, -top)
        pad_right = max(0, right - new_width)
        pad_bottom = max(0, bottom - new_height)
        
        if any([pad_left, pad_top, pad_right, pad_bottom]):
            padded_img = Image.fromarray(np.pad(
                np.array(rotated_img),
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='edge'
            ))
            
            left = left + pad_left
            top = top + pad_top
            right = left + crop_width
            bottom = top + crop_height
            
            cropped_img = padded_img.crop((left, top, right, bottom))
        else:
            left = max(0, min(left, new_width - crop_width))
            top = max(0, min(top, new_height - crop_height))
            right = left + crop_width
            bottom = top + crop_height
            
            cropped_img = rotated_img.crop((left, top, right, bottom))
        
        final_img = cropped_img.resize(output_size, Image.LANCZOS)
        return final_img
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def align_dataset(input_folder, output_folder, output_size=(178, 218), keep_jpeg_folder=False):
    """Process all face images in a folder to match CelebA format."""
    input_path = Path(input_folder)

    if not input_path.exists():
        print(f"Input folder {input_folder} does not exist!")
        return None

    jpeg_folder = input_path.parent / f"{input_path.name}_jpeg_temp"
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    print(f"=== Aligning dataset: {input_folder} ===")
    print(f"Output folder: {output_path}")

    # Convert to JPEG
    jpeg_files = convert_to_jpeg(input_folder, jpeg_folder)
    if not jpeg_files:
        print("No images were successfully converted to JPEG. Skipping.")
        return None

    # Process JPEG images
    print(f"Processing {len(jpeg_files)} JPEG images for face alignment...")
    successful = 0
    failed = 0

    for i, jpeg_file in enumerate(jpeg_files):
        jpeg_path = Path(jpeg_file)
        print(f"Processing {i+1}/{len(jpeg_files)}: {jpeg_path.name}")

        jpeg_file = str(jpeg_file).replace("\\", "\\\\")
        aligned_img = align_and_crop_to_celeba(jpeg_file, output_size)

        if aligned_img is not None:
            original_name = jpeg_path.stem
            if original_name.endswith('_temp'):
                original_name = original_name[:-5]

            output_file = output_path / f"{original_name}_aligned.jpg"
            aligned_img.save(output_file, 'JPEG', quality=95)
            successful += 1
        else:
            failed += 1

    # Cleanup
    if not keep_jpeg_folder:
        try:
            shutil.rmtree(jpeg_folder)
            print(f"Removed intermediate JPEG folder: {jpeg_folder}")
        except Exception as e:
            print(f"Warning: Could not remove intermediate JPEG folder: {str(e)}")

    print(f"Alignment complete! Successfully: {successful}, Failed: {failed}")
    return output_path if successful > 0 else None

def get_activations(files, model, batch_size=64, dims=2048, device='cuda', num_workers=4):
    """Extract features from images using InceptionV3."""
    model.eval()
    dataset = FastImageDataset(files)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                           num_workers=num_workers, pin_memory=True)
    pred_arr = np.empty((len(files), dims))
    start_idx = 0
    for batch in tqdm(dataloader, desc="Extracting activations"):
        batch = batch.to(device, non_blocking=True)
        with torch.no_grad():
            pred = model(batch)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2)
            pred_arr[start_idx:start_idx + pred.size(0)] = pred.cpu().numpy()
            start_idx += pred.size(0)
    return pred_arr

def fast_fid(mu1, sigma1, feats2):
    """Calculate FID between precomputed stats and features."""
    mu2 = np.mean(feats2, axis=0)
    if feats2.shape[0] == 1:
        sigma2 = np.zeros((feats2.shape[1], feats2.shape[1]))
    else:
        sigma2 = np.cov(feats2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return fid

def list_images(folder):
    """List all image files in a folder."""
    return sorted([
        os.path.join(folder, fname)
        for fname in os.listdir(folder)
        if fname.split('.')[-1].lower() in IMAGE_EXTENSIONS
    ])

def select_reference_subset(reference_dir, enhanced_dir, batch_size=64, num_workers=4, 
                           device='cuda', dims=2048, k=1, output_dir=None):
    """Select subset of reference images based on k-NN with enhanced dataset."""
    reference_files = list_images(reference_dir)
    enhanced_files = list_images(enhanced_dir)
    print(f"Found {len(reference_files)} reference images, {len(enhanced_files)} enhanced images.")

    # Load InceptionV3
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    # Extract features
    print("Extracting features for reference images...")
    features_ref = get_activations(reference_files, model, batch_size, dims, device, num_workers)
    print("Extracting features for enhanced images...")
    features_enh = get_activations(enhanced_files, model, batch_size, dims, device, num_workers)

    # Find k-th nearest neighbors
    print(f"Finding the {k}-th nearest neighbors...")
    dists = pairwise_distances(features_ref, features_enh)
    selected_indices = set()
    for i in range(dists.shape[0]):
        sorted_indices = np.argsort(dists[i])
        neighbor_idx = sorted_indices[k-1]
        selected_indices.add(neighbor_idx)
    
    selected_indices = list(selected_indices)
    selected_files = [enhanced_files[i] for i in selected_indices]
    selected_feats = features_enh[selected_indices]

    # Compute FID for validation
    mu_ref = np.mean(features_ref, axis=0)
    sigma_ref = np.cov(features_ref, rowvar=False)
    fid = fast_fid(mu_ref, sigma_ref, selected_feats)
    print(f"FID between reference and selected enhanced subset: {fid:.4f}")

    # Copy selected images to output directory
    if output_dir is None:
        output_dir = Path(enhanced_dir).parent / "selected_enhanced_subset"
    else:
        output_dir = Path(output_dir)
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Copying {len(selected_files)} selected images to {output_dir}")
    for src_file in selected_files:
        shutil.copy(src_file, output_dir)

    print(f"Selected subset saved to: {output_dir}")
    return str(output_dir)

def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                  device='cpu', num_workers=1):
    """Calculate activation statistics for FID computation."""
    act = get_activations(files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Frechet distance between two multivariate Gaussians."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape
    
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def calculate_fid(path1, path2, batch_size, device, dims, num_workers=1):
    """Calculate FID between two datasets."""
    if not os.path.exists(path1):
        raise RuntimeError('Invalid path: %s' % path1)
    if not os.path.exists(path2):
        raise RuntimeError('Invalid path: %s' % path2)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    # Dataset 1
    path1 = pathlib.Path(path1)
    files1 = sorted([file for ext in IMAGE_EXTENSIONS
                    for file in path1.glob('*.{}'.format(ext))])
    if not files1:
        raise RuntimeError(f'No images found in {path1}')
    print(f"Computing statistics for dataset 1 ({len(files1)} images)...")
    mu1, sigma1 = calculate_activation_statistics(files1, model, batch_size, dims, device, num_workers)

    # Dataset 2
    path2 = pathlib.Path(path2)
    files2 = sorted([file for ext in IMAGE_EXTENSIONS
                    for file in path2.glob('*.{}'.format(ext))])
    if not files2:
        raise RuntimeError(f'No images found in {path2}')
    print(f"Computing statistics for dataset 2 ({len(files2)} images)...")
    mu2, sigma2 = calculate_activation_statistics(files2, model, batch_size, dims, device, num_workers)

    # Calculate FID
    print("Calculating FID...")
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f"FID: {fid:.4f}")
    return fid

def main():
    parser = argparse.ArgumentParser(
        description='Consolidated FID Pipeline: Align datasets, select reference subset, and calculate FID scores',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('enhanced_dataset', help='Path to enhanced dataset')
    parser.add_argument('raw_dataset', help='Path to raw dataset')
    parser.add_argument('reference_dataset', help='Path to reference dataset')
    
    # Optional arguments
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (cuda:0, cpu, etc.)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--dims', type=int, default=2048, choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                       help='Dimensionality of Inception features')
    parser.add_argument('--k', type=int, default=1, help='k-th nearest neighbor for selection')
    parser.add_argument('--no-subset-selection', action='store_true', 
                       help='Skip subset selection and use full reference dataset')
    parser.add_argument('--output-size', nargs=2, type=int, default=[178, 218], 
                       help='Output size for aligned images (width height)')
    parser.add_argument('--keep-jpeg', action='store_true', 
                       help='Keep intermediate JPEG folders after alignment')
    parser.add_argument('--output-dir', help='Base output directory (default: parent of enhanced dataset)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda:0' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Using {args.num_workers} workers")
    
    # Setup output directory
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        base_output_dir = Path(args.enhanced_dataset).parent
    
    base_output_dir.mkdir(exist_ok=True)
    
    # Step 1: Align all datasets
    print("\n" + "="*60)
    print("STEP 1: ALIGNING DATASETS")
    print("="*60)
    
    enhanced_aligned = base_output_dir / f"{Path(args.enhanced_dataset).name}_aligned"
    raw_aligned = base_output_dir / f"{Path(args.raw_dataset).name}_aligned"
    reference_aligned = base_output_dir / f"{Path(args.reference_dataset).name}_aligned"
    
    output_size = tuple(args.output_size)
    
    print("Aligning enhanced dataset...")
    enhanced_result = align_dataset(args.enhanced_dataset, enhanced_aligned, output_size, args.keep_jpeg)
    
    print("\nAligning raw dataset...")
    raw_result = align_dataset(args.raw_dataset, raw_aligned, output_size, args.keep_jpeg)
    
    print("\nAligning reference dataset...")
    reference_result = align_dataset(args.reference_dataset, reference_aligned, output_size, args.keep_jpeg)
    
    if not all([enhanced_result, raw_result, reference_result]):
        print("ERROR: One or more datasets failed to align properly!")
        return
    
    # Step 2: Select reference subset (optional)
    if not args.no_subset_selection:
        print("\n" + "="*60)
        print("STEP 2: SELECTING REFERENCE SUBSET")
        print("="*60)
        
        selected_reference_dir = select_reference_subset(
            str(reference_aligned), str(enhanced_aligned),
            args.batch_size, args.num_workers, device, args.dims, args.k,
            base_output_dir / "selected_reference_subset"
        )
        reference_for_fid = selected_reference_dir
    else:
        print("\n" + "="*60)
        print("STEP 2: SKIPPING SUBSET SELECTION (using full reference dataset)")
        print("="*60)
        reference_for_fid = str(reference_aligned)
    
    # Step 3: Calculate FID scores
    print("\n" + "="*60)
    print("STEP 3: CALCULATING FID SCORES")
    print("="*60)
    
    print(f"\nCalculating FID: Reference vs Enhanced")
    print(f"Reference: {reference_for_fid}")
    print(f"Enhanced: {enhanced_aligned}")
    fid_enhanced = calculate_fid(
        reference_for_fid, str(enhanced_aligned),
        args.batch_size, device, args.dims, args.num_workers
    )
    
    print(f"\nCalculating FID: Reference vs Raw")
    print(f"Reference: {reference_for_fid}")
    print(f"Raw: {raw_aligned}")
    fid_raw = calculate_fid(
        reference_for_fid, str(raw_aligned),
        args.batch_size, device, args.dims, args.num_workers
    )
    
    # Final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"FID (Reference vs Enhanced): {fid_enhanced:.4f}")
    print(f"FID (Reference vs Raw): {fid_raw:.4f}")
    print(f"Improvement: {fid_raw - fid_enhanced:.4f}")
    
    if not args.no_subset_selection:
        print(f"\nReference subset used: {reference_for_fid}")
    else:
        print(f"\nFull reference dataset used: {reference_for_fid}")
    
    print(f"\nAligned datasets saved to:")
    print(f"  Enhanced: {enhanced_aligned}")
    print(f"  Raw: {raw_aligned}")
    print(f"  Reference: {reference_aligned}")

if __name__ == "__main__":
    main()