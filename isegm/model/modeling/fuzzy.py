"""
Fuzzy Connectedness Segmentation without an External Mask
---------------------------------------------------------
This script performs fuzzy connectedness segmentation using multiple seed points.
If a fixed mask is not provided (or is set to None), the algorithm uses a closeness
threshold (based on the pixel affinity) to decide if a neighbor should be processed.
This allows the segmentation to focus on the region around the seeds based solely on
the image's edge characteristics.
"""

import cv2
import numpy as np
import heapq
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_gradient_magnitude, uniform_filter


def fuzzy_connectedness(color_image, seeds, mask=None, sigma=1, connectivity=4, closeness_threshold=0.95,image_format='RGB'):
    """
    Perform fuzzy connectedness segmentation constrained either to a provided mask or
    based on the closeness of neighbors (via a threshold on affinity).

    Parameters:
    -----------
    color_image : numpy.ndarray
        Input color image (H x W x 3).
    seeds : list of tuples
        List of (row, column) seed coordinates.
    mask : numpy.ndarray or None
        Optional binary mask (H x W) where 255 indicates the region of interest.
        If None, the algorithm uses closeness thresholding.
    sigma : float, optional
        Standard deviation for Gaussian gradient smoothing (default: 1).
    connectivity : int, optional
        Neighborhood connectivity to use for region growing (4 or 8; default: 4).
    closeness_threshold : float, optional
        Threshold on affinity for allowing neighbor propagation when mask is not used (default: 0.5).

    Returns:
    --------
    tuple: (processed_images, fuzzy_map, gradient, affinity)
        processed_images : dict
            Dictionary containing intermediate images for visualization:
            - 'original': The original color image (converted to RGB for display).
            - 'masked_color': The color image (or masked version if a mask is provided).
            - 'masked_gray': The grayscale version of the image.
       fuzzy_output: dict
            - 'fuzzy_map_pos': numpy.ndarray. Positive fuzzy connectivity map (H x W).
            - 'fuzzy_map_neg': numpy.ndarray. Negative fuzzy connectivity map (H x W).
            - 'gradient' : numpy.ndarray. The computed gradient magnitude map (H x W).
            - 'affinity' : numpy.ndarray. The pixel affinity map derived from the gradient (H x W).
    """
    # ---------------------------
    # 1. Image Preprocessing
    # ---------------------------
    if color_image.ndim != 3 or color_image.shape[2] != 3:
        raise ValueError("color_image must be a 3-channel (H x W x 3) image.")

    color_image  = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) if image_format == 'BGR' else color_image
    
    # Get image dimensions from the mask if provided, otherwise from the color image.
    h, w = color_image.shape[:2]
    
    # Convert the color image to grayscale.
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    
    # If a mask is provided, apply it; otherwise, work with the full image.
    if mask is not None:
        mask = (mask > 0).astype(np.uint8)  # Ensure mask is binary.
        # Here we could use bitwise_and if needed, but for now we assume the mask is only used in region growing.
        masked_color = cv2.bitwise_and(color_image, color_image, mask=mask)
        masked_gray = cv2.bitwise_and(gray_image, gray_image, mask=mask)
    else:
        masked_color = color_image.copy()
        masked_gray = gray_image.copy()
    
    # ------------------------------------
    # 2. Compute Gradient & Affinity Maps
    # ------------------------------------
    # Compute gradient magnitude with Gaussian smoothing to reduce noise.
    gradient = gaussian_gradient_magnitude(masked_gray.astype(np.float32), sigma=sigma)
    
    # Normalize gradient (avoid division by zero).
    max_gradient = gradient.max() if gradient.max() != 0 else 1e-7
    # Compute affinity: higher affinity corresponds to smoother regions (lower gradient).
    affinity = np.exp(-gradient / max_gradient) ###
    
    # ----------------------------
    # 3. Neighborhood Setup
    # ----------------------------
    # Define neighbor offsets based on connectivity.
    neighbor_config = {
        4: [(-1, 0), (1, 0), (0, -1), (0, 1)],
        8: [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),          (0, 1),
            (1, -1),  (1, 0), (1, 1)]
    }
    try:
        neighbors = neighbor_config[connectivity]
    except KeyError:
        raise ValueError(f"Invalid connectivity {connectivity}. Must be 4 or 8.")
    
    
    
    # def _compute_local_statistics(image, neighborhood_size):
    #     local_mean = uniform_filter(image, neighborhood_size)
    #     local_sqr_mean = uniform_filter(image**2, neighborhood_size)
    #     local_variance = local_sqr_mean - local_mean**2
    #     local_std = np.sqrt(local_variance)
    #     return local_mean, local_std
    
    
    # local_mean, local_std = _compute_local_statistics(image, connectivity)
    
    
    # Region growing helper function
    def _grow_regions(seeds):
        # Initialize the fuzzy map (to store connectivity strength for each pixel).
        fuzzy_map = np.zeros((h, w), dtype=np.float32)
        # ---------------------------------
        # 4. Region Growing Setup (Heap)
        # ---------------------------------
        # Priority queue (heap) stores pixels to process, prioritized by connectivity strength.
        pq = []
        visited = set()
        # Add seeds to the priority queue.
        for seed in seeds:
            row, col = seed
            row, col = int(row), int(col)  # Ensure integer coordinates
            if 0 <= row < h and 0 <= col < w:
                # If a mask is provided, ensure the seed is inside the mask.
                if mask is not None and mask[row, col] == 0:
                    continue
                initial_strength = affinity[row, col]
                # Push negative strength because heapq implements a min-heap.
                heapq.heappush(pq, (-initial_strength, (row, col)))
        
        # ---------------------------------
        # 5. Region Growing Process
        # ---------------------------------
        while pq:
            strength, (x, y) = heapq.heappop(pq)
            strength = -strength  # Convert back to positive value.
            
            if (x, y) in visited:
                continue
            
            visited.add((x, y))
            fuzzy_map[x, y] = strength
            
            # Process each neighbor.
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                # Check neighbor bounds.
                if not (0 <= nx < h and 0 <= ny < w):
                    continue
                if (nx, ny) in visited:
                    continue
                
                # If a mask is provided, process only if the neighbor is inside the mask.
                # Otherwise, process the neighbor if its affinity is above the closeness threshold.
                if mask is not None:
                    condition = mask[nx, ny] > 0
                else:
                    condition = affinity[nx, ny] >= closeness_threshold
                    # intensity = image[x, y]
                    # threshold = local_mean[x, y] + 2 * local_std[x, y]
                    # neighbor_intensity = image[nx, ny]
                    # condition =  abs(neighbor_intensity - intensity) <= threshold
                
                if not np.any(condition):
                    continue        
                
                # Calculate edge affinity as the minimum of current pixel's affinity and neighbor's.
                edge_affinity = min(affinity[x, y], affinity[nx, ny])
                # Update connectivity strength for the neighbor.
                new_strength = min(strength, edge_affinity)
            
                # intensity based affinities.
                #diff = (image[x, y].astype(float) - image[nx, ny].astype(float))**2
                #diff = np.linalg.norm(image[x, y].astype(float) - image[nx, ny].astype(float))
                #intensity_affinity = np.exp(- (diff) / (2 * (delta ** 2)))
                
                # Combine using the product of gradient and intensity affinities.
                #combined_affinity = (edge_affinity * intensity_affinity)
                #new_strength = min(strength, intensity_affinity)
                
                # if new_strength > 0.8:  # Connectivity threshold
                #         heapq.heappush(pq, (-new_strength, (ny, nx)))
                
                heapq.heappush(pq, (-new_strength, (nx, ny)))
        
        return fuzzy_map

    # Process both seed types
    fuzzy_pos = _grow_regions(seeds.get('pos_seeds', [])) if seeds.get('pos_seeds') else None
    fuzzy_neg = _grow_regions(seeds.get('neg_seeds', [])) if seeds.get('neg_seeds') else None
    
    # Apply thresholding and resolve overlaps
    if fuzzy_pos is not None:
        fuzzy_pos = np.where(fuzzy_pos >= 0.5, fuzzy_pos, 0)

    if fuzzy_neg is not None:
        fuzzy_neg = np.where(fuzzy_neg >= 0.5, fuzzy_neg, 0)

    # Resolve overlaps: if both maps are non-zero in the same location, clear both
    if fuzzy_pos is not None and fuzzy_neg is not None:
        overlap_mask = (fuzzy_pos > 0) & (fuzzy_neg > 0)
        fuzzy_pos[overlap_mask] = 0
        fuzzy_neg[overlap_mask] = 0
    
    # ---------------------------
    # 6. Prepare Processed Images for Visualization
    # ---------------------------
    processed_images = {
        'original': color_image,
        'masked_color': masked_color,
        'masked_gray': masked_gray
    }
    
    fuzzy_output = {
        'fuzzy_map_pos': fuzzy_pos,
        'fuzzy_map_neg': fuzzy_neg,
        'gradient': gradient,
        'affinity': affinity
    }
    
    return processed_images, fuzzy_output


def fuzzy_connectedness_batch(batched_image, batched_seeds,batch_mask = None, sigma=1, connectivity=4,closeness_threshold = 0.5, image_format='normalized'):
    """
    Process a batched image tensor of shape (B, C, H, W) with corresponding seeds.
    
    Parameters:
        batched_image: numpy.ndarray of shape (B, C, H, W)
        batched_seeds: list of length B, where each element is a list of seed tuples (row, col)
    
    Returns:
        List of results per image. Each result is a tuple:
          (processed_images, fuzzy_map, gradient, affinity)
    """
    all_results = {}
    B, C, H, W = batched_image.shape
    for i in range(B):
        # Convert each image from (C, H, W) to (H, W, C)
        img_np = batched_image[i].clone().permute(1,2,0).cpu().numpy() # unnormalize
        img_np = img_np * 255.0 if image_format == 'normalized' else img_np
        #img_np = np.transpose(img_np, (1, 2, 0))
        seeds = batched_seeds[i]  # List of seed tuples for image i
        processed_images, fuzzy_output = fuzzy_connectedness(img_np, seeds, batch_mask, sigma, connectivity, closeness_threshold)
        all_results[i] = {'processed_images': processed_images, 'fuzzy_output': fuzzy_output}
    return all_results

## get points
#points[:,:points.shape[1] // 2,:]
#points[:,:points.shape[1] // 2,:][0][points[:,:points.shape[1] // 2,:][0][:, 0] != -1][:, :2]

def visualize_fuzzy(processed_images,fuzzy_output,seeds):
    # ---------------------------
    # Visualization
    # ---------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original image with positive and negative seeds marked
    axes[0, 0].imshow(processed_images['original'])

    # Plot positive seeds in red
    if 'pos_seeds' in seeds and seeds['pos_seeds'] is not None:
        for seed in seeds['pos_seeds']:
            axes[0, 0].scatter(seed[1], seed[0], color="red", marker="o", s=30, label='Positive Seed')

    # Plot negative seeds in blue
    if 'neg_seeds' in seeds and seeds['neg_seeds'] is not None:
        for seed in seeds['neg_seeds']:
            axes[0, 0].scatter(seed[1], seed[0], color="blue", marker="o", s=30, label='Negative Seed')

    axes[0, 0].set_title("Original Image with Seeds")
    
    
    # Optional: avoid duplicate labels in legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0, 0].legend(unique.values(), unique.keys(), loc='lower right')
        

    # Masked grayscale image (or full grayscale if mask is None).
    img_gray = axes[0, 1].imshow(processed_images['masked_gray'], cmap="gray")
    axes[0, 1].set_title("Grayscale Image")
    #axes[0, 1].axis("off")
    fig.colorbar(img_gray, ax=axes[0, 1], fraction=0.027, pad=0.04)

    # Gradient magnitude map.
    img_grad = axes[1, 0].imshow(fuzzy_output['gradient'], cmap="inferno")
    axes[1, 0].set_title("Gradient Magnitude")
    axes[1, 0].axis("off")
    fig.colorbar(img_grad, ax=axes[1, 0], fraction=0.027, pad=0.04)

    # Affinity map.
    img_affinity = axes[1, 1].imshow(fuzzy_output['affinity'], cmap="viridis")
    axes[1, 1].set_title("Affinity Map")
    axes[1, 1].axis("off")
    fig.colorbar(img_affinity, ax=axes[1, 1], fraction=0.027, pad=0.04)


    ## Plot Fuzzy maps
    maps_and_titles = [
    (fuzzy_output['fuzzy_map_pos'], "Fuzzy Connectedness Map (Positive)"),
    (fuzzy_output['fuzzy_map_neg'], "Fuzzy Connectedness Map (Negative)")
]

    # Filter out None maps
    valid_maps = [(fuzzy_map, title) for fuzzy_map, title in maps_and_titles if fuzzy_map is not None]

    if valid_maps:
        fig, axes = plt.subplots(1, len(valid_maps), figsize=(6 * len(valid_maps), 5))
        if len(valid_maps) == 1:
            axes = [axes]  # Make iterable if only one axis
        
        for ax, (fmap, title) in zip(axes, valid_maps):
            img = ax.imshow(fmap, cmap="jet")
            ax.set_title(title)
            ax.axis("off")
            fig.colorbar(img, ax=ax, fraction=0.027, pad=0.04)

        
        #plt.show()
    plt.tight_layout()
    plt.show()


def fuzzy_connectedness_batch(batched_image, points, sigma=1, connectivity=4,closeness_threshold=0.95, image_scale='normalized'):
    """
    Process a batched image tensor of shape (B, C, H, W) with corresponding seeds.
    
    Parameters:
        batched_image: numpy.ndarray of shape (B, C, H, W)
        batched_seeds: list of length B, where each element is a list of seed tuples (row, col)
    
    Returns:
        List of results per image. Each result is a tuple:
          (processed_images, fuzzy_map, gradient, affinity)
    """
    all_results = {}
    B, C, H, W = batched_image.shape
    
    # split points into positive and negative
    pos_seeds = points[:,:points.shape[1] // 2,:]
    neg_seeds = points[:,points.shape[1] // 2:,:]
        
    # get points for each image in batch, remove -1 points and convert to numpy
    seeds = {
       'pos_seeds': [(pos_seeds[i][pos_seeds[i][:, 0] != -1][:, :2]).cpu().numpy() for i in range(pos_seeds.shape[0])],
       'neg_seeds': [(neg_seeds[i][neg_seeds[i][:, 0] != -1][:, :2]).cpu().numpy() for i in range(neg_seeds.shape[0])]
    }
    
    for i in range(B):
        # Convert each image from (C, H, W) to (H, W, C)
        img_np = batched_image[i].clone().permute(1,2,0).numpy() # unnormalize
        img_np = img_np * 255.0 if image_scale == 'normalized' else img_np
        #img_np = np.transpose(img_np, (1, 2, 0))
        seeds = {
            'pos_seeds': seeds['pos_seeds'][i],
            'neg_seeds': seeds['neg_seeds'][i]
            }
        # List of seed tuples for image i
        #processed_images, fuzzy_output = fuzzy_connectedness(img_np, seeds, sigma, connectivity, expansion_radius)
        processed_images, fuzzy_output = fuzzy_connectedness(img_np, seeds = seeds, sigma =sigma,
                                                     closeness_threshold=closeness_threshold,
                                                     connectivity=connectivity,image_format='RGB')
        all_results[i] = {'processed_images': processed_images, 'fuzzy_output': fuzzy_output}
    return all_results

## get points
#points[:,:points.shape[1] // 2,:]
#points[:,:points.shape[1] // 2,:][0][points[:,:points.shape[1] // 2,:][0][:, 0] != -1][:, :2]
