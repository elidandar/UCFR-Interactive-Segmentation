import numpy as np
from copy import deepcopy
import cv2
import os
from skimage.measure import label, regionprops

import pdb


def get_adaptive_radius(error_mask, min_radius=1, max_radius=5, scale_factor=0.1):
    """
    Computes an adaptive click radius using region area and local error density.
    Parameters:
    -----------
    error_mask : np.ndarray
        Binary mask where 1 indicates error pixels.
    min_radius : int
        Minimum allowable radius.
    max_radius : int
        Maximum allowable radius.
    scale_factor : float
        Scaling factor for square root of region area.
    Returns:
    --------
    int : adaptive radius in [min_radius, max_radius]
    """
    labeled, _ = label(error_mask, return_num=True)
    regions = regionprops(labeled)
    if not regions:
        return min_radius

    largest_region = max(regions, key=lambda x: x.area)
    area = largest_region.area

    # Area-based adjustment
    radius = min_radius + scale_factor * np.sqrt(area)

    # Local density-based refinement
    local_patch = error_mask[largest_region.slice]
    local_ratio = np.mean(local_patch)
    
    radius *= (1.0 + local_ratio)

    return int(np.clip(round(radius), min_radius, max_radius))

def get_local_error_region(mask, center, window_size=15):
        y, x = center
        half = window_size // 2
        y1, y2 = max(0, y - half), min(mask.shape[0], y + half + 1)
        x1, x2 = max(0, x - half), min(mask.shape[1], x + half + 1)
        return mask[y1:y2, x1:x2]



# =================================================================================
# --- All G-CFR Helper Functions
# =================================================================================
def get_boundary_uncertainty_with_edges(segmentation_prob, edge_prob, alpha=0.5):
    """
    Combines segmentation uncertainty with both Sobel gradients and model's edge predictions.
    
    Args:
        segmentation_prob: HxW array, predicted probabilities for segmentation.
        edge_prob: HxW array, predicted probabilities for edges (from edge head).
        alpha: weight balancing Sobel gradients vs edge head [0,1].
    """
    # --- Standard uncertainty (high near 0.5) ---
    uncertainty_map = 1.0 - np.abs(2.0 * segmentation_prob - 1.0)

    # --- Sobel gradient on segmentation prob ---
    prob_image_8u = (segmentation_prob * 255).astype(np.uint8)
    grad_x = cv2.Sobel(prob_image_8u, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(prob_image_8u, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    if np.max(gradient_magnitude) > 0:
        gradient_magnitude /= np.max(gradient_magnitude)

    # --- Fuse Sobel + Edge head ---
    boundary_map = alpha * gradient_magnitude + (1 - alpha) * edge_prob

    # --- Boundary-aware uncertainty ---
    boundary_uncertainty = uncertainty_map * boundary_map

    return boundary_uncertainty


# =================================================================================
#                                ---  Clicker Class ---
# =================================================================================
class Clicker(object):
        
    def __init__(self, gt_mask=None, adaptive_radius=None, init_clicks=None, 
                 ignore_label=-1, click_indx_offset=0):
        """
        Initializes a Clicker object with a specified click strategy.

        Parameters:
        -----------
        gt_mask : np.ndarray, optional
            Ground truth segmentation mask (binary, where 1 indicates the object).
        adaptive_radius : bool, optional
            If True, uses adaptive radius based on local error density. Defaults to False.
        init_clicks : list, optional
            A list of initial Click objects.
        ignore_label : int, optional
            Label value in gt_mask to ignore. Defaults to -1.
        click_indx_offset : int, optional
            Offset for click indices. Defaults to 0.
        """
        self.click_indx_offset = click_indx_offset
        self.adaptive_radius = adaptive_radius if adaptive_radius is not None else False        
        self.click_counter = 0 # To name visualization files uniquely
        
        if gt_mask is not None:
            self.gt_mask = gt_mask == 1
            self.not_ignore_mask = gt_mask != ignore_label
        else:
            self.gt_mask = None

        self.reset_clicks()

        if init_clicks is not None:
            for click in init_clicks:
                self.add_click(click)
        
    def make_next_click(self, pred_mask):
        #assert self.gt_mask is not None, "Ground truth mask must be provided for click selection in evaluation."
        assert self.gt_mask is not None or hasattr(self, 'fn_mask'), \
            "Ground truth or pre-computed error masks must be provided for click selection."
        
        
        click, click_data_for_vis = self._get_next_click(pred_mask)
        
        if click is None:
            return None, {}
        self.add_click(click)
        return click, click_data_for_vis  
        
    def get_clicks(self, clicks_limit=None):
        return self.clicks_list[:clicks_limit]

    def _get_next_click(self, pred_mask, alternate = False,  padding=True):
        
        if pred_mask is not None:
            self.fn_mask = np.logical_and(np.logical_and(self.gt_mask, np.logical_not(pred_mask)), self.not_ignore_mask)
            self.fp_mask = np.logical_and(np.logical_and(np.logical_not(self.gt_mask), pred_mask), self.not_ignore_mask)

        # Apply not_clicked_map early for effective calculation
        fn_mask_effective = np.logical_and(self.fn_mask, self.not_clicked_map)
        fp_mask_effective = np.logical_and(self.fp_mask, self.not_clicked_map)

        if not np.any(fn_mask_effective) and not np.any(fp_mask_effective):
            return None, {} # No more errors to click

        if padding:
            fn_mask_padded = np.pad(fn_mask_effective, ((1, 1), (1, 1)), 'constant')
            fp_mask_padded = np.pad(fp_mask_effective, ((1, 1), (1, 1)), 'constant')
        else:
            fn_mask_padded = fn_mask_effective
            fp_mask_padded = fp_mask_effective

        fn_mask_dt = cv2.distanceTransform(fn_mask_padded.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask_padded.astype(np.uint8), cv2.DIST_L2, 0)

        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        fn_max_dist = np.max(fn_mask_dt) if np.any(fn_mask_dt) else 0
        fp_max_dist = np.max(fp_mask_dt) if np.any(fp_mask_dt) else 0

        # Handle cases where no max_dist, return None
        if fn_max_dist == 0 and fp_max_dist == 0:
            return None, {}

        #pdb.set_trace()  # Debugging point to inspect fn_mask_dt, fp_mask_dt, etc.
        if alternate and len(self.clicks_list) > 0:
            is_positive = not self.clicks_list[-1].is_positive
        else:
            is_positive = fn_max_dist > fp_max_dist 
            
        if is_positive:
            coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)
            error_mask = fn_mask_effective.copy() # Use the effective mask for radius calc
            max_current_dist = fn_max_dist
            
            # For visualization
            error_type_str = "FN"
            current_dt_map = fn_mask_dt
        else:
            coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)
            error_mask = fp_mask_effective.copy() # Use the effective mask for radius calc
            max_current_dist = fp_max_dist
            
            # For visualization
            error_type_str = "FP"
            current_dt_map = fp_mask_dt
        
        #center = (coords_y[0], coords_x[0])
        #print(coords_x, coords_y)
        #center = (coords_y[-1], coords_x[-1]) # Default to first max_dist point
        
        # Use middle point of max distance locations as center
        middle_idx = len(coords_y) // 2
        center = (coords_y[middle_idx], coords_x[middle_idx]) 

        # Initialize radius and window_size
        radius = 0
        window_size = 0

        if self.adaptive_radius:
            # Scale window size based on error distance (clamp between 2 and 15)
            window_size = int(np.clip(max_current_dist * 2.0, 2, 15))
            # Make sure window_size is odd for symmetry (only if it starts above 1)
            if window_size % 2 == 0 and window_size > 1:
                window_size += 1
            if window_size == 0: window_size = 3 # Ensure minimum valid window size

            local_patch = get_local_error_region(error_mask, center, window_size=window_size)
            
            if not np.any(local_patch):
                radius = 2 # default to smallest if no error in patch
            else:
                radius = get_adaptive_radius(local_patch, min_radius=1, max_radius=5, scale_factor=0.2)        
        else:
            radius = 5  # default to fixed radius
            
        # Store data for external visualization
        click_data_for_vis = {
            'pred_mask': pred_mask, # Current prediction
            'gt_mask': self.gt_mask, # Ground truth
            'error_mask': error_mask, # The specific error mask (FN/FP effective)
            'dt_map': current_dt_map, # Distance transform map used
            'center': center, # Click coordinates
            'is_positive': is_positive, # Click type
            'radius': radius, # Adaptive radius
            'window_size': window_size, # Window size for local patch
            'max_dt': max_current_dist, # Max DT value
            'error_type_str': error_type_str, # "FN" or "FP"
        }
        # Create the Click object
        return Click(is_positive=is_positive, coords=center, radius=radius), click_data_for_vis
    
    
    def _make_next_g_cfr_click(self, seg_prob, edge_prob, click_radius=5, alpha=1.0,thr = [0.49,0.51]):
        """
        Guided CFR pseudo-click selection:
        - One click from segmentation gradients
        
        Args:
            seg_prob: HxW segmentation probabilities
            edge_prob: HxW edge prediction probabilities
            click_radius: pixel radius of click
            alpha: weight for Sobel vs edge in boundary-aware uncertainty
        Returns:
            clicks: list of Click objects
            boundary_uncertainty: HxW array
        """
        clicks = []

        # Compute boundary-aware uncertainty
        boundary_uncertainty = get_boundary_uncertainty_with_edges(seg_prob, edge_prob, alpha=1.0)

        # Click highest uncertainty from segmentation gradient (Sobel-dominant)
        sobel_grad = get_boundary_uncertainty_with_edges(seg_prob, edge_prob, alpha=alpha)
        idx_sobel = np.unravel_index(np.argmax(sobel_grad), sobel_grad.shape)
        prob_sobel = seg_prob[idx_sobel]

        # use threshold so clcik only when model is uncertain about that region (not confidently correct or wrong)
        if thr[0]< prob_sobel <= thr[1]:
            pass
        else:
            clicks.append(Click(is_positive=prob_sobel > thr[1], coords=idx_sobel, radius=click_radius))
    
        for nc in clicks:
            self.add_click(nc)
            
        return boundary_uncertainty, clicks
    
    
    def _reinforce_user_click(self, clicker, cfr_clicker, image_shape):
        """
        Adds 8 "helper" clicks around the last user click to reinforce their intent.
        
        Args:
            clicker (Clicker): The original clicker with the user's clicks.
            cfr_clicker (Clicker): The clicker for the cascade loop to add new clicks to.
            image_shape (tuple): The (height, width) of the image to check boundaries.
        """
        # Make sure there is at least one user click to reinforce
        if not clicker.clicks_list:
            return # Use 'return' to exit the function early

        image_height, image_width = image_shape
        last_user_click = clicker.clicks_list[-1]
        y, x = last_user_click.coords
        is_positive = last_user_click.is_positive
        
        # Use the same radius as the original click for the helper clicks
        #radius = last_user_click.radius
        radius = 5
        # Define the distance for the helper clicks
        distance = 1

        # Define all 8 directions (horizontal, vertical, and diagonal)
        offsets = [
            (-distance, 0), (distance, 0), (0, -distance), (0, distance),  # '+' shape
            (-distance, -distance), (-distance, distance),                 # 'X' shape
            (distance, -distance), (distance, distance)
        ]

        for dy, dx in offsets:
            # Calculate new coordinates
            new_coords = (y + dy, x + dx)
            
            # Check if the new coordinates are inside the image
            if 0 <= new_coords[0] < image_height and 0 <= new_coords[1] < image_width:
                new_click = Click(is_positive=is_positive, coords=new_coords, radius=radius)
                cfr_clicker.add_click(new_click)
    
    def _hybrid_score(self,prob_map, uncertainty_map):
        """
        Compute hybrid score = uncertainty * (1 - |prob - 0.5|)
        """
        return uncertainty_map * (1.0 - np.abs(prob_map - 0.5))

    
    def add_click(self, click):
        if click is None:
            return
        coords = click.coords

        click.indx = self.click_indx_offset + self.num_pos_clicks + self.num_neg_clicks
        if click.is_positive:
            self.num_pos_clicks += 1
        else:
            self.num_neg_clicks += 1

        self.clicks_list.append(click)
        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = False

    def _remove_last_click(self):
        click = self.clicks_list.pop()
        coords = click.coords

        if click.is_positive:
            self.num_pos_clicks -= 1
        else:
            self.num_neg_clicks -= 1

        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = True
    
    def reset_clicks(self):
        if self.gt_mask is not None:
            self.not_clicked_map = np.ones_like(self.gt_mask, dtype=np.bool)

        self.num_pos_clicks = 0
        self.num_neg_clicks = 0

        self.clicks_list = []

    def get_state(self):
        return deepcopy(self.clicks_list)

    def set_state(self, state):
        self.reset_clicks()
        for click in state:
            self.add_click(click)

    def __len__(self):
        return len(self.clicks_list)
    
class Click:
    def __init__(self, is_positive, coords, radius=5, indx=None):
        self.is_positive = is_positive
        self.coords = coords
        self.indx = indx
        self.radius = radius

    @property
    def coords_and_indx(self):
        return (*self.coords, self.indx)

    def copy(self, **kwargs):
        self_copy = deepcopy(self)
        for k, v in kwargs.items():
            setattr(self_copy, k, v)
        return self_copy
