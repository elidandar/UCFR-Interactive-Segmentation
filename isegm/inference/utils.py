from datetime import timedelta
from pathlib import Path
import torch
import cv2
import numpy as np
from scipy import ndimage

from isegm.data.datasets import GrabCutDataset, BerkeleyDataset, DavisDataset, \
    SBDEvaluationDataset, PascalVocDataset, BraTSDataset, ssTEMDataset, OAIZIBDataset, HARDDataset
from isegm.utils.serialization import load_model


def get_time_metrics(all_ious, elapsed_time):
    n_images = len(all_ious)
    n_clicks = sum(map(len, all_ious))

    mean_spc = elapsed_time / n_clicks
    mean_spi = elapsed_time / n_images

    return mean_spc, mean_spi


def load_is_model(checkpoint, device, eval_ritm, **kwargs):
    if isinstance(checkpoint, (str, Path)):
        state_dict = torch.load(checkpoint, map_location='cpu')
        # print("Load pre-trained checkpoint from: %s" % checkpoint)
    else:
        state_dict = checkpoint

    if isinstance(state_dict, list):
        model = load_single_is_model(state_dict[0], device, eval_ritm, **kwargs)
        models = [load_single_is_model(x, device, eval_ritm, **kwargs) for x in state_dict]

        return model, models
    else:
        return load_single_is_model(state_dict, device, eval_ritm, **kwargs)


# def load_single_is_model(state_dict, device, eval_ritm, **kwargs):
#     model = load_model(state_dict['config'], eval_ritm, **kwargs)
#     model.load_state_dict(state_dict['state_dict'], strict=True)

#     for param in model.parameters():
#         param.requires_grad = False
#     model.to(device)
#     model.eval()

#     return model


def load_single_is_model(state_dict, device, eval_ritm, **kwargs):
    model = load_model(state_dict['config'], eval_ritm, **kwargs)
    try:
        model.load_state_dict(state_dict['state_dict'], strict=True)
    except RuntimeError as e:
        print(f"Warning: {e}\nRetrying with strict=False.")
        model.load_state_dict(state_dict['state_dict'], strict=False)
        
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()

    return model


def get_dataset(dataset_name, cfg):
    if dataset_name == 'GrabCut':
        dataset = GrabCutDataset(cfg.GRABCUT_PATH)
    elif dataset_name == 'Berkeley':
        dataset = BerkeleyDataset(cfg.BERKELEY_PATH)
    elif dataset_name == 'DAVIS':
        dataset = DavisDataset(cfg.DAVIS_PATH)
    elif dataset_name == 'SBD':
        dataset = SBDEvaluationDataset(cfg.SBD_PATH)
    elif dataset_name == 'SBD_Train':
        dataset = SBDEvaluationDataset(cfg.SBD_PATH, split='train')
    elif dataset_name == 'PascalVOC':
        dataset = PascalVocDataset(cfg.PASCALVOC_PATH, split='val')
    elif dataset_name == 'COCO_MVal':
        dataset = DavisDataset(cfg.COCO_MVAL_PATH)
    elif dataset_name == 'BraTS':
        dataset = BraTSDataset(cfg.BraTS_PATH)
    elif dataset_name == 'ssTEM':
        dataset = ssTEMDataset(cfg.ssTEM_PATH)
    elif dataset_name == 'OAIZIB':
        dataset = OAIZIBDataset(cfg.OAIZIB_PATH)
    elif dataset_name == 'HARD':
        dataset = HARDDataset(cfg.HARD_PATH)
    else:
        dataset = None

    return dataset


def get_iou(gt_mask, pred_mask, ignore_label=-1):
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    intersection = np.logical_and(np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    union = np.logical_and(np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()

    return intersection / union


def compute_noc_metric(all_ious, iou_thrs, max_clicks=20):
    def _get_noc(iou_arr, iou_thr):
        vals = iou_arr >= iou_thr
        return np.argmax(vals) + 1 if np.any(vals) else max_clicks

    noc_list = []
    noc_list_std = []
    over_max_list = []
    for iou_thr in iou_thrs:
        scores_arr = np.array([_get_noc(iou_arr, iou_thr)
                               for iou_arr in all_ious], dtype=np.int)

        score = scores_arr.mean()
        score_std = scores_arr.std()
        over_max = (scores_arr == max_clicks).sum()

        noc_list.append(score)
        noc_list_std.append(score_std)
        over_max_list.append(over_max)

    return noc_list, noc_list_std, over_max_list


def get_area_category(area_ratio,AREA_THRESHOLDS=None):
    for category, thresholds in AREA_THRESHOLDS.items():
        if thresholds['min_ratio'] <= area_ratio < thresholds['max_ratio']:
            return category
    return None # Should not happen if thresholds cover all ranges

def compute_noc_by_size(all_ious, all_area_ratios, iou_thrs, area_thrs, max_clicks=20):
    """
    Computes NoC metrics categorized by object size.

    Parameters:
    -----------
    all_ious : list of np.ndarray
        List where each element is an array of IoU values for a single instance
        across multiple clicks.
    all_area_ratios : list of float
        List where each element is the area ratio (object area / image area)
        for the corresponding instance in all_ious.
    iou_thrs : list of float
        List of IoU thresholds to compute NoC for (e.g., [0.8, 0.85, 0.9, 0.95]).
    max_clicks : int, optional
        Maximum number of clicks allowed. Defaults to 20.

    Returns:
    --------
    dict: A dictionary where keys are area categories ('tiny', 'small', etc.)
          and values are dictionaries containing 'noc_list', 'noc_list_std', 'over_max_list'
          for that category.
    """
    if len(all_ious) != len(all_area_ratios):
        raise ValueError("Lengths of all_ious and all_area_ratios must match.")

    # Group instances by size category
    categorized_ious = {category: [] for category in area_thrs.keys()}
    
    for i, iou_arr in enumerate(all_ious):
        area_ratio = all_area_ratios[i]
        category = get_area_category(area_ratio,area_thrs)
        if category:
            categorized_ious[category].append(iou_arr)
        # else: handle edge cases if any area_ratio doesn't fit

    results_by_category = {}

    # Helper function for _get_noc (same as your existing one)
    def _get_noc(iou_arr, iou_thr):
        vals = iou_arr >= iou_thr
        return np.argmax(vals) + 1 if np.any(vals) else max_clicks

    # Compute NoC metrics for each category
    for category, ious_in_category in categorized_ious.items():
        if not ious_in_category:
            # No instances for this category, populate with zeros or NaNs
            results_by_category[category] = {
                'noc_list': [np.nan] * len(iou_thrs),
                'noc_list_std': [np.nan] * len(iou_thrs),
                'over_max_list': [0] * len(iou_thrs),
                'count': 0
            }
            continue
        
        noc_list = []
        noc_list_std = []
        over_max_list = []

        for iou_thr in iou_thrs:
            scores_arr = np.array([_get_noc(iou_arr, iou_thr)
                                   for iou_arr in ious_in_category], dtype=np.int32) # Use int32 for numpy version compatibility

            score = scores_arr.mean()
            score_std = scores_arr.std()
            over_max = (scores_arr == max_clicks).sum()

            noc_list.append(score)
            noc_list_std.append(score_std)
            over_max_list.append(over_max)
        
        results_by_category[category] = {
            'noc_list': noc_list,
            'noc_list_std': noc_list_std,
            'over_max_list': over_max_list,
            'count': len(ious_in_category) # Number of instances in this category
        }
    
    return results_by_category


def get_results_table_by_size(category_name, category_results, iou_thrs, n_clicks):
    """
    Generates the header and a formatted table row for a single object size category.

    Parameters:
    -----------
    category_name : str
        The name of the object size category (e.g., 'tiny', 'medium').
    category_results : dict
        A dictionary containing 'noc_list', 'noc_list_std', 'over_max_list', 'count'
        for this specific category, as returned by compute_noc_by_size.
    iou_thrs : list of float
        List of IoU thresholds (e.g., [0.8, 0.85, 0.9, 0.95]).
    n_clicks : int
        Maximum number of clicks used for evaluation (for >=N@X% columns).

    Returns:
    --------
    tuple: (header_string, row_string)
        header_string: The formatted table header for size-based results.
        row_string: The formatted table row for the given category's results.
    """
    # Extract results for clarity
    noc_list = category_results['noc_list']
    over_max_list = category_results['over_max_list']
    count = category_results['count']

    # Build dynamic header for IoU thresholds
    iou_header_cols = ''.join([f'{"NoC@" + str(int(thr * 100)) + "%":^9}|' for thr in iou_thrs])
    over_max_header_cols = ''.join([f'{">="+str(n_clicks)+"@" + str(int(thr * 100)) + "%":^9}|' for thr in iou_thrs if thr >= 0.85]) # Assuming >=85% as in original table

    # Construct the header string
    table_header = (f'|{"Category":^10}|{"Count":^7}|'
                    f'{iou_header_cols}'
                    f'{over_max_header_cols}')
    row_width = len(table_header) # For separator lines

    header_line = '-' * row_width
    header_full = f'{header_line}\n{table_header}\n{header_line}'

    # Construct the table row string
    table_row = f'|{category_name:<10}|{count:<7}|'

    # Add NoC values
    for i, noc_val in enumerate(noc_list):
        if not np.isnan(noc_val):
            table_row += f'{noc_val:^9.2f}|'
        else:
            table_row += f'{"N/A":^9}|'

    #iou_thrs is [0.8, 0.85, 0.9, 0.95], then over_max_list[1] is for 85%, [2] for 90%, [3] for 95%
    
    if len(over_max_list) > 1: # Check if at least 85% is in the list
        table_row += f'{over_max_list[1]:^9}|'
    else: table_row += f'{"N/A":^9}|' # Fallback for 85%
    
    if len(over_max_list) > 2: # Check if at least 90% is in the list
        table_row += f'{over_max_list[2]:^9}|'
    else: table_row += f'{"N/A":^9}|' # Fallback for 90%

    if len(over_max_list) > 3: # Check if at least 95% is in the list
        table_row += f'{over_max_list[3]:^9}|'
    else: table_row += f'{"N/A":^9}|' # Fallback for 95%

    return header_full, table_row


def find_checkpoint(weights_folder, checkpoint_name):
    weights_folder = Path(weights_folder)
    if ':' in checkpoint_name:
        model_name, checkpoint_name = checkpoint_name.split(':')
        models_candidates = [x for x in weights_folder.glob(f'{model_name}*') if x.is_dir()]
        assert len(models_candidates) == 1
        model_folder = models_candidates[0]
    else:
        model_folder = weights_folder

    if checkpoint_name.endswith('.pth'):
        if Path(checkpoint_name).exists():
            checkpoint_path = checkpoint_name
        else:
            checkpoint_path = weights_folder / checkpoint_name
    else:
        model_checkpoints = list(model_folder.rglob(f'{checkpoint_name}*.pth'))
        assert len(model_checkpoints) == 1
        checkpoint_path = model_checkpoints[0]

    return str(checkpoint_path)

def get_results_table(noc_list, over_max_list, brs_type, dataset_name, mean_spc, elapsed_time,
                      n_clicks=20, model_name=None):
    table_header = (f'|{"BRS Type":^13}|{"Dataset":^11}|'
                    f'{"NoC@80%":^9}|{"NoC@85%":^9}|{"NoC@90%":^9}|{"NoC@95%":^9}|'
                    f'{">="+str(n_clicks)+"@85%":^9}|{">="+str(n_clicks)+"@90%":^9}|{">="+str(n_clicks)+"@95%":^9}|'
                    f'{"SPC,s":^7}|{"Time":^9}|')
    row_width = len(table_header)

    header = f'Eval results for model: {model_name}\n' if model_name is not None else ''
    header += '-' * row_width + '\n'
    header += table_header + '\n' + '-' * row_width

    eval_time = str(timedelta(seconds=int(elapsed_time)))
    table_row = f'|{brs_type:^13}|{dataset_name:^11}|'
    table_row += f'{noc_list[0]:^9.2f}|'
    table_row += f'{noc_list[1]:^9.2f}|' if len(noc_list) > 1 else f'{"?":^9}|'
    table_row += f'{noc_list[2]:^9.2f}|' if len(noc_list) > 2 else f'{"?":^9}|'
    table_row += f'{noc_list[3]:^9.2f}|' if len(noc_list) > 3 else f'{"?":^9}|'
    table_row += f'{over_max_list[1]:^9}|' if len(noc_list) > 1 else f'{"?":^9}|'
    table_row += f'{over_max_list[2]:^9}|' if len(noc_list) > 2 else f'{"?":^9}|'
    table_row += f'{over_max_list[3]:^9}|' if len(noc_list) > 3 else f'{"?":^9}|'
    table_row += f'{mean_spc:^7.3f}|{eval_time:^9}|'

    return header, table_row

def extract_edge(mask, method='dist_transf'):
    """ Extract the object boundaries."""
    if mask.max() == 0:
        return np.zeros_like(mask)
    elif method == 'dist_transf':
        dt = ndimage.distance_transform_edt(mask)
        edge = np.logical_and(dt <= 1, mask)
    else:
        raise NotImplementedError
    return (edge > 0).astype(np.float32) 


# General util function to get the boundary of a binary mask.
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask.astype(np.uint8), kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode
    
    

import numpy as np
import cv2

def save_edge_comparison(gt_mask, pred_mask, dilation_ratio=0.01, is_edges=False, save_path="edge_comparison.png"):
    """
    Saves an overlay of GT and Pred edges with different colors for False Positives and False Negatives, 
    and adds text to define the colors.
    
    Args:
        gt_mask (np.array): Ground truth mask (binary, HxW).
        pred_mask (np.array): Predicted mask (binary, HxW).
        is_edges (bool): If True, assumes input is already edge-detected; otherwise, computes edges.
        save_path (str): Path to save the output image.
    """
    # Convert masks to boundaries (edges) if not already processed
    if not is_edges:
        gt_edge = mask_to_boundary(gt_mask, dilation_ratio=dilation_ratio)
        pred_edge = mask_to_boundary(pred_mask.astype(np.int32), dilation_ratio=dilation_ratio)
    else:
        gt_edge = gt_mask
        pred_edge = pred_mask.astype(np.int32)

    # Debugging: Check if edge masks are non-empty
    #print(f"GT Edges sum: {np.sum(gt_edge)}, Pred Edges sum: {np.sum(pred_edge)}")

    if np.sum(gt_edge) == 0 and np.sum(pred_edge) == 0:
        print("Warning: Both GT and Pred edges are empty. Skipping save.")
        return

    # Convert to boolean masks
    #gt_edge = gt_edge.astype(bool)
    #pred_edge = pred_edge.astype(bool)

    # Define different error regions
    tp = gt_edge & pred_edge  # True Positives (Green)
    fp = pred_edge & ~gt_edge  # False Positives (Red)
    fn = gt_edge & ~pred_edge  # False Negatives (Blue)

    # Debugging: Check for error types
    print(f"TP: {np.sum(tp)}, FP: {np.sum(fp)}, FN: {np.sum(fn)}")

    # Create an RGB visualization mask
    edge_map = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    edge_map[tp] = [0, 255, 0]   # Green for correct edges
    edge_map[fp] = [255, 0, 0]   # Red for false positives
    edge_map[fn] = [0, 0, 255]   # Blue for false negatives

    # Add text to define the colors
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (255, 255, 255)  # White color for the text
    thickness = 2
    text = "Green:True +ve, Blue:False +ve, Red:False -ve"
    
    # Add background for text to improve visibility
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = 10
    text_y = 30  # Start placing the text at the top of the image

    # Ensure the text box does not cover the image content
    if text_y + text_size[1] > gt_mask.shape[0]:
        text_y = gt_mask.shape[0] - text_size[1] - 10  # If it overlaps, place it at the bottom

    # Draw a filled rectangle as background for text
    cv2.rectangle(edge_map, (text_x - 5, text_y - 5), (text_x + text_size[0] + 5, text_y + text_size[1] + 5), (0, 0, 0), -1)
    
    # Place the text on the image
    cv2.putText(edge_map, text, (text_x, text_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

    # Debugging: Ensure the visualization map has valid pixel values
    print(f"Max pixel value in edge_map: {edge_map.max()}")

    if np.any(edge_map > 0):
        # Ensure correct color format for OpenCV (BGR)
        #edge_map = cv2.cvtColor(edge_map, cv2.COLOR_RGB2BGR)
        
        # Save the image
        success = cv2.imwrite(save_path, edge_map)
        if success:
            print(f"Edge comparison saved at {save_path}")
        else:
            print("Error: Failed to save image.")
    else:
        print("Warning: No edges detected. Image not saved.")



def one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    """
    For every value v in `labels`, the value in the output will be either 1 or 0. Each vector along the `dim`-th
    dimension has the "one-hot" format, i.e., it has a total length of `num_classes`,
    with a one and `num_class-1` zeros.
    Note that this will include the background label, thus a binary mask should be treated as having two classes.

    Args:
        labels: input tensor of integers to be converted into the 'one-hot' format. Internally `labels` will be
            converted into integers `labels.long()`.
        num_classes: number of output channels, the corresponding length of `labels[dim]` will be converted to
            `num_classes` from `1`.
        dtype: the data type of the output one_hot label.
        dim: the dimension to be converted to `num_classes` channels from `1` channel, should be non-negative number.

    Example:

    For a tensor `labels` of dimensions [B]1[spatial_dims], return a tensor of dimensions `[B]N[spatial_dims]`
    when `num_classes=N` number of classes and `dim=1`.

    .. code-block:: python

        from monai.networks.utils import one_hot
        import torch

        a = torch.randint(0, 2, size=(1, 2, 2, 2))
        out = one_hot(a, num_classes=2, dim=0)
        print(out.shape)  # torch.Size([2, 2, 2, 2]) 

        a = torch.randint(0, 2, size=(2, 1, 2, 2, 2))
        out = one_hot(a, num_classes=2, dim=1)
        print(out.shape)  # torch.Size([2, 2, 2, 2, 2])

    """

    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels



# from torchmetrics.segmentation import HausdorffDistance

# def get_hausdorff_distance(pred,gt, distance_metric="euclidean",num_classes=2):
#     """
#     Calculate the Hausdorff distance between two binary masks.
#     Args:
#         pred (torch.Tensor): Predicted binary mask.
#         gt (torch.Tensor): Ground truth binary mask.
#     Returns:
#         hd (float): Hausdorff distance between the two masks.
#     """
#     # Convert binary masks to one-hot format
#     pred = one_hot(torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).int(), num_classes=2, dim=1).int()
#     gt = one_hot(torch.from_numpy(gt).clamp(0, 1).unsqueeze(0).unsqueeze(0), num_classes=2, dim=1).int()
    
#     # Calculate Hausdorff distance
#     hausdorff_distance = HausdorffDistance(distance_metric=distance_metric, num_classes=num_classes)
#     hd = hausdorff_distance(pred, gt)
    
#     return hd


from collections.abc import Sequence
from typing import Union
from typing import Union, Sequence
import numpy as np
import monai.metrics

def get_hausdorff_distance(
    y_pred, 
    y, 
    include_background: bool = False,
    distance_metric: str = "euclidean",
    percentile: Union[float, None] = None, 
    directed: bool = False,
    spacing: Union[
        int, float, np.ndarray, Sequence[Union[int, float, np.ndarray, Sequence[Union[int, float]]]], None
    ] = None,
):
    
    ## Convert from numpy to torch tensors
    y_pred = torch.from_numpy(y_pred).unsqueeze(0).unsqueeze(0) ## HW-->11HW
    y = torch.from_numpy(y).unsqueeze(0).unsqueeze(0) ## HW-->11HW
    
    hd = monai.metrics.compute_hausdorff_distance(
        y_pred,
        y,
        include_background=include_background,
        distance_metric=distance_metric,
        percentile=percentile,
        directed=directed,
        spacing=spacing
    )
    
    return hd


def get_average_surface_distance(
    y_pred, 
    y, 
    include_background: bool = False,
    distance_metric: str = "euclidean",
    symmetric: bool = False,
    spacing: Union[
        int, float, np.ndarray, Sequence[Union[int, float, np.ndarray, Sequence[Union[int, float]]]], None
    ] = None,
):
    
    ## Convert from numpy to torch tensors
    y_pred = torch.from_numpy(y_pred).unsqueeze(0).unsqueeze(0) ## HW-->11HW
    y = torch.from_numpy(y).unsqueeze(0).unsqueeze(0) ## HW-->11HW
    
    asd = monai.metrics.compute_average_surface_distance(
        y_pred,
        y,
        include_background=include_background,
        distance_metric=distance_metric,
        symmetric=symmetric,
        spacing=spacing
    )
    
    return asd


def get_normalised_surface_dice(
    y_pred, 
    y, 
    class_thresholds: list[float],
    include_background: bool = False,
    distance_metric: str = "euclidean",
    use_subvoxels: bool = False,
    spacing: Union[
        int, float, np.ndarray, Sequence[Union[int, float, np.ndarray, Sequence[Union[int, float]]]], None
    ] = None,
):
    
    ## Convert from numpy to torch tensors
    y_pred = torch.from_numpy(y_pred).unsqueeze(0).unsqueeze(0) ## HW-->11HW
    y = torch.from_numpy(y).unsqueeze(0).unsqueeze(0) ## HW-->11HW
    
    asd = monai.metrics.compute_surface_dice(
        y_pred,
        y,
        class_thresholds=class_thresholds,
        include_background=include_background,
        distance_metric=distance_metric,
        use_subvoxels=use_subvoxels,
        spacing=spacing
    )
    
    return asd

def compute_segmentation_distance_metrics(pred_mask, gt_mask, class_thresgolds = [1]):
    """Compute Hausdorff Distance, Average Surface Distance, and Normalized Surface Dice safely."""
    metrics = {"hd": None, "asd": None, "nsd": None}
    
    try:
        metrics["hd"] = get_hausdorff_distance(pred_mask, gt_mask)
    except Exception as e:
        print(f"[Warning] Hausdorff Distance computation failed: {e}")

    try:
        metrics["asd"] = get_average_surface_distance(pred_mask, gt_mask)
    except Exception as e:
        print(f"[Warning] Average Surface Distance computation failed: {e}")

    try:
        metrics["nsd"] = get_normalised_surface_dice(pred_mask, gt_mask, class_thresholds=class_thresgolds)
    except Exception as e:
        print(f"[Warning] Normalized Surface Dice computation failed: {e}")
    
    return metrics

    
