import sys
import pickle
import argparse
from pathlib import Path
import cv2
import torch
import os
from tqdm import tqdm
import numpy as np
from scipy import ndimage
import shutil

sys.path.insert(0, '.')
from isegm.inference import utils
from isegm.utils.exp import load_config_file
from isegm.utils.vis import draw_probmap, draw_with_blend_and_clicks,draw_extremes
from isegm.inference.predictors import get_predictor
from isegm.inference.evaluation import evaluate_dataset
from isegm.model.modeling.pos_embed import interpolate_pos_embed_inference

import pdb

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['NoBRS', 'RGB-BRS', 'DistMap-BRS',
                                         'f-BRS-A', 'f-BRS-B', 'f-BRS-C'],
                        default='NoBRS',
                        help='')

    group_checkpoints = parser.add_mutually_exclusive_group(required=False) ## Default is true
    group_checkpoints.add_argument('--checkpoint', type=str,
                                   default='./weights/edge_model_50_epochs.pth',
                                   help='The path to the checkpoint. '
                                        'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
                                        'or an absolute path. The file extension can be omitted.')
    group_checkpoints.add_argument('--exp-path', type=str, default='',
                                   help='The relative path to the experiment with checkpoints.'
                                        '(relative to cfg.EXPS_PATH)')

    parser.add_argument('--datasets', type=str, default='GrabCut,Berkeley', #GrabCut,Berkeley,DAVIS,PascalVOC,COCO_MVal,SBD,OAIZIB',
                        help='List of datasets on which the model should be tested. '
                             'Datasets are separated by a comma. Possible choices: '
                             'GrabCut, Berkeley, DAVIS, SBD, PascalVOC')

    group_device = parser.add_mutually_exclusive_group()
    group_device.add_argument('--gpus', type=str, default='0',
                              help='ID of used GPU.')
    group_device.add_argument('--cpu', 
                              action='store_true',
                              #default=True,
                              help='Use only CPU for inference.')

    group_iou_thresh = parser.add_mutually_exclusive_group()
    group_iou_thresh.add_argument('--target-iou', type=float, default=0.95,
                                  help='Target IoU threshold for the NoC metric. (min possible value = 0.8)')
    group_iou_thresh.add_argument('--iou-analysis', action='store_true', default=False,
                                  help='Plot mIoU(number of clicks) with target_iou=1.0.')

    parser.add_argument('--n-clicks', type=int, default=20,
                        help='Maximum number of clicks for the NoC metric.')
    
    parser.add_argument('--adaptive-radius',action='store_true',
                        help='Whether to use adaptive raduis for disk maps')
    
    parser.add_argument('--min-n-clicks', type=int, default=1,
                        help='Minimum number of clicks for the evaluation.')
    parser.add_argument('--thresh', type=float, required=False, default=0.5,
                        help='The segmentation mask is obtained from the probability outputs using this threshold.')
    parser.add_argument('--clicks-limit', type=int, default=None)
    parser.add_argument('--eval-mode', type=str, default='cvpr',
                        help="Possible choices: cvpr, fixed<number>, or fixed<number>,<number>,(e.g. fixed400, fixed400,600).")

    parser.add_argument('--eval-ritm', action='store_true', default=False)
    parser.add_argument('--cf-n', default=0, type=int,
                        help='cascade-forward step')
    
    parser.add_argument('--cf-click', action='store_true',
                        help='cascade-forward clicks')
    parser.add_argument('--acf', action='store_true', default=False,
                        help='adaptive cascade-forward')
    parser.add_argument('--save-ious', action='store_true', default=False)
    parser.add_argument('--print-ious', action='store_true', default=False)
    parser.add_argument('--vis-preds', action='store_true', default=False)
    parser.add_argument('--model-name', type=str, default=None,
                        help='The model name that is used for making plots.')
    parser.add_argument('--config-path', type=str, default='./config.yml',
                        help='The path to the config file.')
    parser.add_argument('--logs-path', type=str, default='',
                        help='The path to the evaluation logs. Default path: cfg.EXPS_PATH.')

    args = parser.parse_args()
    if args.cpu:
        print('Using CPU for inference.')
        args.device = torch.device('cpu')
    else:
        print(f'Using GPUs: {args.gpus}')
        args.device = torch.device(f"cuda:{args.gpus.split(',')[0]}")

    if (args.iou_analysis or args.print_ious) and args.min_n_clicks <= 1:
        args.target_iou = 1.01
    else:
        args.target_iou = max(0.8, args.target_iou)

    cfg = load_config_file(args.config_path, return_edict=True)
    cfg.EXPS_PATH = Path(cfg.EXPS_PATH)

    if args.logs_path == '':
        args.logs_path = cfg.EXPS_PATH
    else:
        args.logs_path = Path(args.logs_path)

    return args, cfg

# Define area category thresholds for object size analysis
area_thrs = {
    'tiny': {'min_ratio': 0.0, 'max_ratio': 0.01},  # < 1%
    'small': {'min_ratio': 0.01, 'max_ratio': 0.05}, # >= 1% and < 5%
    'medium': {'min_ratio': 0.05, 'max_ratio': 0.20}, # >= 5% and < 20%
    'large': {'min_ratio': 0.20, 'max_ratio': 0.50}, # >= 20% and < 50%
    'very_large': {'min_ratio': 0.50, 'max_ratio': 1.0} # >= 50%
}

def main():
    args, cfg = parse_args()

    checkpoints_list, logs_path, logs_prefix = get_checkpoints_list_and_logs_path(args, cfg)
    logs_path.mkdir(parents=True, exist_ok=True)

    single_model_eval = len(checkpoints_list) == 1
    assert not args.iou_analysis if not single_model_eval else True, \
        "Can't perform IoU analysis for multiple checkpoints"
    print_header = True
    for dataset_name in args.datasets.split(','):
        #pdb.set_trace()
        dataset = utils.get_dataset(dataset_name, cfg)

        for checkpoint_path in checkpoints_list:
            print('Using checkpoint', checkpoint_path)
            model = utils.load_is_model(checkpoint_path, args.device, args.eval_ritm)

            predictor_params, zoomin_params = get_predictor_and_zoomin_params(args, dataset_name, eval_ritm=args.eval_ritm)

            # For SimpleClick models, we usually need to interpolate the positional embedding
            if not args.eval_ritm:
                interpolate_pos_embed_inference(model.backbone, zoomin_params['target_size'], args.device)

            predictor = get_predictor(model, args.mode, args.device,
                                      prob_thresh=args.thresh,
                                      predictor_params=predictor_params,
                                      zoom_in_params=zoomin_params)

            vis_callback = get_prediction_vis_callback(logs_path, dataset_name, args.thresh, args.n_clicks) if args.vis_preds else None
            #pdb.set_trace()
            dataset_results = evaluate_dataset(dataset, predictor, pred_thr=args.thresh,
                                               max_iou_thr=args.target_iou,
                                               min_clicks=args.min_n_clicks,
                                               max_clicks=args.n_clicks,
                                               adaptive_radius = args.adaptive_radius,
                                               callback=vis_callback)

            row_name = args.mode if single_model_eval else checkpoint_path.stem
            
            metric_types = ['ious', 'bious', 'hds','asds','nsds']
            for metric in metric_types:
                if args.iou_analysis:
                    save_iou_analysis_data(args, dataset_name,
                                           logs_path,
                                            logs_prefix, dataset_results,
                                            model_name=args.model_name,
                                            type = metric)

            for metric in metric_types:
                save_results(args, row_name, dataset_name, logs_path, logs_prefix, dataset_results, 
                            type=metric,
                            save_ious=single_model_eval and args.save_ious,
                            single_model_eval=single_model_eval,
                            print_header=print_header)
                
            # plot and save click progression
            save_click_progress(dataset_name,args, logs_path, dataset_results,
                                save_dir = "click_progress_plots_" + ("adaptive_r" if args.adaptive_radius else "constant_r"))
            
            #vis_gcfr(dataset_results, dataset_name, logs_path, args)
            
            print_header = False

    # # uncomment the following lines for memory analysis
    # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

def get_predictor_and_zoomin_params(args, dataset_name, apply_zoom_in=True, eval_ritm=False):
    predictor_params = {
        'cascade_step': args.cf_n + 1,
        'cascade_adaptive': args.acf,
        'cascade_clicks': args.cf_click,
        'adaptive_radius': args.adaptive_radius,
    }

    if args.clicks_limit is not None:
        if args.clicks_limit == -1:
            args.clicks_limit = args.n_clicks
        predictor_params['net_clicks_limit'] = args.clicks_limit

    zoom_in_params = None
    if apply_zoom_in and eval_ritm:
        if args.eval_mode == 'cvpr':
            zoom_in_params = {
                'target_size': 600 if dataset_name == 'DAVIS' else 400
            }
        elif args.eval_mode.startswith('fixed'):
            crop_size = int(args.eval_mode[5:])
            zoom_in_params = {
                'skip_clicks': -1,
                'target_size': (crop_size, crop_size)
            }
        else:
            raise NotImplementedError

    if apply_zoom_in and not eval_ritm:
        if args.eval_mode == 'cvpr':
            #pdb.set_trace()
            zoom_in_params = {
                'skip_clicks': -1,
                'target_size': (672, 672) if dataset_name == 'DAVIS' else (448, 448),
                'expansion_ratio': 1.6 if args.cf_click else 1.4
            }
        elif args.eval_mode.startswith('fixed'):
            crop_size = args.eval_mode.split(',')
            crop_size_h = int(crop_size[0][5:])
            crop_size_w = crop_size_h
            if len(crop_size) == 2:
                crop_size_w = int(crop_size[1])
            zoom_in_params = {
                'skip_clicks': -1,
                'target_size': (crop_size_h, crop_size_w)
            }
        else:
            raise NotImplementedError

    return predictor_params, zoom_in_params

def get_checkpoints_list_and_logs_path(args, cfg):
    logs_prefix = ''
    if args.exp_path:
        rel_exp_path = args.exp_path
        checkpoint_prefix = ''
        if ':' in rel_exp_path:
            rel_exp_path, checkpoint_prefix = rel_exp_path.split(':')

        exp_path_prefix = cfg.EXPS_PATH / rel_exp_path
        candidates = list(exp_path_prefix.parent.glob(exp_path_prefix.stem + '*'))
        assert len(candidates) == 1, "Invalid experiment path."
        exp_path = candidates[0]
        checkpoints_list = sorted((exp_path / 'checkpoints').glob(checkpoint_prefix + '*.pth'), reverse=True)
        assert len(checkpoints_list) > 0, "Couldn't find any checkpoints."

        if checkpoint_prefix:
            if len(checkpoints_list) == 1:
                logs_prefix = checkpoints_list[0].stem
            else:
                logs_prefix = f'all_{checkpoint_prefix}'
        else:
            logs_prefix = 'all_checkpoints'

        logs_path = args.logs_path / exp_path.relative_to(cfg.EXPS_PATH)
    else:
        checkpoints_list = [Path(utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, args.checkpoint))]
        logs_path = args.logs_path / 'others' / checkpoints_list[0].stem

    if args.cf_n > 0:
        cf_prefix = f'acf{args.cf_n}' if args.acf else f'cf{args.cf_n}'
        cf_prefix = f'{cf_prefix}_clk_{args.cf_click}' if args.cf_click else cf_prefix
        if logs_prefix:
            logs_prefix = '_'.join([cf_prefix, logs_prefix])
        else:
            logs_prefix = cf_prefix

    return checkpoints_list, logs_path, logs_prefix


def get_metric_data(metric_type: str, dataset_results: dict):
    """
    Extracts a specified metric, area ratios, clicks, predictions, and elapsed time
    from a dataset_results dictionary.

    Parameters:
        metric_type (str): One of ['ious', 'bious', 'hds', 'asds', 'nsds']
        dataset_results (dict): Dictionary containing 'click_list', 'pred_probs',
                                'metrics' (a list), and 'time'.

    Returns:
        tuple: (metric_values, area_ratios, click_list, pred_probs, elapsed_time)
    """
    
    # Define metric index mapping
    metric_map = {
        'ious': 0,
        'bious': 1,
        'hds': 2,
        'asds': 3,
        'nsds': 4
    }

    # Validate input dictionary structure
    required_keys = {'metrics', 'time'}
    missing_keys = required_keys - dataset_results.keys()
    if missing_keys:
        raise KeyError(f"Missing required keys in dataset_results: {missing_keys}")

    metrics = dataset_results['metrics']
    if not isinstance(metrics, list) or len(metrics) < 6:
        raise ValueError("The 'metrics' field must be a list of at least 6 elements.")

    if metric_type not in metric_map:
        raise ValueError(f"Invalid metric type '{metric_type}'. "
                         f"Expected one of: {list(metric_map.keys())}")

    # Extract components
    try:
        selected_metric = metrics[metric_map[metric_type]]
        area_ratios = metrics[5]
        return (
            selected_metric,
            area_ratios,
            dataset_results['time']
        )
    except Exception as e:
        raise RuntimeError(f"Failed to extract metric data: {e}")

def get_images_and_masks(dataset_results: dict,idx = None):
    """
    Return images, ground-truth masks, click lists, and prediction probabilities.

    Parameters
    ----------
    dataset_results : dict
        Must contain keys: 'image', 'gt_mask', 'click_list', 'pred_probs'.
        Each value may be a list/array or single item.
    
    idx : int, list of int, range, or None, optional
        • None   – return all items  
        • int    – return one item at that index  
        • list/range – return subset of items

    Returns
    -------
    tuple
        (images, gt_masks, click_list, pred_probs)
    """    

    required = {'image', 'gt_mask','metrics','click_list', 'pred_probs'}
    if missing := required - dataset_results.keys():
        raise KeyError(f"dataset_results missing keys: {missing}")

    images = dataset_results['image']
    masks  = dataset_results['gt_mask']
    clicks = dataset_results['click_list']
    pred_probs = dataset_results['pred_probs']
    ious  = dataset_results['metrics'][0]

    # -------- All --------
    if idx is None:
        return images, masks, ious, clicks, pred_probs

    # -------- Single index --------
    if isinstance(idx, int):
        if not (0 <= idx < len(images)):
            raise IndexError(f"Index {idx} out of range (0–{len(images) - 1})")
        return images[idx], masks[idx], ious[idx], clicks[idx], pred_probs[idx]

    # -------- Multiple indices --------
    if isinstance(idx, (list, range)):
        indices = list(idx)
        for i in indices:
            if not isinstance(i, int):
                raise TypeError("All indices must be integers.")
            if not (0 <= i < len(images)):
                raise IndexError(f"Index {i} out of range (0–{len(images) - 1})")
        return [images[i] for i in indices], [masks[i] for i in indices], [ious[i] for i in indices], [clicks[i] for i in indices], [pred_probs[i] for i in indices]

    # -------- Unsupported type --------
    raise TypeError(f"Unsupported index type: {type(idx)}")

def get_extreme_area_ratio_indices(area_ratios: list,low_n=5, mid_n=3, high_n=2):
    """
    Get indices for lowest, median, and highest area ratios.
    Parameters
    ----------
    area_ratios : list of area ratios
    low_n : int - Number of lowest area ratios to select.
    mid_n : int - Number of median area ratios to select.
    high_n : int - Number of highest area ratios to select.

    Returns
    -------
    List[int] : Combined list of indices: low_n lowest, mid_n around median, high_n highest
    """
    area_ratios_np = np.array(area_ratios)
    total = len(area_ratios_np)

    if total < (low_n + mid_n + high_n):
        raise ValueError(f"Not enough samples: required ≥ {low_n + mid_n + high_n}, got {total}")

    sorted_indices = np.argsort(area_ratios_np)

    # Lowest
    low_indices = sorted_indices[:low_n]

    # Middle (around median)
    mid_start = max((total // 2) - (mid_n // 2), 0)
    mid_end = mid_start + mid_n
    mid_indices = sorted_indices[mid_start:mid_end]

    # Highest
    high_indices = sorted_indices[-high_n:]

    # Combine and return as a list
    selected_indices =  [int(i) for i in list(low_indices) + list(mid_indices) + list(high_indices)]
    return selected_indices

def plot_click_progression(dataset_name, images, gt_masks, pred_probs, ious_list, clicks_list, indices, 
                           logs_path, iou_thr, save_dir='click_progress_plots',):
    """
        Plot click progression for each image.
        Parameters
        ----------
        dataset_name : str
        images : list of image paths
        gt_masks : list of ground truth masks
        pred_probs : list of predicted probabilities
        ious_list : list of IoU values
        clicks_list : list of clicks
        logs_path : str - Path to log file
        iou_thr : float - IoU threshold
        save_dir : str - Directory to save plots
        
        Returns
        -------
        saves visual progressions of segmentation masks with clicks.
    
    """
    
    assert len(images) == len(gt_masks) == len(pred_probs) == len(clicks_list)
    
    for i in tqdm(range(len(indices)), desc="Saving visual progressions"):
        #pdb.set_trace()
        image = images[i]
        gt_mask = gt_masks[i]
        gt_mask = draw_probmap(gt_mask)
        clicks = clicks_list[i]
        preds = pred_probs[i]
        ious = ious_list[i] if ious_list is not None else [None] * len(preds)
        
        s_dir = (logs_path / save_dir / dataset_name / f'sample_{indices[i]+1}')
        
        if s_dir.exists():
            shutil.rmtree(s_dir)
        s_dir.mkdir(parents=True)
        
        for step in range(len(preds)):
            prob_map = preds[step]
            pred_mask = (prob_map > iou_thr).astype(np.uint8)
            current_clicks = clicks[:step + 1]
            iou = ious[step] if step < len(ious) else None

            vis = draw_extremes(image, mask=pred_mask, clicks_list=current_clicks, iou=iou)
            prob_map_vis = draw_probmap(prob_map)
            
            out_path = os.path.join(s_dir, f"sample_{indices[i]+1}_click_{step+1}.png")
            
            #pdb.set_trace()
            combined_vis  = np.hstack((vis, prob_map_vis, np.where(gt_mask<255,150,255).astype(np.uint8)))
            cv2.imwrite(out_path, cv2.cvtColor(combined_vis, cv2.COLOR_RGB2BGR))
        
    print(f'click progress plots saved to { logs_path / save_dir / dataset_name}')
    with open(logs_path / save_dir / dataset_name / 'indices.txt', 'w') as f:
        f.write(','.join(map(str, np.array(indices)+1)))

def save_results(args, row_name, dataset_name, logs_path, logs_prefix, dataset_results, type,
                 save_ious=False, print_header=True, single_model_eval=False):
    all_ious, all_area_ratios, elapsed_time = get_metric_data(type,dataset_results)
    mean_spc, mean_spi = utils.get_time_metrics(all_ious, elapsed_time)

    #TODO: For 
    iou_thrs = np.arange(0.8, min(0.95, args.target_iou) + 0.001, 0.05).tolist()
    noc_list, noc_list_std, over_max_list = utils.compute_noc_metric(all_ious, iou_thrs=iou_thrs, max_clicks=args.n_clicks)

    row_name = 'last' if row_name == 'last_checkpoint' else row_name
    model_name = str(logs_path.relative_to(args.logs_path)) + ':' + logs_prefix if logs_prefix else logs_path.stem
    overall_header, overall_table_row = utils.get_results_table(noc_list, over_max_list, row_name, dataset_name,
                                                mean_spc, elapsed_time, args.n_clicks,
                                                model_name=model_name)

    if args.print_ious:
        min_num_clicks = min(len(x) for x in all_ious)
        mean_ious = np.array([x[:min_num_clicks] for x in all_ious]).mean(axis=0)
        miou_str = ' '.join([f'mIoU@{click_id}={mean_ious[click_id - 1].item():.2%};'
                             for click_id in [_ for _ in range(1, 21)] if click_id <= min_num_clicks])
        overall_table_row += '; ' + miou_str
    else:
        #pdb.set_trace()
        target_iou_int = int(args.target_iou * 100)
        if target_iou_int not in [80, 85, 90, 95]:
            noc_list, _, over_max_list = utils.compute_noc_metric(all_ious, iou_thrs=[args.target_iou],
                                                               max_clicks=args.n_clicks)
            overall_table_row += f' NoC@{args.target_iou:.1%} = {noc_list[0]:.2f};'
            overall_table_row += f' >={args.n_clicks}@{args.target_iou:.1%} = {over_max_list[0]}'

    if print_header and type=='ious':
        print(overall_header)
        print(overall_table_row)

    # --- NoC by Object Size Table Generation ---
    size_results = utils.compute_noc_by_size(all_ious, all_area_ratios, iou_thrs=iou_thrs,
                                             area_thrs = area_thrs,max_clicks=args.n_clicks)

     ## Use a dummy call to get the header structure (efficient enough as it's once)
    dummy_category_results = {'noc_list': [np.nan]*len(iou_thrs), 'over_max_list': [0]*len(iou_thrs), 'count':0}
    size_table_full_header, _ = utils.get_results_table_by_size("dummy", dummy_category_results, iou_thrs, args.n_clicks)
    
    ##Extract the width of the second line of the header for consistent row printing
    size_table_header_lines = size_table_full_header.split('\n')
    size_row_width = len(size_table_header_lines[1]) if len(size_table_header_lines) > 1 else 0

    ##Prepare the rows for each category
    category_rows_to_print = []
    for category in area_thrs.keys():
        results = size_results[category]
        _, category_table_row = utils.get_results_table_by_size(category, results, iou_thrs, args.n_clicks)
        category_rows_to_print.append(category_table_row)

    if print_header and type == 'ious':
        print(f"\n--- NoC Metrics by Object Size Category for Model: {model_name} ---")
        print(dataset_name) 
        print(size_table_full_header)
        for row in category_rows_to_print: # Iterate pre-calculated rows
            print(row)
        print('-' * size_row_width) 
    
    # File Saving Logic
    suffix = "adaptive_r" if args.adaptive_radius else "constant_r"
    log_dir = Path(logs_path) / f"mean_noc_table_{suffix}"
    log_dir.mkdir(parents=True, exist_ok=True)

    base_filename_prefix = ''
    if logs_prefix:
        base_filename_prefix = logs_prefix + '_'
    if not single_model_eval:
        base_filename_prefix += f'{dataset_name}_'

    overall_log_filename = f'{base_filename_prefix}{args.eval_mode}_{args.mode}_{args.n_clicks}_{type}.txt'
    overall_log_path = log_dir / overall_log_filename

    # Function to handle writing to log files
    def _write_to_log_file(path, header_content, row_content, mode='a'):
        if mode == 'a' and path.exists():
            # Append rows only, assume header exists
            with open(path, 'a') as f:
                f.write(row_content + '\n')
        else:
            # Write header and first row
            with open(path, 'w') as f:
                f.write(header_content + '\n')
                f.write(row_content + '\n')

    # Save Overall Results
    if type == 'ious':
        _write_to_log_file(overall_log_path, overall_header, overall_table_row, mode='a' if overall_log_path.exists() else 'w')

    # Save Size-Based Results to a separate file
    size_log_filename = f'{base_filename_prefix}{args.eval_mode}_{args.mode}_{args.n_clicks}_{type}_by_size.txt'
    size_log_path = log_dir / size_log_filename

    # Prepare the content for the size log file
    size_file_content_lines = []
    size_file_content_lines.append(f"Eval results by size for model: {model_name}")
    size_file_content_lines.append(dataset_name)
    size_file_content_lines.append(size_table_full_header)
    for row in category_rows_to_print:
        size_file_content_lines.append(row)
    size_file_content_lines.append('-' * size_row_width)

    # Write to the size log file
    if type == 'ious':
        size_file_mode = 'a' if size_log_path.exists() else 'w'
        with open(size_log_path, size_file_mode) as f:
            if size_file_mode == 'a':
                f.write('\n') # Add a newline to separate different runs if appending
            f.write('\n'.join(size_file_content_lines) + '\n')
        
    if save_ious: # This block was originally for pickle dumping all_ious
        ious_path = Path(logs_path) / 'noc_metrics' / type / (logs_prefix if logs_prefix else '')
        ious_path.mkdir(parents=True, exist_ok=True)
        with open(ious_path / f'{dataset_name}_{args.eval_mode}_{args.mode}_{args.n_clicks}_{type}.pkl', 'wb') as fp:
            pickle.dump(all_ious, fp)


def save_iou_analysis_data(args, dataset_name, logs_path, logs_prefix, dataset_results, model_name=None,type = 'ious'):
    #all_ious, all_bious, all_hd, _,_,_ = dataset_results
    all_ious, all_area_ratios, _ = get_metric_data(type,dataset_results)

    name_prefix = ''
    if logs_prefix:
        name_prefix = logs_prefix + '_'
    name_prefix += dataset_name + '_'
    if model_name is None:
        model_name = str(logs_path.relative_to(args.logs_path)) + ':' + logs_prefix if logs_prefix else logs_path.stem

    pkl_path = logs_path / f'metric_analysis/{type}/{name_prefix}{args.eval_mode}_{args.mode}_{args.n_clicks}_{type}.pickle'
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open('wb') as f:
        pickle.dump({
            'dataset_name': dataset_name,
            'model_name': f'{model_name}_{args.mode}',
            'all_ious': all_ious,
            'all_area_ratios': all_area_ratios
        }, f)


def save_click_progress(dataset_name, args, logs_path, dataset_results,save_dir,type = 'ious'):
    _, all_area_ratios, _ = get_metric_data(type,dataset_results)
    indices = get_extreme_area_ratio_indices(all_area_ratios, low_n=5, mid_n=3, high_n=2)
    images, gt_masks, ious, clicks_list, pred_probs = get_images_and_masks(dataset_results, idx= indices)
    plot_click_progression(dataset_name,images, gt_masks, pred_probs, 
                           ious, clicks_list, indices,
                           logs_path,iou_thr = args.thresh,
                           save_dir=save_dir)


def get_prediction_vis_callback(logs_path, dataset_name, prob_thresh, max_clicks):
    save_path = logs_path / 'predictions_vis' / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)

    cache = {}

    def callback(image, gt_mask, pred_probs, iou,
                 sample_id, click_indx, clicks_list, success,
                 zoom_in):

        if cache.get('sample_id') != sample_id or cache.get('click_indx', -1) > click_indx:
            # move to next sample
            cache['sample_id'] = sample_id
            cache['plot'] = None
            cache['iou'] = 0
            cache['click_indx'] = -1

        cache['iou'] = max(iou, cache['iou'])
        cache['click_indx'] = click_indx

        sample_path = save_path / f'{sample_id}.jpg'

        rmin, rmax, cmin, cmax = zoom_in._object_roi

        pred_map = pred_probs > prob_thresh
        prob_map = draw_probmap(pred_probs)[..., ::-1]
        image_with_mask = draw_with_blend_and_clicks(image, pred_map, clicks_list=clicks_list)

        image_with_mask = cv2.putText(image_with_mask, f'clk={click_indx}', (0, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 
                                      1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(image_with_mask, (cmin, rmin), (cmax, rmax), (0, 0, 255), 2)

        error_map = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        error_map[(gt_mask > 0) & ~pred_map] = (255, 0, 0)  # under-segm. fn
        error_map[(gt_mask < 1) & pred_map] = (0, 0, 255)  # over-segm. fp
        error_map = cv2.putText(error_map, f'iou={iou:.4}', (0, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 255, 0), 2, cv2.LINE_AA)

        gt_map = gt_mask[..., None].astype(np.uint8)
        gt_map = np.repeat(gt_map, 3, axis=2) * 255

        row1 = np.concatenate((image_with_mask, gt_map), axis=1)
        row2 = np.concatenate((prob_map, error_map), axis=1)
        plot = np.concatenate((row1, row2), axis=1)

        if cache.get('plot', None) is not None:
            plot = np.concatenate((cache['plot'], plot), axis=0)

        cache['plot'] = plot

        if click_indx + 1 == max_clicks and cache['iou'] <= 0.9:
            cv2.imwrite(str(sample_path), plot)

    return callback


# In your scripts/evaluate_model_test.py, after you run evaluate_dataset

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

def visualize_all_g_cfr_steps(image, gt_mask, final_prediction, g_cfr_debug, click_list, thresh, output_dir, sample_idx):
    """
    Visualizes each step of the G-CFR process.
    Creates a multi-panel static image or an animated GIF for each sample.
    """
    
    #pdb.set_trace()
    uncertainty_maps = g_cfr_debug['uncertainty_maps_per_step']
    pseudo_clicks = g_cfr_debug['pseudo_clicks_per_step']
    user_initial_pred = g_cfr_debug['intial_predictions_per_step']

    num_steps = len(uncertainty_maps)
    
    if num_steps == 0:
        print(f"No G-CFR steps to visualize for sample {sample_idx}.")
        return

    num_cols = 5 # For (Uncertainty Map, Refined Prediction) per step
    num_rows = num_steps + 1 # For (Original, Initial Pred) + each step's visualization

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
    
    # Flatten axes for easier iteration if it's not 2D
    if num_rows == 1 or num_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.ravel() # Flatten to 1D for easy iteration

    current_ax_idx = 0

    # Row 0: Original Image and Initial Prediction (before any pseudo-clicks)
    axes[current_ax_idx].imshow(image)
    axes[current_ax_idx].set_title(f'Original Image (Sample {sample_idx})')
    axes[current_ax_idx].axis('off')
    current_ax_idx += 1

    axes[current_ax_idx].imshow(gt_mask>.5, cmap='gray')
    axes[current_ax_idx].set_title('Ground truth')
    axes[current_ax_idx].axis('off')
    current_ax_idx += 1
    
    
    dt = ndimage.distance_transform_edt(gt_mask>0.5)
    edges = np.logical_and(dt <= 1, gt_mask)
    edges = (edges > 0).astype(np.float32)

    axes[current_ax_idx].imshow(edges , cmap='gray')
    axes[current_ax_idx].set_title('Ground truth edge')
    axes[current_ax_idx].axis('off')
    current_ax_idx += 1

    
    user_init=axes[current_ax_idx].imshow(user_initial_pred[0][0], cmap='jet',vmin=0,vmax=1)
    axes[current_ax_idx].set_title('Initial Prediction (User Clicks Only)')
    axes[current_ax_idx].axis('off')
    fig.colorbar(user_init, ax=axes[current_ax_idx],fraction=0.026, pad=0.04)
    current_ax_idx += 1
    
    user_init=axes[current_ax_idx].imshow(1.0 - np.abs(2.0 * user_initial_pred[0][0] - 1.0), cmap='viridis',vmin=0,vmax=1)
    axes[current_ax_idx].set_title('Model Uncertainty (User Clicks Only)')
    axes[current_ax_idx].axis('off')
    fig.colorbar(user_init, ax=axes[current_ax_idx],fraction=0.026, pad=0.04)
    
    current_ax_idx += 1

    refined_predictions_per_step = g_cfr_debug['refined_predictions_per_step'] 
    #pdb.set_trace()
    for step_idx in range(num_steps):
        current_uncertainty_map = uncertainty_maps[step_idx]
        current_pseudo_clicks = pseudo_clicks[step_idx]
        current_refined_prediction = refined_predictions_per_step[-step_idx+1]
        current_user_initial_pred = user_initial_pred[step_idx]

        # Convert uncertainty map to heatmap
        if np.max(current_uncertainty_map) > 0:
            uncertainty_heatmap =  (current_uncertainty_map / np.max(current_uncertainty_map) * 255).astype(np.uint8)
            uncertainty_heatmap = cv2.applyColorMap(uncertainty_heatmap, cv2.COLORMAP_VIRIDIS)
        else:
            uncertainty_heatmap = np.zeros_like(image, dtype=np.uint8)

        # Draw the pseudo-click on the heatmap
        vis_uncertainty = uncertainty_heatmap.copy()
        #pdb.set_trace()
        clicks_to_draw = current_pseudo_clicks if isinstance(current_pseudo_clicks, list) else [current_pseudo_clicks]
        for click in clicks_to_draw:
            if click is None:
                continue 
            y, x = click.coords
            # Green for positive (hole-filling), Red for negative (blob-removal/boundary)
            color = (0, 255, 0) if click.is_positive else (0, 0, 255) 
            
            # Draw a small white outline to make the click more visible
            #cv2.circle(vis_uncertainty, (x, y), click.radius, (255, 255, 255), -1)
            cv2.circle(vis_uncertainty, (x, y), click.radius, color, -1)
        
        # Panel 1: Uncertainty Map + Click
        uncertainty = axes[current_ax_idx].imshow(vis_uncertainty, vmin=0,vmax=1)
        axes[current_ax_idx].set_title(f'Step {step_idx+1}: Uncertainty + Click')
        axes[current_ax_idx].axis('off')
        fig.colorbar(uncertainty, ax=axes[current_ax_idx],fraction=0.026, pad=0.04)
        current_ax_idx += 1
        
        axes[current_ax_idx].imshow(current_user_initial_pred[0]> thresh, cmap='gray')
        axes[current_ax_idx].set_title('Curent Prediction (User Clicks Only)')
        axes[current_ax_idx].axis('off')
        current_ax_idx += 1
        
        axes[current_ax_idx].imshow(current_user_initial_pred[1]> (thresh-.15), cmap='gray')
        axes[current_ax_idx].set_title('Curent Prediction (User Clicks Only)')
        axes[current_ax_idx].axis('off')
        current_ax_idx += 1
        
        # Panel 2: Refined Prediction
        axes[current_ax_idx].imshow(current_refined_prediction>thresh, cmap='gray')
        axes[current_ax_idx].set_title(f'Step {step_idx+1}: Refined Prediction')
        axes[current_ax_idx].axis('off')
        current_ax_idx += 1
        
        axes[current_ax_idx].imshow(1.0 - np.abs(2.0 * current_refined_prediction - 1.0),  cmap='jet',vmin=0,vmax=1)
        axes[current_ax_idx].set_title('Model Uncertainty (User + Pseudo Clicks)')
        axes[current_ax_idx].axis('off')
        fig.colorbar(uncertainty, ax=axes[current_ax_idx],fraction=0.026, pad=0.04)
        current_ax_idx += 1

    plt.tight_layout()
    plt.savefig(output_dir / f"sample_{sample_idx:03d}_cfr_steps.png", bbox_inches='tight',dpi = 500)
    plt.close(fig)


def vis_gcfr(dataset_results, dataset_name,logs_path, args, save_dir = 'g_cfr_all_steps'):
    # Create a directory for the visualizations
    s_dir = (logs_path / save_dir / dataset_name )
    thresh = args.thresh
        
    if s_dir.exists():
        shutil.rmtree(s_dir)
    s_dir.mkdir(parents=True)
    
    # Loop through the results and save a visualization for each sample
    
    
    for i in tqdm(range(len(dataset_results['image'])), desc="Saving G-CFR progressions"):
        image = dataset_results['image'][i]
        gt_mask = dataset_results['gt_mask'][i]
        final_pred = dataset_results['pred_probs'][i] 
        g_cfr_info = dataset_results['g_cfr_info'][i]
        click_list = dataset_results['click_list'][i]
        
        # This function needs the initial prediction too, not just the final one
        # And the full list of predictions per step if available
        visualize_all_g_cfr_steps(image, gt_mask, final_pred, g_cfr_info, click_list, thresh, s_dir, i)

    print(f"Saved G-CFR step visualizations to {s_dir}")


if __name__ == '__main__':
    main()