from time import time

import numpy as np
import torch
import cv2
import copy
import json

from isegm.inference import utils
from isegm.inference.clicker import Clicker, Click
#from isegm.inference.clicker_visualizer import ClickerVisualizer
import pdb

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm

def evaluate_dataset(dataset, predictor, **kwargs):
    all_ious, all_bious, all_hd, all_asd, all_nsd, all_area_ratios = [], [], [], [], [], []
    all_clicks, all_pred_probs = [], []
    all_imgs, all_gt_masks = [],[]
    all_g_cfr_debug_info = []

    start_time = time()
    
    dataset_clicks = []
    
    # Load the pre-recorded clicks
    try:
        with open("recorded_clicks.json", "r") as f:
            recorded_clicks = json.load(f)
    except FileNotFoundError:
        print("\nNo pre-recorded clicks found. Starting fresh.")
        recorded_clicks = None
    #len(dataset)
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)
        image_area = sample.image.shape[0] * sample.image.shape[1]  # Total image area
        #pdb.set_trace()
        recorded_clicks_for_sample = recorded_clicks[index] if recorded_clicks is not None else None    

        for object_id in sample.objects_ids:
            
            gt_mask = sample.gt_mask(object_id)
            object_area = np.sum(gt_mask)  # Count of foreground pixels
            area_ratio = object_area / image_area  # Object area relative to image size
            all_area_ratios.append(area_ratio)  # Store area ratio
            
            ## Get metric results
            results = evaluate_sample(sample.image, gt_mask, predictor,
                                                sample_id=index,replay_clicks=recorded_clicks_for_sample, **kwargs)
            (
                clicks, pred_probs, sample_ious, sample_bious, sample_hd, 
                sample_asd, sample_nsd, _ , clicks_for_image,g_cfr_debug_info
                ) = results
            
            all_g_cfr_debug_info.append(g_cfr_debug_info)
            
            dataset_clicks.append(clicks_for_image)
            
            # All all metrics
            all_ious.append(sample_ious)
            all_bious.append(sample_bious)
            all_hd.append(sample_hd)
            all_asd.append(sample_asd)
            all_nsd.append(sample_nsd)
            
            # append
            all_clicks.append(clicks)
            all_pred_probs.append(pred_probs)
            all_imgs.append(sample.image)
            all_gt_masks.append(gt_mask)
            
            #print(f"Sample {index}\nIOU: {sample_ious}\nBIOU: {sample_bious} \nHD: {sample_hd} \nASD: {sample_asd}\nNSD: {sample_nsd}\n")
    
    
    #print(f"Dataset stored clicks {dataset_clicks}")
    
    # Save the recorded clicks to a file
    # with open("recorded_clicks.json", "w") as f:
    #     json.dump(dataset_clicks, f)
    # print("\nSuccessfully recorded and saved click sequences.")
    
    end_time = time()
    elapsed_time = end_time - start_time
    return {
        'image': all_imgs,
        'gt_mask': all_gt_masks,
        'click_list': all_clicks,
        'pred_probs': all_pred_probs,
        'metrics': [all_ious, all_bious, all_hd, all_asd, all_nsd, all_area_ratios],
        'g_cfr_info': all_g_cfr_debug_info,
        'time': elapsed_time
    }
    
def evaluate_sample(image, gt_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    adaptive_radius = False,sample_id=None, callback=None, replay_clicks=None):
    
    predictor.reset()
    clicker = Clicker(gt_mask=gt_mask,adaptive_radius=adaptive_radius)
    
    # Initialize Visualizer for EACH image
    # visualizer = ClickerVisualizer(output_dir='./save',image_id=sample_id,
    #                                   original_image=cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    pred_mask = np.zeros_like(gt_mask)
    pred_prob_list= []
    
    ious_list, bious_list, hd_list, asd_list, nsd_list = [], [], [], [], []

    with torch.no_grad():
        predictor.set_input_image(image)
         
        default_radius = copy.deepcopy(predictor.net.dist_maps.norm_radius)
        predictor.net.dist_maps.small_radius = None
        #predictor.net.dist_maps.annulus_weight = 0.5
        #predictor.net.dist_maps.annulus_weight = 0.5
        
        
        #pdb.set_trace()
        click_iterator = replay_clicks if replay_clicks is not None else range(max_clicks)
        
        clicks_for_image = []
        
        #for click_indx in range(max_clicks):
        for click_indx, click_data in enumerate(click_iterator):
            
            _, vis_data  = clicker.make_next_click(pred_mask)
            
            #pdb.set_trace()
            ## TODO: To change radius of disk map
            if adaptive_radius and (click_indx > 100): # 3
                predictor.net.dist_maps.small_radius = clicker.clicks_list[-1].radius
                
                if replay_clicks is not None:
                    clicker._remove_last_click()
                    #clicker.click_indx_offset = click_indx
                    replayed_click = Click(is_positive=replay_clicks[click_indx]['is_positive'],
                                        coords=replay_clicks[click_indx]['coords'],
                                        radius=clicker.clicks_list[-1].radius) # We can handle adaptive radius later if needed
                    clicker.add_click(replayed_click)  
                
            else:
                predictor.net.dist_maps.norm_radius = default_radius
                clicker.clicks_list[-1].radius = default_radius
           
            #pdb.set_trace()
                click_info = {
                    "is_positive": bool(clicker.clicks_list[-1].is_positive),
                    "coords": [int(clicker.clicks_list[-1].coords[0]), int(clicker.clicks_list[-1].coords[1])] # Ensure coords are JSON serializable
                }
                
                clicks_for_image.append(click_info)
            
            #pdb.set_trace()
            pred_probs, g_cfr_debug_info = predictor.get_prediction(clicker)
            
            # --- Visualize the click step ---
            #visualizer.visualize_click_step(clicker.click_counter, vis_data) # Use clicker.click_counter for global click ID
            
            pred_mask = pred_probs > pred_thr
            ## calculate mask iou
            iou = utils.get_iou(gt_mask, pred_mask)
            
            ## get edge from mask and calculate biou
            gt_edge = utils.mask_to_boundary(gt_mask,dilation_ratio = 0.01)
            pred_edge = utils.mask_to_boundary(pred_mask.astype(np.int32()),dilation_ratio = 0.01)
            #utils.plot_edge(gt_edge,pred_edge,is_save = True)
            biou = utils.get_iou(gt_edge, pred_edge)
            
            ious_list.append(iou)
            bious_list.append(biou)
            
            ## Compute distance
            metrics = utils.compute_segmentation_distance_metrics(pred_mask, gt_mask)

            # Append results
            pred_prob_list.append(pred_probs.copy())
            hd_list.append(metrics["hd"])
            asd_list.append(metrics["asd"])
            nsd_list.append(metrics["nsd"])
            
            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                if callback is not None: ## vis callback
                    callback(image, gt_mask, pred_probs, iou,
                            sample_id, click_indx, clicker.clicks_list, True,
                            predictor.zoom_in)
                break

            if callback is not None: ## vis callback
                callback(image, gt_mask, pred_probs, iou,
                         sample_id, click_indx,
                         clicker.clicks_list, False,
                         predictor.zoom_in)
                
        # reassign default radius
        predictor.net.dist_maps.norm_radius = copy.deepcopy(default_radius)
        
        
        return (
        np.array(clicker.clicks_list),
        np.array(pred_prob_list),
        np.array(ious_list, dtype=np.float32), 
        np.array(bious_list, dtype=np.float32),
        np.array([hd.cpu().numpy() if isinstance(hd, torch.Tensor) else hd for hd in hd_list], dtype=np.float32),
        np.array([asd.cpu().numpy() if isinstance(asd, torch.Tensor) else asd for asd in asd_list], dtype=np.float32),
        np.array([nsd.cpu().numpy() if isinstance(nsd, torch.Tensor) else nsd for nsd in nsd_list], dtype=np.float32),
        #np.array(nsd_list, dtype=np.float32),
        pred_probs, clicks_for_image,g_cfr_debug_info)
        
    
