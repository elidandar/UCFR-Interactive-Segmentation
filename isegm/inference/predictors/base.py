import torch
import torch.nn.functional as F
from torchvision import transforms
from isegm.inference.transforms import AddHorizontalFlip, SigmoidForPred, LimitLongestSide
from isegm.inference import utils
from isegm.inference.transforms import ZoomIn
from copy import deepcopy
import pdb

class BasePredictor(object):
    def __init__(self, model, device,
                 net_clicks_limit=None,
                 with_flip=False,
                 zoom_in=None,
                 max_size=None,
                 cascade_step=0,
                 cascade_adaptive=False,
                 cascade_clicks=False,
                 adaptive_radius=False,
                 **kwargs):
        self.with_flip = with_flip
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.device = device
        self.zoom_in = zoom_in
        self.prev_prediction = None
        self.prev_prediction_edges = None
        self.model_indx = 0
        self.click_models = None
        self.net_state_dict = None
        self.cascade_step = cascade_step
        self.cascade_adaptive = cascade_adaptive
        self.cascade_clicks = cascade_clicks
        self.adaptive_radius = adaptive_radius
        
        self.save_prev_mask = []
        self.g_cfr_debug_info = {
            'uncertainty_maps_per_step': [],
            'pseudo_clicks_per_step': [],
            'refined_predictions_per_step': [],
            'intial_predictions_per_step': []
        }
        
        self.save_prev_mask= []

        if isinstance(model, tuple):
            self.net, self.click_models = model
        else:
            self.net = model

        self.to_tensor = transforms.ToTensor()

        self.transforms = [zoom_in] if zoom_in is not None else []
        if max_size is not None:
            self.transforms.append(LimitLongestSide(max_size=max_size))
        self.transforms.append(SigmoidForPred())
        if with_flip:
            self.transforms.append(AddHorizontalFlip())
            
        self.on_cascade = False  # Flag to indicate if we are in a cascade loop


    def reset(self):
        """Clears the internal state of the predictor."""
        self.save_prev_mask = []
        self.g_cfr_debug_info = {
            'uncertainty_maps_per_step': [],
            'pseudo_clicks_per_step': [],
            'refined_predictions_per_step': [],
            'intial_predictions_per_step': []
        }

    
    def set_input_image(self, image):
        image_nd = self.to_tensor(image)
        for transform in self.transforms:
            transform.reset()
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)
        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])
        self.prev_prediction_edges = torch.zeros_like(self.original_image[:, :1, :, :])
        self.aux_fn = torch.zeros_like(self.original_image[:, :1, :, :])
        self.aux_fn = torch.zeros_like(self.original_image[:, :1, :, :])
        

    # ---------------------------
    # Main Prediction Function
    # ---------------------------
    def get_prediction(self, clicker, prev_mask=None, on_cascade=False):
        clicks_list = clicker.get_clicks()

        # Guided CFR block
        if self.cascade_step > 1 and not on_cascade:
            prediction, debug_info = self._guided_cfr_refinement(clicker, prev_mask)
            return prediction, debug_info

        # Handle model selection for progressive clicks
        if self.click_models is not None:
            model_indx = min(clicker.click_indx_offset + len(clicks_list), len(self.click_models)) - 1
            if model_indx != self.model_indx:
                self.model_indx = model_indx
                self.net = self.click_models[model_indx]

        # Prepare input
        input_image = self.original_image
        if prev_mask is None:
            prev_mask = self.prev_prediction
         
        if hasattr(self.net, 'with_prev_mask'):
            input_image = torch.cat((input_image, prev_mask), dim=1)

        # Apply transforms
        image_nd, clicks_lists, is_image_changed = self.apply_transforms(input_image, [clicks_list],on_cascade=self.on_cascade)

        # Get logits from network
        pred_logits = self._get_prediction(image_nd, clicks_lists, is_image_changed)

        # Finalize prediction (resize, inverse transforms, aux handling, caching)
        prediction, aux_fn, aux_fp, prediction_edges = self._finalize_prediction(pred_logits, image_nd)

        # Zoom-in refinement loop
        if self.zoom_in is not None and self.zoom_in.check_possible_recalculation():
            self.zoom_in.reset()  
            return self.get_prediction(clicker)

        # Save state for next iteration
        self.prev_prediction = prediction
        self.prev_prediction_edges = prediction_edges
        self.aux_fn = aux_fn
        self.aux_fp = aux_fp

        #pdb.set_trace()
        return prediction.cpu().numpy()[0, 0], self.g_cfr_debug_info

    # ---------------------------
    # Guided CFR Refinement
    # ---------------------------
    def _guided_cfr_refinement(self, clicker, prev_mask):
        cfr_clicker = deepcopy(clicker)
        for step in range(self.cascade_step):
            #pdb.set_trace()
            if self.cascade_clicks:
                if step > 0:
                    print('Using U-CFR Click')
                    #self.on_cascade = True
                    if step == 1:
                        cfr_clicker._reinforce_user_click(clicker, cfr_clicker, self.original_image.shape[2:])
                    
                    # default threshold is 0.49, 0.51 but can be changed to be more or less strict about when to add CFR clicks 
                    # (e.g., [0.4, 0.6] would add clicks in a wider range of uncertainty, while [0.45, 0.55] 
                    # would be more conservative and only add clicks when the model is very uncertain)
                    uncertainty_map, new_clicks = cfr_clicker._make_next_g_cfr_click(self.prev_prediction.squeeze().cpu().numpy(),
                                                           self.prev_prediction_edges.squeeze().cpu().numpy(),
                                                           click_radius=5, alpha=1.0,
                                                           thr=[0.49, 0.51])
                    
                    #pdb.set_trace()
                    if new_clicks:  # only add if confident
                        # Store debug info
                        self.g_cfr_debug_info['pseudo_clicks_per_step'].append(new_clicks)
                        self.g_cfr_debug_info['uncertainty_maps_per_step'].append(uncertainty_map)
                        self.g_cfr_debug_info['intial_predictions_per_step'].append(
                            [self.prev_prediction.squeeze().cpu().numpy().copy(),
                             self.prev_prediction_edges.squeeze().cpu().numpy().copy()]
                        )
                    
                        if self.adaptive_radius:
                            self.net.dist_maps.small_radius = cfr_clicker.clicks_list[-1].radius

          
            # Recursive refinement
            prediction, _ = self.get_prediction(cfr_clicker, None, on_cascade=True)

            self.g_cfr_debug_info['refined_predictions_per_step'].append(prediction)

            # Convergence check (stability)
            if self.cascade_adaptive and prev_mask is not None:
                current_mask_binary = (prediction > 0.49)
                prev_mask_binary = (prev_mask > 0.49)
                iou_stability = utils.get_iou(current_mask_binary, prev_mask_binary)

                if iou_stability > getattr(self, "stability_threshold", 0.99):
                    return prediction, self.g_cfr_debug_info

            prev_mask = prediction

        return prediction, self.g_cfr_debug_info

    # ---------------------------
    # Finalize Prediction
    # ---------------------------
    def _finalize_prediction(self, pred_logits, image_nd):
        # Resize logits to input size
        prediction = F.interpolate(pred_logits['instances'], mode='bilinear',
                                align_corners=True, size=image_nd.size()[2:])
        prediction_edges = F.interpolate(pred_logits['edges'], mode='bilinear',
                                        align_corners=True, size=image_nd.size()[2:])

        # Aux handling
        if 'aux' in pred_logits:
            aux = F.interpolate(pred_logits['aux'], mode='bilinear',
                                align_corners=True, size=image_nd.size()[2:])
            aux_fn, aux_fp = aux[:, 1:, :, :], aux[:, :1, :, :]
        else:
            aux_fn, aux_fp = None, None

        # Inverse transforms
        for t in reversed(self.transforms):
            prediction = t.inv_transform(prediction)
            prediction_edges = t.inv_transform(prediction_edges)
            
            # if aux_fn is not None: aux_fn = t.inv_transform(aux_fn)
            # if aux_fp is not None: aux_fp = t.inv_transform(aux_fp)

        return prediction, aux_fn, aux_fp, prediction_edges

    
    # ---------------------------
    #  Prediction
    # ---------------------------
    def _get_prediction(self, image_nd, clicks_lists, is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)
        
        #pdb.set_trace()
        output = self.net(image_nd, points_nd)#, small_radius_override=None)
        
        result = {
            'instances': output['instances'],
            'edges': output['edges']
        }

        # Check if 'instances_aux' exists in the output and is not None,
        if 'instances_aux' in output and output['instances_aux'] is not None:
            result['aux'] = output['instances_aux']
        
        return result
    
    def _get_transform_states(self):
        return [x.get_state() for x in self.transforms]

    def _set_transform_states(self, states):
        assert len(states) == len(self.transforms)
        for state, transform in zip(states, self.transforms):
            transform.set_state(state)

    def apply_transforms(self, image_nd, clicks_lists, on_cascade=False):
        is_image_changed = False
        #pdb.set_trace()
        for t in self.transforms:
            # Check if the current transform 't' is a ZoomIn object
            if isinstance(t, ZoomIn) and on_cascade:
                # If it is, call its transform method with the extra 'on_cascade' flag
                image_nd, clicks_lists = t.transform(image_nd, clicks_lists, on_cascade)
            else:
                # For all other transforms, call the standard transform method
                image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            
        return image_nd, clicks_lists, is_image_changed
    

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        #pdb.set_trace()
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

            neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)
    

    def get_states(self):
        return {
            'transform_states': self._get_transform_states(),
            'prev_prediction': self.prev_prediction.clone()
        }

    def set_states(self, states):
        self._set_transform_states(states['transform_states'])
        self.prev_prediction = states['prev_prediction']
        
