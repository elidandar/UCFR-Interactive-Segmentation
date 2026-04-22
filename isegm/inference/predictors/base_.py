import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from isegm.inference.transforms import AddHorizontalFlip, SigmoidForPred, LimitLongestSide


class BasePredictor(object):
    def __init__(self, model, device,
                 net_clicks_limit=None,
                 with_flip=False,
                 zoom_in=None,
                 max_size=None,
                 cascade_step=0,
                 cascade_adaptive=False,
                 cascade_clicks=1,
                 **kwargs):
        self.with_flip = with_flip
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.device = device
        self.zoom_in = zoom_in
        self.prev_prediction = None
        self.model_indx = 0
        self.click_models = None
        self.net_state_dict = None
        self.cascade_step = cascade_step
        self.cascade_adaptive = cascade_adaptive
        self.cascade_clicks = cascade_clicks

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

    def set_input_image(self, image):
        image_nd = self.to_tensor(image)
        for transform in self.transforms:
            transform.reset()
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)
        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])
        self.prev_prediction_edges = torch.zeros_like(self.original_image[:, :1, :, :])

    def get_prediction(self, clicker, prev_mask=None, on_cascade=False):
        clicks_list = clicker.get_clicks()

        if len(clicks_list) <= self.cascade_clicks and self.cascade_step > 0 and not on_cascade:
            for i in range(self.cascade_step):
                prediction_mask = self.get_prediction(clicker, None, True)
                if self.cascade_adaptive and prev_mask is not None:
                    diff_num = (
                        (prediction_mask > 0.49) != (prev_mask > 0.49)
                    ).sum()
                    if diff_num <= 20:
                        return prediction_mask
                prev_mask = prediction_mask
            return prediction_mask

        if self.click_models is not None:
            model_indx = min(clicker.click_indx_offset + len(clicks_list), len(self.click_models)) - 1
            if model_indx != self.model_indx:
                self.model_indx = model_indx
                self.net = self.click_models[model_indx]

        input_image = self.original_image
        #print(f'Printing image shape from get_prediction {input_image.shape}')
        if prev_mask is None:
            prev_mask = self.prev_prediction
        if hasattr(self.net, 'with_prev_mask') and self.net.with_prev_mask:
    
            ## Calculate image gradients
            image_grad = self.gradient_image(input_image, method='sobel')
            # Conver to channel, W, H and to torch tensor
            image_grad = torch.from_numpy(image_grad.astype(np.float32)).unsqueeze(0).unsqueeze(0).contiguous().to(self.device)
            #print(f'Printing image gradient shape from get_prediction {image_grad.shape}')
                    
            input_image = torch.cat((input_image, prev_mask, image_grad), dim=1)
            
        image_nd, clicks_lists, is_image_changed = self.apply_transforms(
            input_image, [clicks_list]
        )
        #print(f'Printing image concat(image,prev_mask,image_grad) shape from get_prediction {image_nd.shape}')
        pred_logits = self._get_prediction(image_nd, clicks_lists, is_image_changed)
        prediction_mask = F.interpolate(pred_logits['pred_mask'], mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])
        
        prediction_edges = F.interpolate(pred_logits['pred_edges'], mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])
        
        for t in reversed(self.transforms):
            prediction_mask = t.inv_transform(prediction_mask)
            prediction_edges = t.inv_transform(prediction_edges)
            

        if self.zoom_in is not None and self.zoom_in.check_possible_recalculation():
            return self.get_prediction(clicker)

        self.prev_prediction = prediction_mask
        self.prev_prediction_edges = prediction_edges

        #prediction_edges
        
        return prediction_mask.cpu().numpy()[0, 0]

    def _get_prediction(self, image_nd, clicks_lists, is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)
        #print(f'Printing image shape from _get_prediction {image_nd.shape}')
        
        with torch.no_grad():
            self.net.eval()
            with torch.cuda.amp.autocast():
                output = self.net(image_nd, points_nd)
       
        return {
            'pred_mask': output['instances'],
            'pred_edges': output['edges']
        }

    def _get_transform_states(self):
        return [x.get_state() for x in self.transforms]

    def _set_transform_states(self, states):
        assert len(states) == len(self.transforms)
        for state, transform in zip(states, self.transforms):
            transform.set_state(state)

    def apply_transforms(self, image_nd, clicks_lists):
        is_image_changed = False
        for t in self.transforms:
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            is_image_changed |= t.image_changed
        #print(f'Printing from apply transform: {image_nd.shape}')

        return image_nd, clicks_lists, is_image_changed
    

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
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
    
     
    def gradient_image(self, image_input, output_path='gradient.png', method='sobel'):
        """
        Computes the gradient of an image using the specified method.

        Parameters:
        - image_input: torch.Tensor or np.ndarray
            The input image as a PyTorch tensor (C, H, W) or a NumPy array (H, W, C).
        - output_path: str
            The file path to save the gradient image.
        - method: str
            The gradient computation method: 'laplacian', 'sobel', or 'prewitt'.

        Returns:
        - img_grad: np.ndarray
            The computed gradient image as a NumPy array.
        """
        try:
            # Handle input type: PyTorch tensor or NumPy array
            if isinstance(image_input, torch.Tensor):
                # Convert PyTorch tensor to NumPy array
                image_np = image_input.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            elif isinstance(image_input, np.ndarray):
                # Ensure the NumPy array is in uint8 format
                if image_input.dtype != np.uint8:
                    image_np = image_input.astype(np.uint8)
                else:
                    image_np = image_input
            else:
                raise ValueError("Input must be a PyTorch tensor or a NumPy array.")

            # Ensure the image is 3-channel
            if image_np.ndim != 3 or image_np.shape[2] != 3:
                raise ValueError("Input image must have 3 channels (H, W, C).")

            # Split channels
            img_r = image_np[:, :, 0]
            img_g = image_np[:, :, 1]
            img_b = image_np[:, :, 2]

            # Gradient computation
            if method == 'laplacian':
                grad_r = cv2.Laplacian(img_r, cv2.CV_64F)
                grad_g = cv2.Laplacian(img_g, cv2.CV_64F)
                grad_b = cv2.Laplacian(img_b, cv2.CV_64F)
            elif method == 'sobel':
                # https://github.com/liewjunhao/thin-object-selection/blob/main/dataloaders/helpers.py#L65
                grad_r = np.sqrt(cv2.Sobel(img_r, cv2.CV_64F, 1, 0)**2 + cv2.Sobel(img_r, cv2.CV_64F, 0, 1)**2)
                grad_g = np.sqrt(cv2.Sobel(img_g, cv2.CV_64F, 1, 0)**2 + cv2.Sobel(img_g, cv2.CV_64F, 0, 1)**2)
                grad_b = np.sqrt(cv2.Sobel(img_b, cv2.CV_64F, 1, 0)**2 + cv2.Sobel(img_b, cv2.CV_64F, 0, 1)**2)
            else:
                raise ValueError("Unsupported method. Choose from 'laplacian' or 'sobel.")

            # Compute the combined gradient magnitude
            gradient = np.sqrt(grad_r**2 + grad_g**2 + grad_b**2)

            # Normalize to [0, 1]
            img_grad = (gradient - gradient.min()) / ((gradient.max() - gradient.min()) + 1e-12)

            # Save the gradient image (scale back to [0, 255] for saving)
            # success = cv2.imwrite(output_path, (img_grad * 255).astype(np.uint8))
            # if success:
            #     print(f"Gradient image saved to {output_path}")
            # else:
            #     print("Failed to save the gradient image.")

            return img_grad

        except Exception as e:
            print(f"An error occurred: {e}")

    def get_states(self):
        return {
            'transform_states': self._get_transform_states(),
            'prev_prediction': self.prev_prediction.clone()
        }

    def set_states(self, states):
        self._set_transform_states(states['transform_states'])
        self.prev_prediction = states['prev_prediction']
