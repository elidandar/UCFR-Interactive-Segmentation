import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from scipy.ndimage import distance_transform_edt
from isegm.utils import misc 

# can find here: https://github.com/CoinCheung/pytorch-loss/blob/af876e43218694dc8599cc4711d9a5c5e043b1b2/label_smooth.py
#from .label_sm import LabelSmoothSoftmaxCEV1 as LSSCE
from torchvision import transforms
from functools import partial
from operator import itemgetter
import monai.losses

class NormalizedFocalLossSigmoid(nn.Module):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, max_mult=-1, eps=1e-12,
                 from_sigmoid=False, detach_delimeter=True,
                 batch_axis=0, weight=None, size_average=True,
                 ignore_label=-1):
        super(NormalizedFocalLossSigmoid, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._from_logits = from_sigmoid
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._max_mult = max_mult
        self._k_sum = 0
        self._m_max = 0

    def forward(self, pred, label):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = torch.sigmoid(pred)

        alpha = torch.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = torch.where(sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred))

        beta = (1 - pt) ** self._gamma

        sw_sum = torch.sum(sample_weight, dim=(-2, -1), keepdim=True)
        beta_sum = torch.sum(beta, dim=(-2, -1), keepdim=True)
        mult = sw_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = beta * mult
        if self._max_mult > 0:
            beta = torch.clamp_max(beta, self._max_mult)

        with torch.no_grad():
            ignore_area = torch.sum(label == self._ignore_label, dim=tuple(range(1, label.dim()))).cpu().numpy()
            sample_mult = torch.mean(mult, dim=tuple(range(1, mult.dim()))).cpu().numpy()
            if np.any(ignore_area == 0):
                self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult[ignore_area == 0].mean()

                beta_pmax, _ = torch.flatten(beta, start_dim=1).max(dim=1)
                beta_pmax = beta_pmax.mean().item()
                self._m_max = 0.8 * self._m_max + 0.2 * beta_pmax

        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            bsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(sample_weight.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (bsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return loss

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)
        sw.add_scalar(tag=name + '_m', value=self._m_max, global_step=global_step)


class FocalLoss(nn.Module):
    def __init__(self, axis=-1, alpha=0.25, gamma=2,
                 from_logits=False, batch_axis=0,
                 weight=None, num_class=None,
                 eps=1e-9, size_average=True, scale=1.0,
                 ignore_label=-1):
        super(FocalLoss, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._scale = scale
        self._num_class = num_class
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average

    def forward(self, pred, label, sample_weight=None):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = torch.sigmoid(pred)

        alpha = torch.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = torch.where(sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred))

        beta = (1 - pt) ** self._gamma

        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            tsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(label.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (tsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return self._scale * loss


class SoftIoU(nn.Module):
    def __init__(self, from_sigmoid=False, ignore_label=-1):
        super().__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label

    def forward(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label

        if not self._from_sigmoid:
            pred = torch.sigmoid(pred)

        loss = 1.0 - torch.sum(pred * label * sample_weight, dim=(1, 2, 3)) \
            / (torch.sum(torch.max(pred, label) * sample_weight, dim=(1, 2, 3)) + 1e-8)

        return loss


class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, ignore_label=-1):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

    def forward(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label
        label = torch.where(sample_weight, label, torch.zeros_like(label))

        if not self._from_sigmoid:
            loss = torch.relu(pred) - pred * label + F.softplus(-torch.abs(pred))
        else:
            eps = 1e-12
            loss = -(torch.log(pred + eps) * label
                     + torch.log(1. - pred + eps) * (1. - label))

        loss = self._weight * (loss * sample_weight)
        return torch.mean(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))


class BoundaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gt: torch.Tensor, label: torch.Tensor):
        # finding positive/negative boundaries
        gt_arr = (gt.detach().cpu().numpy()[:, 0, :, :] > 0.5).astype(np.uint8)

        dts_pos = []
        dts_neg = []

        for bindex in range(len(gt)):
            gt_mask_pos = gt_arr[bindex]
            gt_mask_neg = (gt_arr[bindex] == 0).astype(np.uint8)

            dt_pos = cv2.distanceTransform(gt_mask_pos, cv2.DIST_L1, 3) == 1
            dt_neg = cv2.distanceTransform(gt_mask_neg, cv2.DIST_L1, 3) == 1

            dts_pos.append([dt_pos])
            dts_neg.append([dt_neg])

        dts_pos = torch.tensor(np.array(dts_pos), device=gt.device)
        dts_neg = torch.tensor(np.array(dts_neg), device=gt.device)
        # end finding

        size_average = np.prod(gt.size())

        loss_pos = F.binary_cross_entropy_with_logits(
            label[dts_pos], gt[dts_pos], reduction='sum'
        ) / size_average
        loss_neg = F.binary_cross_entropy_with_logits(
            label[dts_neg], gt[dts_neg], reduction='sum'
        ) / size_average

        return loss_pos + loss_neg


class ErrorCount(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gt: torch.Tensor, label: torch.Tensor):
        size = np.prod(gt.size())
        diff_map = gt != (torch.sigmoid(label) > 0.49)
        diff_map = diff_map.sum() / size
        return diff_map


class SigmoidBinaryDiceLoss(nn.Module):
    """ Dice Loss for edge detection
    """
    def __init__(self, from_sigmoid=False, use_reciprocal=False, batch_axis=0):
        super(SigmoidBinaryDiceLoss, self).__init__()
        
        self._from_sigmoid = from_sigmoid
        #self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis
        self._reciprocal = use_reciprocal
       
       
    def forward(self, pred, label):
        batchsize = pred.size(0)
        
        if not self._from_sigmoid:
            pred = torch.sigmoid(pred)

        # convert to 1D
        input_pred = pred.view(batchsize, -1)
        target_label = label.view(batchsize, -1)

        # compute dice score
        intersect = torch.sum(input_pred * target_label, 1)
        input_area = torch.sum(input_pred * input_pred, 1)
        target_area = torch.sum(target_label * target_label, 1)

        sum = input_area + target_area
        epsilon = torch.tensor(1e-12)

        if self._reciprocal:
            batch_loss = sum / (2 * intersect + epsilon)
            
        else:
            batch_loss = 1.0 - (2 * intersect + epsilon) / (sum + epsilon)
            
        loss = batch_loss.mean()

        return loss
    
class SigmoidGeneralizedDiceLoss(nn.Module):
    """Generalized Dice Loss (GDL) for imbalanced segmentation tasks.
    
    Attributes:
        from_sigmoid (bool): If False, applies a sigmoid activation to predictions.
        batch_axis (int): Axis along which to compute the loss.
    """

    def __init__(self, from_sigmoid=False, batch_axis=0):
        super(SigmoidGeneralizedDiceLoss, self).__init__()
        self._from_sigmoid = from_sigmoid
        self._batch_axis = batch_axis

    def forward(self, pred, label):
        """
        Args:
            pred: (B,1,H,W) - Model logits (before sigmoid) or probabilities.
            label: (B,1,H,W) - Ground truth binary mask (0 or 1).
        Returns:
            Generalized Dice Loss value (scalar).
        """
        batchsize = pred.size(0)
        
        if not self._from_sigmoid:
            pred = torch.sigmoid(pred)  # Convert logits to probabilities

        # Flatten the tensors (B,1,H,W) → (B, H*W)
        pred = pred.view(batchsize, -1)
        label = label.view(batchsize, -1)

        # Compute per-class weights (handle class imbalance)
        label_sum = torch.sum(label, dim=1)  # Sum per batch
        weights = 1.0 / (label_sum ** 2 + 1e-12)  # Prevent division by zero

        # Compute intersection and union
        intersect = torch.sum(pred * label, dim=1)
        union = torch.sum(pred + label, dim=1)

        # Apply class weights
        numerator = 2 * torch.sum(weights * intersect)
        denominator = torch.sum(weights * union)

        # Compute Generalized Dice Loss
        gdl = 1.0 - (numerator + 1e-12) / (denominator + 1e-12)  # Small epsilon for stability

        return gdl.mean()  # Average over batch

class SigmoidWeightedBinaryCrossEntropyLoss(nn.Module):
    """Weighted Binary Cross Entropy Loss for (B,1,H,W) format"""

    def __init__(self,from_sigmoid = False, weighted=True, weight_value=None, ignore_index=-1):
        super(SigmoidWeightedBinaryCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self._from_sigmoid = from_sigmoid
        self._weighted = weighted
        self.weight_value = weight_value

    def forward(self, pred, target):
        """
        input_: (B,1,H,W) - Model prediction (logits)
        target: (B,1,H,W) - Ground truth (0 or 1)
        """
        if not self._from_sigmoid:
            pred = torch.sigmoid(pred)  # Convert logits to probabilities


        #y_onehot = one_hot(target, num_classes=2, dim=1)
        #weight = self._class_weights(y_onehot)
        # Compute class weights (inverse frequency weighting)
        with torch.no_grad():
            fg_pixels = target.sum()  # Foreground (1s)
            bg_pixels = target.numel() - fg_pixels  # Background (0s)
            weight_fg = bg_pixels / (fg_pixels + 1e-6)
            weight_bg = fg_pixels / (bg_pixels + 1e-6)

        # Compute weighted BCE
        weight = (target * weight_fg + (1 - target) * weight_bg) if (self._weighted and self.weight_value is None) else self.weight_value
        loss = F.binary_cross_entropy(pred, target, weight=weight, reduction="mean")
    
        return loss
    @staticmethod
    def _class_weights(input_):
        # normalize the input_ first
        flattened = torch.flatten(input_)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = (nominator / denominator).detach()
        return class_weights
 

class MADLoss(nn.Module):
    def __init__(self, from_sigmoid=False, normalize=False, clip_value=None):
        """
        Initialize the MADLoss class.

        Args:
            from_sigmoid (bool): If True, assumes pred_edges is already sigmoid-activated.
            normalize (bool): If True, normalizes the distance transforms.
            clip_value (float or None): If specified, clips the distances to this value.
        """
        super(MADLoss, self).__init__()
        self.from_sigmoid = from_sigmoid
        self.normalize = normalize  # Store normalization flag
        self.clip_value = clip_value  # Store clip threshold

    def forward(self, pred_edges, gt_edges):
        """
        Compute Mean Absolute Distance (MAD) loss.

        Args:
            pred_edges (torch.Tensor): Predicted edge map (B, 1, H, W), raw logits or sigmoid probabilities.
            gt_edges (torch.Tensor): Ground truth edge map (B, 1, H, W), values in [0,1].

        Returns:
            torch.Tensor: MAD loss value.
        """
        # Apply sigmoid if logits are provided
        if not self.from_sigmoid:
            pred_edges = torch.sigmoid(pred_edges)

        # Ensure gt_edges is a float tensor
        gt_edges = gt_edges.float()

        # Compute distance transforms using a proper EDT
        dt_gt = self.batch_distance_transform(gt_edges)  # Distance to nearest GT edge
        dt_pred = self.batch_distance_transform(pred_edges)  # Distance to nearest Pred edge

        # Normalize or clip distance transforms
        if self.normalize:
            max_distance = (dt_pred.shape[2] ** 2 + dt_pred.shape[3] ** 2) ** 0.5  # Diagonal of the image
            dt_gt /= max_distance
            dt_pred /= max_distance
        elif self.clip_value is not None:
            dt_gt = torch.clamp(dt_gt, max=self.clip_value)
            dt_pred = torch.clamp(dt_pred, max=self.clip_value)

        # Compute MAD loss using continuous values
        loss1 = (dt_pred * gt_edges).mean()  # Distance from GT to Pred
        loss2 = (dt_gt * pred_edges).mean()  # Distance from Pred to GT

        return (loss1 + loss2) / 2  # Symmetric MAD loss

    @staticmethod
    def batch_distance_transform(edge_maps):
        """
        Compute the Euclidean Distance Transform (EDT) for a batch of edge maps.

        Args:
            edge_maps (torch.Tensor): Binary image tensor (B, 1, H, W) where edges are 1s.

        Returns:
            torch.Tensor: Euclidean Distance Transform tensor of shape (B, 1, H, W).
        """
        batch_size, _, height, width = edge_maps.shape
        dt_maps = torch.zeros_like(edge_maps)

        for i in range(batch_size):
            # Convert to NumPy for EDT computation
            edge_np = edge_maps[i, 0].detach().cpu().numpy()
            dt = distance_transform_edt(1 - edge_np)  # Compute Euclidean Distance Transform
            dt_maps[i, 0] = torch.tensor(dt, device=edge_maps.device, dtype=edge_maps.dtype)

        return dt_maps
 
    
class HausdorffDTLoss(nn.Module):
    """
    Compute channel-wise binary Hausdorff loss based on distance transform. It can support both multi-classes and
    multi-labels tasks. The data `input` (BNHW[D] where N is number of classes) is compared with ground truth `target`
    (BNHW[D]).

    Note that axis N of `input` is expected to be logits or probabilities for each class, if passing logits as input,
    must set `sigmoid=True` or `softmax=True`, or specifying `other_act`. And the same axis of `target`
    can be 1 or N (one-hot format).

    The original paper: Karimi, D. et. al. (2019) Reducing the Hausdorff Distance in Medical Image Segmentation with
    Convolutional Neural Networks, IEEE Transactions on medical imaging, 39(2), 499-513
    """
    def __init__(
        self,
        alpha: float = 2.0,
        include_background: bool = False,
        to_onehot_y: bool = False,
        from_sigmoid: bool = False,
        softmax: bool = False,
        other_act = None,
        reduction: str = None,
        batch: bool = False,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a loss value is computed independently from each item in the batch
                before any `reduction`.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super(HausdorffDTLoss, self).__init__()
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(from_sigmoid) + int(softmax) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")

        self.alpha = alpha
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self._from_logits = from_sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.batch = batch
        self.reduction = reduction

    @torch.no_grad()
    def distance_field(self, img: torch.Tensor) -> torch.Tensor:
        """Generate distance transform.

        Args:
            img (np.ndarray): input mask as NCHWD or NCHW.

        Returns:
            np.ndarray: Distance field.
        """
        field = torch.zeros_like(img)

        for batch_idx in range(len(img)):
            fg_mask = img[batch_idx] > 0.5

            # For cases where the mask is entirely background or entirely foreground
            # the distance transform is not well defined for all 1s,
            # which always would happen on either foreground or background, so skip
            if fg_mask.any() and not fg_mask.all():
                fg_dist = torch.from_numpy(distance_transform_edt(fg_mask)).float()  # type: ignore
                bg_mask = ~fg_mask
                bg_dist = torch.from_numpy(distance_transform_edt(bg_mask)).float()  # type: ignore

                field[batch_idx] = fg_dist + bg_dist

        return field

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNHW[D], where N is the number of classes.
            target: the shape should be BNHW[D] or B1HW[D], where N is the number of classes.

        Raises:
            ValueError: If the input is not 2D (NCHW) or 3D (NCHWD).
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        Example:
            >>> import torch
            >>> from monai.losses.hausdorff_loss import HausdorffDTLoss
            >>> from monai.networks.utils import one_hot
            >>> B, C, H, W = 7, 5, 3, 2
            >>> input = torch.rand(B, C, H, W)
            >>> target_idx = torch.randint(low=0, high=C - 1, size=(B, H, W)).long()
            >>> target = one_hot(target_idx[:, None, ...], num_classes=C)
            >>> self = HausdorffDTLoss(reduction='none')
            >>> loss = self(input, target)
            >>> assert np.broadcast_shapes(loss.shape, input.shape) == input.shape
        """
        if input.dim() != 4 and input.dim() != 5:
            raise ValueError("Only 2D (NCHW) and 3D (NCHWD) supported")

            
        if not self._from_logits:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = misc.one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # If skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        device = input.device
        all_f = []
        for i in range(input.shape[1]):
            ch_input = input[:, [i]]
            ch_target = target[:, [i]]
            pred_dt = self.distance_field(ch_input.detach().cpu()).float()
            target_dt = self.distance_field(ch_target.detach().cpu()).float()

            pred_error = (ch_input - ch_target) ** 2
            distance = pred_dt**self.alpha + target_dt**self.alpha

            running_f = pred_error * distance.to(device)
            reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
            if self.batch:
                # reducing spatial dimensions and batch
                reduce_axis = [0] + reduce_axis
            all_f.append(running_f.mean(dim=reduce_axis, keepdim=True))
        f = torch.cat(all_f, dim=1)

        return f

class LogHausdorffDTLoss(HausdorffDTLoss):
    """
    Compute the logarithm of the Hausdorff Distance Transform Loss.

    This class computes the logarithm of the Hausdorff Distance Transform Loss, which is based on the distance transform.
    The logarithm is computed to potentially stabilize and scale the loss values, especially when the original loss
    values are very small.

    The formula for the loss is given by:
        log_loss = log(HausdorffDTLoss + 1)

    Inherits from the HausdorffDTLoss class to utilize its distance transform computation.
    """

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the logarithm of the Hausdorff Distance Transform Loss.

        Args:
            input (torch.Tensor): The shape should be BNHW[D], where N is the number of classes.
            target (torch.Tensor): The shape should be BNHW[D] or B1HW[D], where N is the number of classes.

        Returns:
            torch.Tensor: The computed Log Hausdorff Distance Transform Loss for the given input and target.

        Raises:
            Any exceptions raised by the parent class HausdorffDTLoss.
        """
        log_loss: torch.Tensor = torch.log(super().forward(input, target) + 1)
        return log_loss
    

##
# version 1: use torch.autograd
class LabelSmoothSoftmaxCEV1(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = logits.float() # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss


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


#
# Tools
def kl_div(a,b): # q,p
    return F.softmax(b, dim=1) * (F.log_softmax(b, dim=1) - F.log_softmax(a, dim=1))   

def one_hot2dist(seg):
    res = np.zeros_like(seg)
    for i in range(len(seg)):
        posmask = seg[i].astype(np.bool_)
        if posmask.any():
            negmask = ~posmask
            res[i] = distance_transform_edt(negmask) * negmask - (distance_transform_edt(posmask) - 1) * posmask
    return res

def class2one_hot(seg, C):
    seg = seg.unsqueeze(dim=0) if len(seg.shape) == 2 else seg
    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    return res

# Active Boundary Loss
class ABL(nn.Module):
    def __init__(self, isdetach=True, max_N_ratio = 1/100, ignore_label = 255, label_smoothing=0.2, weight = None, max_clip_dist = 20.):
        super(ABL, self).__init__()
        self.ignore_label = ignore_label
        self.label_smoothing = label_smoothing
        self.isdetach=isdetach
        self.max_N_ratio = max_N_ratio

        self.weight_func = lambda w, max_distance=max_clip_dist: torch.clamp(w, max=max_distance) / max_distance

        self.dist_map_transform = transforms.Compose([
            lambda img: img.unsqueeze(0),
            lambda nd: nd.type(torch.int64),
            partial(class2one_hot, C=1),
            itemgetter(0),
            lambda t: t.cpu().numpy(),
            one_hot2dist,
            lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])

        if label_smoothing == 0:
            self.criterion = nn.CrossEntropyLoss(
                weight=weight,
                ignore_index=ignore_label,
                reduction='none'
            )
        else:
            self.criterion = LabelSmoothSoftmaxCEV1(
                reduction='none',
                ignore_index=ignore_label,
                lb_smooth = label_smoothing
            )

    def logits2boundary(self, logit):
        eps = 1e-5
        _, _, h, w = logit.shape
        max_N = (h*w) * self.max_N_ratio
        kl_ud = kl_div(logit[:, :, 1:, :], logit[:, :, :-1, :]).sum(1, keepdim=True)
        kl_lr = kl_div(logit[:, :, :, 1:], logit[:, :, :, :-1]).sum(1, keepdim=True)
        kl_ud = torch.nn.functional.pad(
            kl_ud, [0, 0, 0, 1, 0, 0, 0, 0], mode='constant', value=0)
        kl_lr = torch.nn.functional.pad(
            kl_lr, [0, 1, 0, 0, 0, 0, 0, 0], mode='constant', value=0)
        kl_combine = kl_lr+kl_ud
        while True: # avoid the case that full image is the same color
            kl_combine_bin = (kl_combine > eps).to(torch.float)
            if kl_combine_bin.sum() > max_N:
                eps *=1.2
            else:
                break
        #dilate
        dilate_weight = torch.ones((1,1,3,3))#.cuda()
        edge2 = torch.nn.functional.conv2d(kl_combine_bin, dilate_weight, stride=1, padding=1)
        edge2 = edge2.squeeze(1)  # NCHW->NHW
        kl_combine_bin = (edge2 > 0)
        return kl_combine_bin

    def gt2boundary(self, gt, ignore_label=-1):  # gt NHW
        gt_ud = gt[:,1:,:]-gt[:,:-1,:]  # NHW
        gt_lr = gt[:,:,1:]-gt[:,:,:-1]
        gt_ud = torch.nn.functional.pad(gt_ud, [0,0,0,1,0,0], mode='constant', value=0) != 0 
        gt_lr = torch.nn.functional.pad(gt_lr, [0,1,0,0,0,0], mode='constant', value=0) != 0
        gt_combine = gt_lr+gt_ud
        del gt_lr
        del gt_ud
        
        # set 'ignore area' to all boundary
        gt_combine += (gt==ignore_label)
        
        return gt_combine > 0

    def get_direction_gt_predkl(self, pred_dist_map, pred_bound, logits):
        # NHW,NHW,NCHW
        eps = 1e-5
        # bound = torch.where(pred_bound)  # 3k
        bound = torch.nonzero(pred_bound*1)
        n,x,y = bound.T
        max_dis = 1e5

        logits = logits.permute(0,2,3,1) # NHWC

        pred_dist_map_d = torch.nn.functional.pad(pred_dist_map,(1,1,1,1,0,0),mode='constant', value=max_dis) # NH+2W+2

        logits_d = torch.nn.functional.pad(logits,(0,0,1,1,1,1,0,0),mode='constant') # N(H+2)(W+2)C
        logits_d[:,0,:,:] = logits_d[:,1,:,:] # N(H+2)(W+2)C
        logits_d[:,-1,:,:] = logits_d[:,-2,:,:] # N(H+2)(W+2)C
        logits_d[:,:,0,:] = logits_d[:,:,1,:] # N(H+2)(W+2)C
        logits_d[:,:,-1,:] = logits_d[:,:,-2,:] # N(H+2)(W+2)C
        
        """
        | 4| 0| 5|
        | 2| 8| 3|
        | 6| 1| 7|
        """
        x_range = [1, -1,  0, 0, -1,  1, -1,  1, 0]
        y_range = [0,  0, -1, 1,  1,  1, -1, -1, 0]
        dist_maps = torch.zeros((0,len(x)))#.cuda() # 8k
        kl_maps = torch.zeros((0,len(x)))#.cuda() # 8k

        kl_center = logits[(n,x,y)] # KC

        for dx, dy in zip(x_range, y_range):
            dist_now = pred_dist_map_d[(n,x+dx+1,y+dy+1)]
            dist_maps = torch.cat((dist_maps,dist_now.unsqueeze(0)),0)

            if dx != 0 or dy != 0:
                logits_now = logits_d[(n,x+dx+1,y+dy+1)]
                # kl_map_now = torch.kl_div((kl_center+eps).log(), logits_now+eps).sum(2)  # 8KC->8K
                if self.isdetach:
                    logits_now = logits_now.detach()
                kl_map_now = kl_div(kl_center, logits_now)
                
                kl_map_now = kl_map_now.sum(1)  # KC->K
                kl_maps = torch.cat((kl_maps,kl_map_now.unsqueeze(0)),0)
                torch.clamp(kl_maps, min=0.0, max=20.0)

        # direction_gt shound be Nk  (8k->K)
        direction_gt = torch.argmin(dist_maps, dim=0)
        # weight_ce = pred_dist_map[bound]
        weight_ce = pred_dist_map[(n,x,y)]
        # print(weight_ce)

        # delete if min is 8 (local position)
        direction_gt_idx = [direction_gt!=8]
        direction_gt = direction_gt[direction_gt_idx]


        kl_maps = torch.transpose(kl_maps,0,1)
        direction_pred = kl_maps[direction_gt_idx]
        weight_ce = weight_ce[direction_gt_idx]

        return direction_gt, direction_pred, weight_ce

    def get_dist_maps(self, target):
        target_detach = target.clone().detach()
        dist_maps = torch.cat([self.dist_map_transform(target_detach[i]) for i in range(target_detach.shape[0])])
        out = -dist_maps
        out = torch.where(out>0, out, torch.zeros_like(out))
        
        return out

    def forward(self, logits, target):
        eps = 1e-10
        target = target.squeeze(1)
        logits = one_hot((logits>0.49).int(),num_classes=2)
        ph, pw = logits.size(2), logits.size(3)
        h, w = target.size(1), target.size(2)

        if ph != h or pw != w:
            logits = F.interpolate(input=logits, size=(
                h, w), mode='bilinear', align_corners=True)

        gt_boundary = self.gt2boundary(target, ignore_label=self.ignore_label)

        dist_maps = self.get_dist_maps(gt_boundary)#.cuda() # <-- it will slow down the training, you can put it to dataloader.

        pred_boundary = self.logits2boundary(logits)
        if pred_boundary.sum() < 1: # avoid nan
            return None # you should check in the outside. if None, skip this loss.
        
        direction_gt, direction_pred, weight_ce = self.get_direction_gt_predkl(dist_maps, pred_boundary, logits) # NHW,NHW,NCHW

        # direction_pred [K,8], direction_gt [K]
        loss = self.criterion(direction_pred, direction_gt) # careful
        
        weight_ce = self.weight_func(weight_ce)
        loss = (loss * weight_ce).mean()  # add distance weight

        return loss

class HausdorffLoss(nn.Module):
    def __init__(self, alpha: float = 2.0, log:bool = False, include_background: bool = False):
        """
        A class for computing the Hausdorff distance loss.

        Parameters:
        - alpha (float): Controls the weighting of the distance field in the loss function.
        - include_background (bool): Whether to include the background class in the loss computation.
        """
        super(HausdorffLoss, self).__init__()
        self.hd_loss = monai.losses.HausdorffDTLoss(alpha=alpha, include_background=include_background)
        self.log = log

    def forward(self, pred, gt):
        """
        Compute the Hausdorff distance loss.

        Parameters:
        - pred (Tensor): The predicted segmentation mask.
        - gt (Tensor): The ground truth segmentation mask.

        Returns:
        - loss (Tensor): The computed loss value.
        """
        loss = self.hd_loss(pred, gt)
        if self.log:
            loss = torch.log(loss + 1)
        
        return loss
    
