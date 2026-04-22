from functools import lru_cache

import cv2
import numpy as np


def visualize_instances(imask, bg_color=255,
                        boundaries_color=None, boundaries_width=1, boundaries_alpha=0.8):
    num_objects = imask.max() + 1
    palette = get_palette(num_objects)
    if bg_color is not None:
        palette[0] = bg_color

    result = palette[imask].astype(np.uint8)
    if boundaries_color is not None:
        boundaries_mask = get_boundaries(imask, boundaries_width=boundaries_width)
        tresult = result.astype(np.float32)
        tresult[boundaries_mask] = boundaries_color
        tresult = tresult * boundaries_alpha + (1 - boundaries_alpha) * result
        result = tresult.astype(np.uint8)

    return result


@lru_cache(maxsize=16)
def get_palette(num_cls):
    palette = np.zeros(3 * num_cls, dtype=np.int32)

    for j in range(0, num_cls):
        lab = j
        i = 0

        while lab > 0:
            palette[j*3 + 0] |= (((lab >> 0) & 1) << (7-i))
            palette[j*3 + 1] |= (((lab >> 1) & 1) << (7-i))
            palette[j*3 + 2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3

    return palette.reshape((-1, 3))


def visualize_mask(mask, num_cls):
    palette = get_palette(num_cls)
    mask[mask == -1] = 0

    return palette[mask].astype(np.uint8)


def visualize_proposals(proposals_info, point_color=(255, 0, 0), point_radius=1):
    proposal_map, colors, candidates = proposals_info

    proposal_map = draw_probmap(proposal_map)
    for x, y in candidates:
        proposal_map = cv2.circle(proposal_map, (y, x), point_radius, point_color, -1)

    return proposal_map


def draw_probmap(x):
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_HOT)


def draw_points(image, points, color, radius=3):
    image = image.copy()
    for p in points:
        if p[0] < 0:
            continue
        if len(p) == 3:
            marker = {
                0: cv2.MARKER_CROSS,
                1: cv2.MARKER_DIAMOND,
                2: cv2.MARKER_STAR,
                3: cv2.MARKER_TRIANGLE_UP
            }[p[2]] if p[2] <= 3 else cv2.MARKER_SQUARE
            image = cv2.drawMarker(image, (int(p[1]), int(p[0])),
                                   color, marker, 4, 1)
        else:
            pradius = radius
            image = cv2.circle(image, (int(p[1]), int(p[0])), pradius, color, -1)

    return image


def draw_instance_map(x, palette=None):
    num_colors = x.max() + 1
    if palette is None:
        palette = get_palette(num_colors)

    return palette[x].astype(np.uint8)


def blend_mask(image, mask, alpha=0.6):
    if mask.min() == -1:
        mask = mask.copy() + 1

    imap = draw_instance_map(mask)
    result = (image * (1 - alpha) + alpha * imap).astype(np.uint8)
    return result


def get_boundaries(instances_masks, boundaries_width=1):
    boundaries = np.zeros((instances_masks.shape[0], instances_masks.shape[1]), dtype=np.bool)

    for obj_id in np.unique(instances_masks.flatten()):
        if obj_id == 0:
            continue

        obj_mask = instances_masks == obj_id
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        inner_mask = cv2.erode(obj_mask.astype(np.uint8), kernel, iterations=boundaries_width).astype(np.bool)

        obj_boundary = np.logical_xor(obj_mask, np.logical_and(inner_mask, obj_mask))
        boundaries = np.logical_or(boundaries, obj_boundary)
    return boundaries
    
 
def draw_with_blend_and_clicks(img, mask=None, alpha=0.6, clicks_list=None, pos_color=(0, 255, 0),
                               neg_color=(0, 0, 255), radius=4):
    result = img.copy()

    if mask is not None:
        palette = get_palette(np.max(mask) + 1)
        rgb_mask = palette[mask.astype(np.uint8)]

        mask_region = (mask > 0).astype(np.uint8)
        result = result * (1 - mask_region[:, :, np.newaxis]) + \
            (1 - alpha) * mask_region[:, :, np.newaxis] * result + \
            alpha * rgb_mask
        result = result.astype(np.uint8)

        # result = (result * (1 - alpha) + alpha * rgb_mask).astype(np.uint8)

    if clicks_list is not None and len(clicks_list) > 0:
        pos_points = [click.coords for click in clicks_list if click.is_positive]
        neg_points = [click.coords for click in clicks_list if not click.is_positive]

        result = draw_points(result, pos_points, pos_color, radius=radius)
        result = draw_points(result, neg_points, neg_color, radius=radius)

    return result


def draw_extremes(img, mask=None, alpha=0.6, clicks_list=None, iou=None,
                               pos_color=(0, 255, 0), neg_color=(0, 0, 255), radius=2):
    result = img.copy()

    # Draw mask overlay
    if mask is not None:
        palette = get_palette(np.max(mask) + 1 if np.max(mask) > 0 else 2)
        palette[1] = [0, 255, 0] # Force ID 1 to Green
        rgb_mask = palette[mask.astype(np.uint8)]

        mask_region = (mask > 0).astype(np.uint8)
        result = result * (1 - mask_region[:, :, np.newaxis]) + \
                 (1 - alpha) * mask_region[:, :, np.newaxis] * result + \
                 alpha * rgb_mask
        result = result.astype(np.uint8)

    # Draw clicks with numbering
    if clicks_list is not None and len(clicks_list) > 0:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.25
        thickness = 1

        # Starting coordinates for text column
        x_text = 5  # horizontal offset from left edge
        y_step = 15  # vertical spacing between lines
        y_text_start = 50  # top margin

        for idx, click in enumerate(clicks_list):
            x, y = click.coords
            color = pos_color if click.is_positive else neg_color
            radius = click.radius if hasattr(click, 'radius') else radius
            #print('Used radius:', click.radius)
            cv2.circle(result, (y, x), radius, color, -1)
            cv2.circle(result, (y, x), radius, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(result, str(idx+1), (y+3, x-3), font, font_scale, color, thickness, cv2.LINE_AA)
            label = f'c{idx+1}, r{click.radius}'
            y_pos = y_text_start + idx * y_step  # vertically stacked
            cv2.putText(result, label, (x_text, y_pos), font, font_scale+0.2 , color, thickness, cv2.LINE_AA)

    # Display IoU score if provided
    if iou is not None:
        text = f"IoU: {iou:.4f}"
        cv2.putText(result, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(result, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    return result

