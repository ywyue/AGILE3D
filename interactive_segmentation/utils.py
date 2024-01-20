try:
    import open3d as o3d
except ImportError:
    raise ImportError('Please install open3d with `pip install open3d`.')
import torch

# constants and flags
USE_TRAINING_CLICKS = False
OBJECT_CLICK_COLOR = [0.2, 0.81, 0.2] # colors between 0 and 1 for open3d
BACKGROUND_CLICK_COLOR = [0.81, 0.2, 0.2] # colors between 0 and 1 for open3d
UNSELECTED_OBJECTS_COLOR = [0.4, 0.4, 0.4]
SELECTED_OBJECT_COLOR = [0.2, 0.81, 0.2]
obj_color = {1: [1, 211, 211], 2: [233,138,0], 3: [41,207,2], 4: [244, 0, 128], 5: [194, 193, 3], 6: [121, 59, 50],
             7: [254, 180, 214], 8: [239, 1, 51], 9: [125, 0, 237], 10: [229, 14, 241]}

def get_obj_color(obj_idx, normalize=False):

    r, g, b = obj_color[obj_idx]

    if normalize:
        r /= 256
        g /= 256
        b /= 256

    return [r, g, b]

def find_nearest(coordinates, value):
    distance = torch.cdist(coordinates, torch.tensor([value]).to(coordinates.device), p=2)
    return distance.argmin().tolist()

def mean_iou_single(pred, labels):
    truepositive = pred*labels
    intersection = torch.sum(truepositive==1)
    uni = torch.sum(pred==1) + torch.sum(labels==1) - intersection

    iou = intersection/uni
    return iou

def mean_iou_scene(pred, labels):

    obj_ids = torch.unique(labels)
    obj_ids = obj_ids[obj_ids!=0]
    obj_num = len(obj_ids)
    iou_sample = 0.0
    iou_dict = {}
    for obj_id in obj_ids:
        obj_iou = mean_iou_single(pred==obj_id, labels==obj_id)
        iou_dict[int(obj_id)] = float(obj_iou)
        iou_sample += obj_iou

    iou_sample /= obj_num

    return iou_sample, iou_dict