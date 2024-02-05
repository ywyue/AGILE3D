from interactive_tool.gui import InteractiveSegmentationGUI
import abc
from datetime import datetime
import numpy as np
import os
from interactive_tool.utils import *
import MinkowskiEngine as ME

from models import build_model

class UserInteractiveSegmentationModel(abc.ABC):
    def __init__(self, device, config, dataloader_test):
        
        self.config = config
        # load model
        self.pretrained_weights_file = config.pretraining_weights

        self.model = build_model(config)
        self.model.to(device)
        self.model.eval()

        self.device = device
        if self.pretrained_weights_file:
            weights = self.pretrained_weights_file
            if not torch.cuda.is_available():
                map_location = 'cpu'
                print('Cuda not found, using CPU')
                model_dict = torch.load(weights, map_location)
            else:
                map_location = None
                model_dict = torch.load(weights, map_location)
            
            missing_keys, unexpected_keys = self.model.load_state_dict(model_dict['model'], strict=False)
            unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
            if len(missing_keys) > 0:
                print('Missing Keys: {}'.format(missing_keys))
            if len(unexpected_keys) > 0:
                print('Unexpected Keys: {}'.format(unexpected_keys))
        
            self.model.eval()

        # init other class parameters
        self.dataloader_test = iter(dataloader_test)
        self.num_clicks = 0
        self.coords=None
        self.pred=None
        self.labels=None
        self.feats=None


        self.cube_size = self.config.cubeedge if hasattr(config, "cubeedge") else 0.1 # the size of the segmentation mask for a single selected point
        self.visualizer = None
        self.grey_mask = None # for points occupied by other objects
        self.object_mask = None
        self.scene_name = None
        self.object_name = None
        self.scene_point_type = None

        self.device = device
        self.quantization_size = config.voxel_size
        self.click_idx = {'0':[]}

    
    def get_next_click(self, click_idx, click_time_idx, click_positions, num_clicks, run_model, gt_labels=None, ori_coords=None, scene_name=None, **args):
        """called by GUI to forward the clicks"""
        
        if run_model:

            if num_clicks == 0:
                return
            else:
                self.click_idx = click_idx

                outputs = self.model.forward_mask(self.pcd_features, self.aux, self.coordinates, 
                                                  self.pos_encodings_pcd, click_idx=[self.click_idx], 
                                                  click_time_idx=[click_time_idx])
                
                pred = outputs['pred_masks'][0].argmax(1)
                
                for obj_id, cids in self.click_idx.items():
                    pred[cids] = int(obj_id)

                pred_full = pred[self.inverse_map]
                self.object_mask[:,0] = pred_full.cpu().numpy()

                if gt_labels is not None:
                    sample_iou, _ = mean_iou_scene(pred_full, gt_labels)
                    sample_iou =  str(round(sample_iou.tolist()*100,1))
                else:
                    sample_iou = 'NA'

                f = open(self.record_file, 'a')
                now = datetime.now()
                num_obj = len(click_idx.keys())-1
                num_click = sum([len(c) for c in click_idx.values()])
                
                line = now.strftime("%Y-%m-%d-%H-%M-%S") + '  ' + scene_name + '  NumObjects:' + str(num_obj) + '  AvgNumClicks:' + str(round(num_click/num_obj,1)) + '  mIoU:' + sample_iou + '\n'
                
                f.write(line)
                f.close()

                np.save(os.path.join(self.mask_folder, 'mask_'+ str(round(num_click/num_obj,1)) + '_' +sample_iou), 
                                     pred_full.cpu().numpy())
                np.save(os.path.join(self.click_folder, 'click_'+ str(round(num_click/num_obj,1)) + '_' + sample_iou), 
                        {
                            'click_idx': click_idx,
                            'click_time': click_time_idx
                        })

                print(line)
        
        # update gui and save new object mask
        self.visualizer.update_colors(colors=self.get_colors(reload_masks=False)) # self.object_mask is already up to date
        negative_semantic = self.object_mask[:, 2].copy() * 2 # negative semantic is 2, positive semantic is 1
        object_semantic = self.object_mask[:, 0] + negative_semantic # as these channels are mutually exclusive, we get 0 for uncertain, 1 for positive and 2 for negative

        self.dataloader_test.update_object(self.object_name, object_semantic, num_new_clicks=num_clicks)

    def reset_masks(self):
        self.object_mask = np.zeros([np.shape(self.original_colors)[0], 3]) # mask,pos,neg
        self.grey_mask = np.zeros([np.shape(self.original_colors)[0]], dtype=bool)
        self.pc_clicks_idx = []
        self.nc_clicks_idx = []
    
    def get_colors(self, reload_masks=False):
        if reload_masks:
            self.reset_masks()
            self.grey_mask = self.dataloader_test.get_occupied_points_idx_except_curr(self.object_name)
            object_semantic = self.dataloader_test.get_object_semantic(self.object_name)
            
        colors = self.original_colors.copy()


        obj_ids = np.unique(self.object_mask[:, 0])
        obj_ids = [obj_id for obj_id in obj_ids if obj_id != 0]
        for obj_id in obj_ids:
            obj_mask = self.object_mask[:, 0] == obj_id
            colors[obj_mask] = get_obj_color(obj_id, normalize=True)

        return colors
    
    def check_previous_next_scene(self):
        """returns whether there are previous or next scenes, called by GUI"""
        num_scenes = len(self.dataloader_test)
        curr_scene_idx = self.dataloader_test.get_curr_scene_id()
        previous = curr_scene_idx > 0
        nxt = curr_scene_idx < num_scenes-1
        return previous, nxt, curr_scene_idx
    
    def set_slider(self, slider_value):
        self.cube_size = slider_value
    
    def run_segmentation(self): # Overrides standard run
        self.scene_name, self.scene_point_type, self.points, labels_full_ori, record_file, mask_folder, click_folder, objects = self.dataloader_test.get_curr_scene()

        self.record_file = record_file
        self.mask_folder  = mask_folder
        self.click_folder = click_folder

        
        colors_full = np.asarray(self.points.vertex_colors if self.scene_point_type == "mesh" else self.points.colors).copy()
        coords_full = np.array(self.points.points if self.scene_point_type == "pointcloud" else self.points.vertices)
        
        self.original_colors = colors_full
        self.coords = coords_full
        
        self.reset_masks()

        ### quantization
        coords_qv, unique_map, inverse_map = ME.utils.sparse_quantize(
                coordinates=self.coords,
                quantization_size=self.quantization_size,
                return_index=True,
                return_inverse=True)
        
        self.coords_qv = coords_qv
        self.colors_qv = torch.from_numpy(colors_full[unique_map]).float()


        if labels_full_ori is not None:
            self.labels_full_ori = torch.from_numpy(labels_full_ori).float().to(self.device)
            self.labels_qv_ori = self.labels_full_ori[unique_map]
        else:
            self.labels_full_ori = None
            self.labels_qv_ori  = None

        self.inverse_map = inverse_map.to(self.device)
        self.raw_coords_qv = torch.from_numpy(coords_full[unique_map]).float().to(self.device)
        
        ### compute backbone features
        data = ME.SparseTensor(
                            coordinates=ME.utils.batched_coordinates([self.coords_qv]),
                            features=self.colors_qv,
                            device=self.device
                            )
        self.pcd_features, self.aux, self.coordinates, self.pos_encodings_pcd = self.model.forward_backbone(data, raw_coordinates=self.raw_coords_qv)

        ### show UI
        self.visualizer = InteractiveSegmentationGUI(self)
        if len(objects) != 0: # init current object to the first one if there are already objects in the data set
            self.object_name = objects[0]
            colors = self.get_colors(reload_masks=True)
        else:
            colors = self.get_colors()
        self.visualizer.run(scene_name = self.scene_name, point_object = self.points, coords = self.coords, 
                            coords_qv = self.raw_coords_qv, colors = colors, original_colors = self.original_colors, 
                            original_labels = self.labels_full_ori, original_labels_qv = self.labels_qv_ori,
                            is_point_cloud=self.scene_point_type=="pointcloud", object_names=objects)

    def load_next_scene(self, quit = False, previous=False):
        self.num_clicks = 0
        if quit:
            # eventually still relevant for pyviz
            return
        prev, nxt, curr_scene_idx = self.check_previous_next_scene()
        if not previous and nxt:
            # load next scene
            self.scene_name, self.scene_point_type, self.points, labels_full_ori, record_file, mask_folder, click_folder, objects = next(self.dataloader_test)
            self.reset_masks()
        elif previous and prev:
            # load previous scene
            self.scene_name, self.scene_point_type, self.points, labels_full_ori, record_file, mask_folder, click_folder, objects = self.dataloader_test.load_scene(curr_scene_idx-1)
            self.reset_masks()
        else: 
            return
        
        self.record_file = record_file
        self.mask_folder  = mask_folder
        self.click_folder = click_folder
        
        colors_full = np.asarray(self.points.vertex_colors if self.scene_point_type == "mesh" else self.points.colors).copy()
        coords_full = np.array(self.points.points if self.scene_point_type == "pointcloud" else self.points.vertices)
        
        self.original_colors = colors_full
        self.coords = coords_full


        self.reset_masks()

        ### quantization
        coords_qv, unique_map, inverse_map = ME.utils.sparse_quantize(
                coordinates=self.coords,
                quantization_size=self.quantization_size,
                return_index=True,
                return_inverse=True)
        
        self.coords_qv = coords_qv
        self.colors_qv = torch.from_numpy(colors_full[unique_map]).float()

        if labels_full_ori is not None:
            self.labels_full_ori = torch.from_numpy(labels_full_ori).float().to(self.device)
            self.labels_qv_ori = self.labels_full_ori[unique_map]
        else:
            self.labels_full_ori = None
            self.labels_qv_ori  = None

        self.inverse_map = inverse_map.to(self.device)
        self.raw_coords_qv = torch.from_numpy(coords_full[unique_map]).float().to(self.device)
        
        ### compute backbone features
        data = ME.SparseTensor(
                            coordinates=ME.utils.batched_coordinates([self.coords_qv]),
                            features=self.colors_qv,
                            device=self.device
                            )
        self.pcd_features, self.aux, self.coordinates, self.pos_encodings_pcd = self.model.forward_backbone(data, raw_coordinates=self.raw_coords_qv)


        if len(objects) != 0: # init current object to the first one if there are already objects in the data set
            self.object_name = objects[0]
            colors = self.get_colors(reload_masks=True)
        else:
            self.object_name = None
            self.reset_masks()
            colors = self.get_colors()
        self.visualizer.set_new_scene(scene_name = self.scene_name, point_object = self.points, coords = self.coords, 
                                      coords_qv = self.raw_coords_qv, colors = colors, original_colors = self.original_colors, 
                                      original_labels = self.labels_full_ori, original_labels_qv = self.labels_qv_ori, 
                                      is_point_cloud=self.scene_point_type=="pointcloud", object_names=objects)
        return
    
    def load_object(self, object_name, load_colors=True):
        """Is called by GUI to either load an existing object or to create a new object"""
        self.object_name = object_name
        self.dataloader_test.add_object(object_name) # does nothing if object already exists
        colors = self.get_colors(reload_masks=True) if load_colors else None
        self.visualizer.select_object(colors=colors)