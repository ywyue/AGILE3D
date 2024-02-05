import os
import numpy as np
import open3d as o3d
from utils.ply import read_ply

class InteractiveDataLoader:
    """
    Interactive Dataloader handling the saving and loading of a scene. It handles groundtruth object semantics
    and can save object semantics as well as calculate the iou with the groundtruth objects.
    The overall convention when using the Dataloader:
        - the parent folder is the dataset, no naming convention
        - in the dataset, there is a folder for each scene, "scene_..."
        - in the scene folder, you can find four different kinds of data:
            - 3d point cloud or mesh is named "scan.ply"
            - groundtruth objects is named "label.ply" (should contain a 'label' attribute that indicates the instance id of each point.)
            - user clicks are saved in 'clicks' folder, segmentation masks are saved in 'masks' folder
            - if iou shall be calculated between groundtruth and user defined objects, the logits are automatically saved in "iou_record.csv"
              (You do not need to create the file, this is done automatically.)
    """
    def __init__(self, config):
        self.config = config
        self.dataset_path = config.dataset_scenes
        self.user_point_type = config.point_type
        self.scene_names = []
        for scene_dir in sorted(os.listdir(self.dataset_path)):
            scene_dir_path = os.path.join(self.dataset_path, scene_dir)
            dir_name_split = scene_dir.split("_")
            if os.path.isdir(scene_dir_path) and dir_name_split[0] == "scene":
                self.scene_names.append(os.path.splitext("_".join(dir_name_split[1::]))[0])
        self.__clear_curr_status()
        self.__index = 0
        self.load_scene(0)
    
    def __clear_curr_status(self):
        self.scene_object_names = []
        self.scene_object_semantics = []
        self.scene_groundtruth_object_names = []
        self.scene_groundtruth_object_masks = [] # is later converted to np ndarray
        self.scene_iou = None # is loaded as a pd dataframe
        self.scene_groundtruth_iou_per_object = []
        self.scene_3dpoints = None
        self.point_type = None # e.g. mesh or point cloud
    
    def get_curr_scene(self):
        """returns the scene_name, scene 3d points, list of object names"""
        return self.scene_names[self.__index], self.point_type, self.scene_3dpoints, self.labels_full_ori, self.record_path, self.mask_folder, self.click_folder, [self.underscore_to_blank(name) for name in self.scene_object_names]
    
    def get_curr_scene_id(self):
        return self.__index

    def load_scene(self, idx):
        """given the scene name, returns the scene name, 3d points, list of object names"""
        # clear current lists
        name = self.scene_names[idx]
        self.__clear_curr_status()
        
        # load scene 3d points
        scene_dir = os.path.join(self.dataset_path, "scene_" + name)
        scene_3dpoints_file = os.path.join(scene_dir, 'scan.ply')

        ### set up place to save recording
        self.exp_folder = os.path.join(scene_dir, self.config.user_name)
        self.record_path = os.path.join(self.exp_folder, "iou_record.csv")
        self.mask_folder  = os.path.join(self.exp_folder, 'masks')
        self.click_folder = os.path.join(self.exp_folder, 'clicks')

        os.makedirs(self.exp_folder, exist_ok = True)
        os.makedirs(self.mask_folder, exist_ok = True)
        os.makedirs(self.click_folder, exist_ok = True)


        if not os.path.exists(os.path.join(scene_dir, 'label.ply')):
            self.labels_full_ori = None
        else:
            point_cloud = read_ply(os.path.join(scene_dir, 'label.ply'))
            self.labels_full_ori = point_cloud['label'].astype(np.int32)
        
        # with open(scene_3dpoints_file, 'rb') as f:
        pcd_type = o3d.io.read_file_geometry_type(scene_3dpoints_file)
        if self.user_point_type is not None and self.user_point_type.lower() == "mesh" and not pcd_type == o3d.io.FileGeometry.CONTAINS_TRIANGLES:
            print("[USER WARNING] You specified the point type to be a mesh, but only a point cloud was found...using point cloud")
        elif self.user_point_type is not None and self.user_point_type.lower() == "pointcloud":
            pcd_type = o3d.io.FileGeometry.CONTAINS_POINTS
        elif self.user_point_type is not None:
            pcd_type = o3d.io.read_file_geometry_type(scene_3dpoints_file)
            print("[USER WARNING] User given preference for point type is unknown. Loading automatic type..")
        
        if pcd_type == o3d.io.FileGeometry.CONTAINS_TRIANGLES:
            self.scene_3dpoints = o3d.io.read_triangle_mesh(scene_3dpoints_file)
            self.point_type = "mesh"
        elif pcd_type == o3d.io.FileGeometry.CONTAINS_POINTS:
            self.scene_3dpoints = o3d.io.read_point_cloud(scene_3dpoints_file)
            self.point_type = "pointcloud"
        else:
            raise Exception(f"Data Format of 3d points in '3dpoints.ply' unknown for scene {name}")
        
        self.__index = self.scene_names.index(name)
        return name, self.point_type, self.scene_3dpoints, self.labels_full_ori, self.record_path, self.mask_folder, self.click_folder, [self.underscore_to_blank(name) for name in self.scene_object_names]

        
    def get_object_semantic(self, name):
        obj_idx = self.scene_object_names.index(self.blank_to_underscore(name))
        obj_semantic = self.scene_object_semantics[obj_idx].copy()
        return obj_semantic
    
    def add_object(self, object_name):
        """given an object name, creates a new file for the object mask, returns True if successful, else False"""
        object_name_underscore = self.blank_to_underscore(object_name)
        if object_name_underscore in self.scene_object_names:
            return
        # update class variables
        shape = np.shape(np.asarray(self.scene_3dpoints.points if self.point_type == "pointcloud" else self.scene_3dpoints.vertices)[:, 0])
        self.scene_object_semantics.append(np.zeros(shape, dtype=np.ubyte))
        self.scene_object_names.append(object_name_underscore)

        return
    
    def update_object(self, object_name, semantic, num_new_clicks):
        """given an object name and the object's mask, overwrites the existing object mask"""
        object_name_underscore = self.blank_to_underscore(object_name)
        obj_idx = self.scene_object_names.index(object_name_underscore)
        assert(self.scene_object_semantics[obj_idx].shape == semantic.shape)
        self.scene_object_semantics[obj_idx] = semantic.copy()

    def get_occupied_points_idx_except_curr(self, curr_object_name):
        """returns mask for 'occupied' points belonging to at least one object"""
        obj_idx = self.scene_object_names.index(self.blank_to_underscore(curr_object_name))
        other_objects = self.scene_object_semantics.copy()
        other_objects.pop(obj_idx)
        mask = np.logical_or.reduce(np.ma.masked_equal(other_objects, 1).mask)
        return mask

    def __iter__(self):
        return self

    def __next__(self):
        if (self.__index + 1) < len(self.scene_names):
            self.__index += 1
            return self.load_scene(self.__index)
        raise StopIteration
    
    def __len__(self):
        return len(self.scene_names)

    @staticmethod
    def blank_to_underscore(name):
        return name.replace(' ', '_')
    
    @staticmethod
    def underscore_to_blank(name):
        return name.replace('_', ' ')