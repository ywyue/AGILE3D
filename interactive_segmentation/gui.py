import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import numpy as np
import time
from interactive_segmentation.utils import get_obj_color, OBJECT_CLICK_COLOR, BACKGROUND_CLICK_COLOR, UNSELECTED_OBJECTS_COLOR, SELECTED_OBJECT_COLOR, find_nearest
import torch


key_mapper = {
    321: 1,
    322: 2,
    323: 3,
    324: 4,
    325: 5,
    326: 6,
    327: 7,
    328: 8,
    329: 9,
    320: 10
}

class InteractiveSegmentationGUI:
    def __init__(self, segmentation_model):
        """GUI for the Interactive Segmentation Model. Shows point cloud and interacts with user, forwards information between model and user"""
        
        ### start webserver
        # o3d.visualization.webrtc_server.enable_webrtc()

        # class attributes to be set later (only here for overview)
        self.curr_scene_name = None
        self.mouse_event = None
        self.original_colors = None
        self.points = None
        self.is_point_cloud = True # point cloud or mesh
        self.old_colors = None # need to be saved at the beginning of each user run, in case the user deselects a point
        self.new_colors = None # changed colors within one run
        self.original_colors = None # in case the user wants to see the original object colors without any sentiment


        self.click_idx = {'0':[]}
        self.click_time_idx = {'0':[]}
        self.click_positions = {'0':[]}
        self.cur_obj_idx = -1
        self.cur_obj_name = None

        self.auto_infer = False

        self.obj_3d_labels = []

        self.key_num = None

        self.vis_mode_semantics = True # whether to show the semantics or the groundtruth colors 
        self.last_key_pressed_time = round(time.time() * 1000) # needed for check because some keys get registered multiple times else
        self.num_clicks = 0
        self.cube_size = 0.02

        # the app
        self.model = segmentation_model
        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window("AGILE3D", 1400, 700)
        font = self.app.add_font(gui.FontDescription(typeface='sans-serif'))
        self.em = self.window.theme.font_size
        self.separation_height = int(round(0.5 * self.em))
        standard_margin = gui.Margins(0.5*self.em, 0.5*self.em, 0.5*self.em, 0.5*self.em)
        zero_margin = gui.Margins(0,0,0,0)

        #### right side of visualization (description and buttons) ###
        # info about usage
        user_guide = UserInstruction(spacing=0, margin=standard_margin, font=font, separation_height=self.separation_height)
        # Collapsable options for selecting/adding more objects
        self.objects_widget = Objects(spacing=0, app=self, margin=standard_margin, font=font, separation_height=self.separation_height, em=self.em)
        self.objects_widget.update_buttons()

        self.click_info = gui.Label("Number of Click: 0") # only visible if at least one object available
        self.click_info.text_color = gui.Color(1.0, 0.5, 0.0)
        self.click_info.font_id = font

        # 
        self.auto_checkbox = gui.Checkbox("Infer automatically when clicking")
        self.auto_checkbox.set_on_checked(self.__on_cb)  # set the callback function

        # Run Button
        self.run_seg_button = gui.Button("RUN/SAVE [Enter]")
        self.run_seg_button.horizontal_padding_em = 2
        self.run_seg_button.vertical_padding_em = 2
        self.run_seg_button.set_on_clicked(self.__run_segmentation)
        self.run_seg_button.background_color = gui.Color(1.0, 0.5, 0.0)

        # Load next/previous scene buttons
        self.prev_next_widget = gui.Horiz(0, zero_margin)
        self.previous_scene_button = gui.Button("Previous Scene")
        self.previous_scene_button.horizontal_padding_em = 0.5
        self.previous_scene_button.vertical_padding_em = 0.5
        self.previous_scene_button.set_on_clicked(self.__previous_scene)
        self.next_scene_button = gui.Button("Next Scene")
        self.next_scene_button.horizontal_padding_em = 0.5
        self.next_scene_button.vertical_padding_em = 0.5
        self.next_scene_button.set_on_clicked(self.__next_scene)
        self.prev_next_widget.add_child(self.previous_scene_button)
        self.prev_next_widget.add_stretch()
        self.prev_next_widget.add_child(self.next_scene_button)
        # save and quit
        self.save_and_quit_button = gui.Button("Save and Quit")
        self.save_and_quit_button.horizontal_padding_em = 0.5
        self.save_and_quit_button.vertical_padding_em = 0.5
        self.save_and_quit_button.set_on_clicked(self.__save_and_quit)
        # add everything to the widget on the right side
        self.right_side = gui.Vert(0, gui.Margins(self.em, self.em, self.em, self.em))
        self.right_side.add_child(user_guide)
        self.right_side.add_fixed(self.separation_height)
        self.right_side.add_child(self.auto_checkbox)
        self.right_side.add_fixed(self.separation_height)
        self.right_side.add_child(self.objects_widget)
        self.right_side.add_fixed(self.separation_height)
        self.right_side.add_child(self.click_info)


        self.right_side.add_fixed(self.separation_height)
        self.right_side.add_stretch()
        self.right_side.add_fixed(self.separation_height*1)
        self.right_side.add_child(self.run_seg_button)
        self.right_side.add_fixed(self.separation_height*1)
        self.right_side.add_child(self.prev_next_widget)
        self.right_side.add_fixed(self.separation_height)
        
        #### left side of visualization (point cloud aka scene) ####
        self.widget3d = gui.SceneWidget()
        self.window.add_child(self.widget3d)
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        # point cloud's material record, needed for renderer
        self.material_record = rendering.MaterialRecord()
        self.material_record.shader = "defaultLit" # possible values: "defaultUnlit", "defaultLit", "normals", "depth"
        self.material_record.point_size = 2 * self.window.scaling
        # information about the currently picked (or unpicked) point
        self.info_coordinate_text = ""
        self.info_coordinate_label = gui.Label(f"Scene Name: {self.curr_scene_name}")
        self.info_coordinate_label.visible = True

        #### put together Window ####
        self.window.add_child(self.info_coordinate_label)
        self.window.set_on_layout(self.__on_layout)
        self.window.add_child(self.right_side)
        # set key callbacks on complete window
        # self.window.set_on_key(self.__key) # TODO: either install open3d package from master or wait for next release
        self.widget3d.set_on_key(self.__key_event)

        # if the user scrolls out too far with point clouds, we have to stop th epoints from getting too small
        self.scrolling_beyond = 0 

        ### register text entries on server
        o3d.visualization.webrtc_server.register_data_channel_message_callback(
            "visualization/gui/TextEdit", self.__set_textfield)

    def __on_layout(self, layout_context):
        """ensures the scene takes up as much space as possible"""
        r = self.window.content_rect
        self.widget3d.frame = r
        buttons_height = r.height
        buttons_width = min(r.width/3, 300)
        info_coo_height = min(r.height, self.info_coordinate_label.calc_preferred_size(layout_context, gui.Widget.Constraints()).height)
        info_coo_width = min(r.width, self.info_coordinate_label.calc_preferred_size(layout_context, gui.Widget.Constraints()).width)
        s = self.objects_widget.objects_buttons.frame
        object_buttons_height = min(r.height, self.objects_widget.objects_buttons.calc_preferred_size(layout_context, gui.ScrollableVert.Constraints()).height//2)
        self.objects_widget.objects_buttons.frame = gui.Rect(s.x, s.y, s.width, object_buttons_height)
        self.info_coordinate_label.frame = gui.Rect(r.x, r.y, info_coo_width, info_coo_height)
        self.right_side.frame = gui.Rect(r.get_right() - buttons_width, r.y, buttons_width, buttons_height)
        return
    
    ########################################################################
    ################# Callback Functions for User Feedback #################
    ########################################################################
    def __run_segmentation(self):
        """Button "Segment" or ENTER pressed by User"""
        if self.vis_mode_semantics:
            self.model.get_next_click(click_idx=self.click_idx, click_time_idx=self.click_time_idx, 
                                      click_positions=self.click_positions, num_clicks=self.num_clicks, 
                                      run_model=True, gt_labels=self.new_labels, 
                                      ori_coords = self.coordinates, scene_name=self.curr_scene_name)
        else:
            self.window.show_message_box("Toggle Object Colors/Semantics", "Please untoggle the scene color with Key <o> first!")


    def __next_scene(self):
        """Button "Next" pressed by User --> load next scene"""
        self.model.load_next_scene(quit = False)
    
    def __previous_scene(self):
        """Button "Previous" pressed by User --> load previous scene"""
        self.model.load_next_scene(quit=False, previous=True)
    
    def __slider_change(self, slider_value):
        """Slider for changing the segmentation cube around a click was changed"""
        self.model.set_slider(slider_value)
        self.cube_size = slider_value

    def __save_and_quit(self):
        """Button "Save and Quit" pressed by User"""
        self.model.load_next_scene(quit = True)

    def __key_event(self, event):
        """Recognizes key events for shortcuts"""

        if event.type == gui.KeyEvent.DOWN and key_mapper.get(event.key) != None:
            self.cur_obj_idx = key_mapper.get(event.key)
            return gui.Widget.EventCallbackResult.HANDLED

        if event.type == gui.KeyEvent.UP:
            self.cur_obj_idx = -1
            
        if event.key == 10: # "Enter" key pressed
            self.__run_segmentation()
            return gui.Widget.EventCallbackResult.HANDLED
        elif event.key == 111: # "o" pressed for toggling Original/Semantics colors
            # keys pressed just slightly longer will be recognized as key events, therefore disable key events for 200ms
            current_time = round(time.time() * 1000)
            if current_time - self.last_key_pressed_time > 200:
                # post to main thread
                gui.Application.instance.post_to_main_thread(self.window, self.__toggle_colors_mode)
                self.last_key_pressed_time = current_time
                return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def __mouse_event(self, event):
        """Callback for User Mouse Events"""
        if event.type == gui.MouseEvent.WHEEL:
            # scroll automatically adapts size of point cloud, but stays within a threshold so the points don't vanish
            if self.material_record.point_size <= self.window.scaling*2.5 and event.wheel_dy >= 0:
                self.scrolling_beyond += event.wheel_dy
            elif self.material_record.point_size <= self.window.scaling*2.5 and self.scrolling_beyond > 0:
                self.scrolling_beyond += event.wheel_dy
            else:
                self.material_record.point_size -= 0.7*event.wheel_dy # linear change, TODO: make consistent (issue opened)
            gui.Application.instance.post_to_main_thread(
                self.window, self.__update_pc_size)
            return gui.Widget.EventCallbackResult.HANDLED


        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and (event.is_modifier_down(gui.KeyModifier.CTRL) or self.cur_obj_idx != -1):
            if not self.vis_mode_semantics:
                self.window.show_message_box("Toggle Object Colors/Semantics", "Please untoggle the scene color with Key <o> first!")
                return gui.Widget.EventCallbackResult.HANDLED
            else:
                self.mouse_event = event
                self.widget3d.scene.scene.render_to_depth_image(self.__point_clicked_event)
                return gui.Widget.EventCallbackResult.HANDLED
        else:
            return gui.Widget.EventCallbackResult.IGNORED

        
    def __point_clicked_event(self, depth_image):
        """called by "__mouse_event", extracts coordinate from event and updates color according to semantic"""
        # Coordinates are expressed in absolute coordinates of the
        # window, but to dereference the image correctly we need them
        # relative to the origin of the widget. Note that even if the
        # scene widget is the only thing in the window, if a menubar
        # exists it also takes up space in the window (except on macOS).
        x = self.mouse_event.x - self.widget3d.frame.x
        y = self.mouse_event.y - self.widget3d.frame.y
        # Note that np.asarray() reverses the axes.
        depth = np.asarray(depth_image)[y, x]

        if depth == 1.0:  # clicked on nothing
            self.info_coordinate_text = ""
        else:
            point = self.widget3d.scene.camera.unproject(
                self.mouse_event.x, self.mouse_event.y, depth, self.widget3d.frame.width,
                self.widget3d.frame.height)
            point = [point[0], point[1], point[2]]

            point_idx = find_nearest(self.coordinates_qv, point)
            click_position = self.ori_coords[find_nearest(self.ori_coords, point)].cpu().tolist()

            segmentation_cube_mask = np.zeros([np.shape(self.coordinates)[0]], dtype=bool)
            segmentation_cube_mask[np.logical_and(np.logical_and(
                (np.absolute(self.coordinates[:, 0] - point[0]) < self.cube_size),
                (np.absolute(self.coordinates[:, 1] - point[1]) < self.cube_size)), 
                (np.absolute(self.coordinates[:, 2] - point[2]) < self.cube_size))] = True
            if segmentation_cube_mask.sum() <= 0: # no point clicked
                return
            if self.mouse_event.is_modifier_down(gui.KeyModifier.SHIFT) and self.mouse_event.is_modifier_down(gui.KeyModifier.CTRL):
                # unselect point via SHIFT and Ctrl
                self.info_coordinate_text = "Unselected coordinate ({:.3f}, {:.3f}, {:.3f})".format(
                    point[0], point[1], point[2])
                #TODO

            
            elif self.mouse_event.is_modifier_down(gui.KeyModifier.CTRL): 
                # Ctrl for background click
                self.info_coordinate_text = "Selected coordinate ({:.3f}, {:.3f}, {:.3f})".format(point[0], point[1], point[2])
            
                self.click_idx['0'].append(point_idx)
                self.click_time_idx['0'].append(self.num_clicks)
                self.click_positions['0'].append(click_position)

                self.new_colors[segmentation_cube_mask] = BACKGROUND_CLICK_COLOR
                self.num_clicks += 1

                if self.auto_infer:
                    self.__run_segmentation()
            
            else: 
                # Number key for object click
                self.info_coordinate_text = "Selected coordinate ({:.3f}, {:.3f}, {:.3f})".format(point[0], point[1], point[2])
               
                ### new object
                if self.click_idx.get(str(self.cur_obj_idx)) == None:
                    self.objects_widget.create_object()
                    self.click_idx[str(self.cur_obj_idx)] = [point_idx]
                    self.click_time_idx[str(self.cur_obj_idx)] = [self.num_clicks]
                    obj_3d_label = self.widget3d.add_3d_label(point, self.cur_obj_name)
                    self.obj_3d_labels.append(obj_3d_label)
                    self.click_positions[str(self.cur_obj_idx)] = [click_position]

                    ### compute new GT labels
                    if self.new_labels is not None:
                        self.new_labels[self.original_labels==self.original_labels_qv[point_idx]] = self.cur_obj_idx
                
                ### existing objects
                else:
                    self.click_idx[str(self.cur_obj_idx)].append(point_idx)
                    self.click_time_idx[str(self.cur_obj_idx)].append(self.num_clicks)
                    self.click_positions[str(self.cur_obj_idx)].append(click_position)

                self.new_colors[segmentation_cube_mask] = get_obj_color(self.cur_obj_idx, normalize=True)
                self.num_clicks += 1

                if self.auto_infer:
                    self.__run_segmentation()

                self.cur_obj_idx = -1

            gui.Application.instance.post_to_main_thread(self.window, self.__update_click_num)
            
        # post to main thread
        gui.Application.instance.post_to_main_thread(
            self.window, self.__update_colors)
    
    def __set_textfield(self, new_object_name):
        self.new_object_name = new_object_name
        print(self.new_object_name)
        def __update_textfield(self):
            self.objects_widget.object_textfield.text_value = self.new_object_name
        gui.Application.instance.post_to_main_thread(
                self.window, self.__update_textfield)

    
    def __exit(self):
        """User confirms quit, exits whole application and terminates all threads"""
        self.app.quit()
        os._exit(1)
    
    ########################################################################
    ##################### Functions used in main Thread ####################
    ########################################################################

    def __toggle_colors_mode(self):
        """called when user presses 'o' in order to toggle between showing groundtruth and semantics colors"""
        if self.vis_mode_semantics:
            # show groundtruth colors, disable clicking
            colors = o3d.utility.Vector3dVector(self.original_colors)
        else:
            # show semantic colors
            colors =  o3d.utility.Vector3dVector(self.new_colors)
        self.vis_mode_semantics = not self.vis_mode_semantics
        if self.is_point_cloud:
            self.points.colors = colors
        else:
            self.points.vertex_colors = colors
        self.widget3d.scene.remove_geometry("Points")
        self.widget3d.scene.add_geometry("Points", self.points, self.material_record)

    def __update_pc_size(self):
        """called when user zooms in, is posted to main thread to update changes in the point size"""
        self.widget3d.scene.modify_geometry_material("Points", self.material_record)
        
    def __update_colors(self):
        """called by "__depth_callback", "update_colors" and "select_object", is posted to main thread to update color in GUI after segmentation, click or object change"""
        self.info_coordinate_label.text = self.__get_info_lable_text()
        self.window.set_needs_layout()
        new_colors = o3d.utility.Vector3dVector(self.new_colors)
        if self.is_point_cloud:
            self.points.colors = new_colors
        else:
            self.points.vertex_colors = new_colors
        self.widget3d.scene.remove_geometry("Points")
        self.widget3d.scene.add_geometry("Points", self.points, self.material_record)
    
    def __update_object(self):
        """called by "select_object", is posted to main thread to update Buttons widget in GUI after loading a new or previous object"""
        # self.objects_widget.toggle_info.visible = len(self.objects_widget.objects)!=0
        self.objects_widget.toggle_info.visible = False
        self.objects_widget.toggle_info.text = f"Currently segmenting object '{self.objects_widget.objects[self.objects_widget.current_object_idx-1]}'.\nSelect a different object:"
        # self.objects_widget.toggle_info.text_color = gui.Color(*get_obj_color(self.objects_widget.current_object_idx, normalize=True))
        self.info_coordinate_label.text = f"Scene Name: {self.curr_scene_name}"
        # self.app.run_one_tick() # necessary because sometimes only updates if there is a new movement from the user

    def __update_click_num(self):
        self.click_info.text = f"Number of Click: {str(self.num_clicks)}"

    def __on_cb(self, is_checked):
        if is_checked:
            self.auto_infer = True
        else:
            self.auto_infer = False

    def __update_scene(self):
        """called by "set_new_scene", is posted to main thread to update scene in GUI"""
        # update the color and info label of the point cloud
        self.__set_prev_next_scene_buttons(*self.model.check_previous_next_scene())
        self.info_coordinate_text = ""
        self.info_coordinate_label.text = self.__get_info_lable_text()
        # (de) activate previous/next scene buttons
        self.window.set_needs_layout()
        self.widget3d.scene.show_geometry("Points", False) # hide until scene changed and camera is set up
        self.widget3d.scene.remove_geometry("Points")
        self.material_record.point_size = 2 * self.window.scaling
        self.widget3d.scene.add_geometry("Points", self.points, self.material_record)
        self.widget3d.scene.show_geometry("Points", True)
        self.app.run_one_tick()
    
    def __exit_dialogue(self):
        """called by "quit", is posted to main thread to show exit confirmation dialog for user"""
        font = self.app.add_font(gui.FontDescription(typeface='sans-serif'))
        em = self.window.theme.font_size
        # text = gui.Label(f"Finished 3D object segmentation.\nYou can review your results here:\n<a href+'{self.link}'>{self.link}</a>")
        text = gui.Label("Finished 3D object segmentation.")
        text.text_color = gui.Color(1.0, 0.5, 0.0)
        text.font_id = font
        self.exit_button = gui.Button("Exit")
        self.exit_button.horizontal_padding_em = 0.5
        self.exit_button.vertical_padding_em = 0.5
        self.exit_button.set_on_clicked(self.__exit)
        vert = gui.Vert(0, gui.Margins(em, em, em, em))
        vert.add_child(text)
        vert.add_child(self.exit_button)
        self.dialog = gui.Dialog("Finished")
        self.dialog.add_child(vert)
        self.window.show_dialog(self.dialog)
    
    def __set_prev_next_scene_buttons(self, previous, nxt, scene_idx):
        """Called by model when setting up a new scene"""
        self.previous_scene_button.visible = previous
        self.next_scene_button.visible = nxt
        self.app.run_one_tick() # necessary because only updates if there is a new movement from the user
        
    
    ########################################################################
    ################# Functions called by Interactive Model#################
    ########################################################################
    def set_new_scene(self, scene_name, point_object, coords, coords_qv, colors, original_colors, original_labels, original_labels_qv, is_point_cloud=True, object_names=[]):
        """called by Model to load next Scene"""
        self.curr_scene_name = scene_name
        self.__init_points(point_object, coords, coords_qv, colors, original_colors, original_labels, original_labels_qv, is_point_cloud)
        self.__set_up_camera()
        # delete previous settings and load new objects

        self.click_idx = {'0':[]}
        self.click_time_idx = {'0':[]}

        self.num_clicks = 0
        self.objects_widget.current_object_idx = None
        self.objects_widget.objects = []
        if len(object_names) != 0: # add objects to GUI and load coloration for segmenting first object
            for obj in object_names:
                self.objects_widget.create_object(obj, load_colors = False)
            self.objects_widget.switch_object(object_names[0])
        else:
            self.objects_widget.update_buttons()
            self.objects_widget.toggle_info.text = ""
            self.objects_widget.toggle_info.visible = False
            self.click_info.text = "Number of Click: 0"
            for obj_3d_label in self.obj_3d_labels:
                self.widget3d.remove_3d_label(obj_3d_label)

        # post to main thread
        gui.Application.instance.post_to_main_thread(
            self.window, self.__update_scene)
    
    def update_colors(self, colors):
        """called by Model to include new segmentation predictions"""
        # reset lists for new click run
        self.old_colors = colors.copy()
        self.new_colors = colors.copy()

        gui.Application.instance.post_to_main_thread(
            self.window, self.__update_colors)
    
    def select_object(self, colors=None):
        """Called by Model when new object added or to load a previously selected object
        if colors is None, only update the object Buttons widget, else also update the point cloud with the colors"""

        if colors is not None:
            self.old_colors = colors.copy()
            self.new_colors = colors.copy()
            gui.Application.instance.post_to_main_thread(
                self.window, self.__update_colors)
        self.objects_widget.update_buttons()
        gui.Application.instance.post_to_main_thread(
            self.window, self.__update_object)
        
    
    def quit(self, link):
        """final call by model, exits all threads after user confirmation"""
        self.link = link
        gui.Application.instance.post_to_main_thread(
            self.window, self.__exit_dialogue)
    
    def run(self, scene_name, point_object, coords, coords_qv, colors, original_colors, original_labels, original_labels_qv, is_point_cloud=True, object_names=[]):
        """first call by Model to load point cloud or mesh for the new object and set up camera view
        arg 'point_cloud_or_mesh' is True for point_cloud
        arg 'objects' can be used if already saved segmentations are reloaded"""
        # set scene name
        self.curr_scene_name = scene_name
        self.info_coordinate_label.text = self.__get_info_lable_text()
        # save points and colors
        self.__init_points(point_object, coords, coords_qv, colors, original_colors, original_labels, original_labels_qv, is_point_cloud)
        self.widget3d.scene.add_geometry("Points", self.points, self.material_record)
        self.__set_up_camera()
        # add objects to GUI and load coloration for segmenting first object
        if len(object_names) != 0:
            for obj in object_names:
                self.objects_widget.create_object(obj, load_colors = False)
            self.objects_widget.switch_object(object_names[0])
        # (de)activate previous/next scene buttons
        self.__set_prev_next_scene_buttons(*self.model.check_previous_next_scene())
        self.app.run()
    
    def __init_points(self, point_object, coords, coords_qv, colors, original_colors, original_labels, original_labels_qv, is_point_cloud):
        self.vis_mode_semantics = True
        self.coordinates = coords
        self.coordinates_qv = coords_qv
        self.old_colors = colors.copy()
        self.new_colors = colors.copy()
        self.original_colors = original_colors
        self.original_labels = original_labels
        self.original_labels_qv = original_labels_qv

        if self.original_labels is not None:
            self.new_labels = torch.zeros(self.original_labels.shape, device=self.original_labels.device)
        else:
            self.new_labels = None

        self.is_point_cloud = is_point_cloud
        self.points = point_object
        new_colors = o3d.utility.Vector3dVector(colors)
        if is_point_cloud:
            self.points.colors = new_colors
            # estimate normals
            self.points.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30), fast_normal_computation=False)
        else:
            self.points.vertex_colors = new_colors
            # compute surface normals
            self.points.compute_triangle_normals()
            self.points.compute_vertex_normals()
        
        self.ori_coords = torch.Tensor(self.coordinates).to(self.coordinates_qv.device)
        
    def __set_up_camera(self):
        bounds = self.widget3d.scene.bounding_box
        center = bounds.get_center()
        self.widget3d.setup_camera(35, bounds, center)
        self.widget3d.look_at(center, [0, -15, 0], [0, 0, 1]) # current dataset has its data at [0, 0, 0]
        self.widget3d.set_on_mouse(self.__mouse_event)
    def __get_info_lable_text(self):
        text = f"Scene Name: {self.curr_scene_name}"
        if len(self.info_coordinate_text) > 0:
            text += f"\n{self.info_coordinate_text}"
        return text

class UserInstruction(gui.CollapsableVert):
    """Some simple collapsable vertical User Instructions"""
    def __init__(self, spacing, margin, font, separation_height):
        gui.CollapsableVert.__init__(self, "Instructions", spacing, margin)
        descr_obj = gui.Label("{: <30}Object".format("[NUMBER + Click]"))
        descr_obj.text_color = gui.Color(*OBJECT_CLICK_COLOR)
        descr_obj.font_id = font
        descr_bg = gui.Label("{: <30}Background".format("[CTRL + Click]"))
        descr_bg.text_color = gui.Color(*BACKGROUND_CLICK_COLOR)
        descr_bg.font_id = font
        descr_unselect = gui.Label("{: <30}Unselect".format("[CTRL + SHIFT + Click]"))
        descr_unselect.text_color = gui.Color(0.8, 0.8, 0.8)
        descr_unselect.font_id = font
        desr_toggle_colors = gui.Label("{: <30}Toggle Colors".format("[O]"))
        desr_toggle_colors.text_color = gui.Color(0.8, 0.8, 0.8)
        desr_toggle_colors.font_id = font

        self.add_child(desr_toggle_colors)
        self.add_child(descr_obj)
        self.add_child(descr_bg)
        # self.add_child(descr_unselect)
        self.background_color = gui.Color(0.450, 0.454, 0.447, 0.5)

class Objects(gui.CollapsableVert):
    """Class that handles all the data and button functionalities for already created objects"""
    def __init__(self, app, spacing, margin, font, separation_height, em):
        gui.CollapsableVert.__init__(self, "Objects", spacing, margin)
        self.current_object_idx = None
        self.objects = []
        self.app = app
        self.separation_height = separation_height
        self.em = em
        # add new object
        textfield_description = gui.Label("Name a new object:")
        textfield_description.text_color = gui.Color(1.0, 0.5, 0.0)
        textfield_description.font_id = font
        new_object_widget = gui.Horiz(0, gui.Margins(0,0,0,0))
        self.object_textfield = gui.TextEdit()
        self.object_textfield.text_value = "- enter name here-"
        new_object_button = gui.Button("Create")
        new_object_button.horizontal_padding_em = 0.1
        new_object_button.vertical_padding_em = 0.1
        new_object_button.set_on_clicked(self.create_object)
        new_object_widget.add_child(self.object_textfield)
        new_object_widget.add_child(new_object_button)
        # already created objects
        self.toggle_info = gui.Label("") # only visible if at least one object available
        self.toggle_info.text_color = gui.Color(1.0, 0.5, 0.0)
        self.toggle_info.font_id = font
        self.toggle_info.visible = False
        self.dynamic_object_widget = gui.WidgetProxy()
        
        self.add_child(self.dynamic_object_widget)

    def update_buttons(self):
        """used to alter the buttons to change the object to segment 
        the dynamic object button widget is able to delete prior information and add buttons as well, which is important for changing the scene
        usual widgets cannot delete prior children, which poses a problem for changing the scene"""
        self.objects_buttons = gui.Vert(0, gui.Margins(0.5*self.em, 0.5*self.em, 0.5*self.em, 0.5*self.em))
        # objects_buttons.frame.height = 3
        for object_idx, object_name in enumerate(self.objects):

            new_obj_row = gui.Horiz(0, gui.Margins(0,0,0,0))
            new_obj_textfield = gui.TextEdit()
            new_obj_textfield.text_value = "- enter name here-"

            butt = gui.Button(object_name)
            butt.horizontal_padding_em = 1
            butt.vertical_padding_em = 0.1

            butt.background_color = gui.Color(*get_obj_color(object_idx+1, normalize=True))
            butt.set_on_clicked(lambda name=object_name: self.switch_object(name)) # switch object is called by all object buttons and will recognize the object by its id
            butt.tooltip = f"Object '{object_name}'"

            new_obj_row.add_child(butt)
            new_obj_row.add_fixed(self.separation_height)
            new_obj_row.add_child(new_obj_textfield)

            self.objects_buttons.add_child(new_obj_row)
            self.objects_buttons.add_fixed(self.separation_height/3)
        self.dynamic_object_widget.set_widget(self.objects_buttons)
    
    def create_object(self, object_name = None, load_colors = False):
        """Button "Create" pressed by User to create new object or load objects for new scene"""
        # if object_name is None:
        #     object_name = self.underscore_to_blank(self.object_textfield.text_value)
        #     self.object_textfield.text_value = "- enter name here-"
        # if object_name in self.objects:
        #     self.app.window.show_message_box("Object Name Duplicate", "That object name already exists. Please choose another name!")
        #     return
        # if object_name in ["", ""]:
        #     self.app.window.show_message_box("Invalid Object Name", "Please enter a valid object name.")
        #     return
        num_objs = len(self.objects)
        object_name = 'object ' + str(num_objs+1)

        self.objects.append(object_name)
        self.current_object_idx = self.objects.index(object_name) + 1
        self.app.cur_obj_idx = self.current_object_idx
        self.app.cur_obj_name = object_name
        self.app.model.load_object(object_name, load_colors = load_colors) # The model calls the functions to adapt the GUI
    
    def switch_object(self, object_name):
        """Object Button pressed by User"""
        if not self.app.vis_mode_semantics:
            self.app.window.show_message_box("Toggle Object Colors/Semantics", "Please untoggle the scene color with Key <o> first!")
            return
        self.current_object_idx = self.objects.index(object_name) + 1
        self.app.cur_obj_idx = self.current_object_idx
        self.app.cur_obj_name = object_name
        self.app.model.load_object(object_name, load_colors = False) # The model calls the functions to adapt the GUI
    

    @staticmethod
    def underscore_to_blank(name):
        return name.replace('_', ' ')