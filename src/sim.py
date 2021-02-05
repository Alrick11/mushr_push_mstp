#!/usr/bin/env python3

import yaml
import os
import glfw
import imageio
import chaospy as Cpy
import numpy as np
import mujoco_py
import time
from collections import defaultdict
from mujoco_py import load_model_from_path, MjSim, MjViewer
from scipy.spatial.transform import Rotation

from . import utils
from .mjviewer_custom import MjViewer_custom

"""
Main Loop 
"""

_BLANK_TRIAL_METRIC = {'success':False, 'block distance to goal': None, 'block pose': None, 'total_time': None, 'steps': None, 'computation_rate': None}

class MushrPushSim:


    def __init__(self, filename=None, relative=True, config=None):
        if filename==None and config==None:
            filename = '/configs/config.yaml'
            self.config = self.load_params(filename, relative)
        elif filename!=None and config==None:
            self.config = self.load_params(filename, relative)
        elif filename==None and config!=None:
            self.config = config
        else:
            raise Exception ("Check config file")
            
        self.rendered = False
        if self.load_model(self.config['model_path']): 
            self.sim = MjSim(self.model)
            self.done = False
            self.trial_done = False
            self.run_done = False
            self.trial_number = -1
            self.run_number = -1
            if not self.config['headless']:
                self.viewer = MjViewer_custom(self.sim)
                self.viewer._record_video = self.config['record_video']
                self.viewer._render_every_frame = self.config['render_every_frame']
                self.video_save_path = self.config['video_save_path']
                self.video_count=0
                self.fps=self.config['fps']
                self.headless=False
            else:
                self.headless = True
                print("Running headless, video recording not available")
                
        self.qpos_idx_lookup = self._get_qpos_idxs()

        # Metrics
        self.metric_steps = 0
        self.metric_start_time = self.sim.data.time
        self.metric_trial_storage = []
        self.metric_run_storage = []
        self.metric_comp_time = 0

        #Running to define goal_pos, goal_quat for the block
        self._set_variational_start_end()


    # -------------- PUBLIC API --------------

    def get_state(self):
        '''
        Returns the state of the sim
        Args: None
        Returns([14, 1] nparray): vector of the current state
        '''
        block_pos = self.sim.data.get_body_xpos("block")
        block_quat = self.sim.data.get_body_xquat("block")
        buddy_pos = self.sim.data.get_body_xpos("buddy")
        buddy_quat = self.sim.data.get_body_xquat("buddy")
        sensors = self.get_sensors()
        # [TODO TUDOR] this is not accurate since internal functions call get_state()
        self.request_time = time.time()
        return np.copy(np.array([block_pos, block_quat, buddy_pos, buddy_quat], dtype=object)), np.copy(sensors)

    def apply_control(self, control, timeout=False):
        '''
        Applies new control and steps sim forward
        Args([vel, angle] nparray): vector of the current control
        Returns: None 
        '''
        self.metric_comp_time += time.time() - self.request_time
        assert (control.size==2)
        self.sim.data.ctrl[0] = control[1] #Angle
        self.sim.data.ctrl[1] = control[0] #Velocity
        
        self.sim.step()
        self.metric_steps += 1
        self._check_end_condition(timeout) #Checking for end condition

    def reset_sim(self):
        '''
        Resets sim to start state
        Args: None
        Returns: None
        '''
        self.sim.reset()
        self.sim.forward()
        
    def setup_next_run(self):
        '''
        Sets internal start poses for block, car
        Args: None
        Returns: None
        '''
        self.run_done = False
        self.trial_number = -1
        self.run_number += 1
        self._set_variational_start_end()
        
    def setup_next_trial(self):
        '''
        Resets sim to start configuration + some noise
        Resets metrics
        Args: None
        Returns: None
        '''
        self.trial_done = False
        self.trial_number += 1
        # [TODO: TUDOR] add noise
        start_block_pos = self.start_block_pos
        start_buddy_pos = self.start_buddy_pos
        start_buddy_quat = self.start_buddy_quat
        
        if self.config['rel_car_block_quat'] and self.trial_number <= self.config['num_trials']-1:
            start_block_quat=self._set_variational_rel_angle()
        else:
            start_block_quat = self.start_block_quat

        self.reset_sim()
        self._set_pose_state(np.concatenate((start_block_pos, start_block_quat)), \
            np.concatenate((start_buddy_pos, start_buddy_quat)))

        self.metric_start_time = self.sim.data.time
        self.metric_steps = 0

    def get_sensors(self):
        '''
        Get's current sensor's state
        Args: None
        Returns(dict): Dictionary of sensor values
        '''
        names = self.sim.model.sensor_names
        start_adr = self.sim.model.sensor_adr
        
        y = defaultdict()
        
        for i in range(len(start_adr)):
            if i==len(start_adr)-1:
                y[names[i]] = self.sim.data.sensordata[start_adr[i]:len(self.sim.data.sensordata)]
            else:
                y[names[i]] = self.sim.data.sensordata[start_adr[i]:start_adr[i+1]]
        
        return dict(y)

    def save(self, name):
        '''
        Saves sim state
        Args (string): filename
        Returns (bool): True=success
        '''
        pass

    def load(self, name):
        '''
        Loads sim from file
        Args (string): filename
        Returns (bool): True = success 
        '''
        pass

    def load_model(self, path):
        '''
        Loads new xml model
        Args (string): path to new xml model 
        Returns (bool): True = success 
        '''
        src_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        path = src_dir + path
        try:
            self.model = mujoco_py.load_model_from_path(path)
            self.sim = mujoco_py.MjSim(self.model)
            return True
        except Exception as e:
            print(type(e))
            print(e)
            return False

    def render(self):
        '''
        Render Sim
        Args: None
        Returns: None 
        '''
        # [TODO: ALRICK] use add_marker() to add walls that are just visualizations
        assert  not self.headless, "No render option when running headless"
        resolution = glfw.get_framebuffer_size(
            self.sim._render_context_window.window)
        
        self._create_visual_boundaries()
        return self.viewer.render()

    def toggle_record_video(self):
        '''
        Toggles whether or not to record video
        Args: None
        Returns: None
        '''
        assert(not self.headless)
        self.viewer._record_video = not self.viewer._record_video
          
    def save_video(self, frames):
        '''
        Saves video
        Args: ??? 
        Returns: None
        '''
        assert(not self.headless)
        writer = imageio.get_writer(self.video_save_path.format(self.video_count), fps=self.fps)
        for f in frames:
            writer.append_data(f)
        writer.close()
        self.video_count+=1
        print ("\nDone saving ...\n")

    def load_params(self, filename, relative):
        '''
        Loads params from filename 
        '''
        if relative:
            src_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        else:
            src_dir = '/'
        with open(src_dir+filename, 'r') as stream:
            config = yaml.safe_load(stream)
        return config
    
    def get_metrics(self, run=None):
        '''
        Returns metrics 
        Args (index): run number from which to get, if None, return all runs
        Returns (int): the success rate
        '''
        if run is not None:
            if run == -1:
                # Give all data
                return self.metric_run_storage
            return self.metric_run_storage[run]
        else:
            # Compute totals for runs
            totals = []
            for run in self.metric_run_storage:
                total = self._compute_totals(run)
                totals.append(total)
            return totals
            
    def _compute_totals(self, run):
        '''
        Compute totals for a given run
        Args (array of dict): Array of trial dictionaries
        Returns (dict): Dictionary of totals/averages
        '''
        totals = _BLANK_TRIAL_METRIC.copy() 
        totals['success'] = 0
        totals['block distance to goal'] = 0
        totals['total_time'] = 0
        totals['steps'] = 0
        totals['computation_rate'] = 0
        num_trials = len(run)
        for trial in run:
            totals['success'] += trial['success']
            totals['block distance to goal'] += trial['block distance to goal']
            totals['total_time'] += trial['total_time']
            totals['steps'] += trial['steps']
            totals['computation_rate'] += trial['computation_rate']

        # Basic averages
        totals['success'] /= num_trials
        totals['block distance to goal'] /= num_trials 
        totals['total_time'] /= num_trials 
        totals['steps'] /= num_trials 
        totals['computation_rate'] /= num_trials 

        return totals

    # -------------- PRIVATE MEMBER FUNCTIONS --------------
    
    def _create_visual_boundaries(self):
        """
        Creates boundaries using marker.
        Need to run this function before each render.
        
        """
        #There wont be boundaries in atomic mode.
        
        if not self.config['atomic_mode']:
            m1=0.0375
            m2=0.08
    
            #Inner walls       
            in_color = np.array([0.7,0.7,0,1])
            in_size_x = np.array([0.75/m1,1,0.1])*m1
            in_size_y = np.array([1,0.75/m1,0.1])*m1
            
            self.viewer.add_marker(pos=np.array([0,0.75-m1,0.00000000]),\
                                        rgba=in_color,\
                                        size=in_size_x)
    
            self.viewer.add_marker(pos=np.array([0,-0.75+m1,0.00000000]),\
                                        rgba=in_color,\
                                        size=in_size_x)
    
            self.viewer.add_marker(pos=np.array([-0.75+m1,0,0.00000000]),\
                                        rgba=in_color,\
                                        size=in_size_y)
    
            self.viewer.add_marker(pos=np.array([0.75-m1,0,0.00000000]),\
                                        rgba=in_color,\
                                        size=in_size_y)
    
                    
            #Outer walls        
            out_color = np.array([0.7, 0.8, 0.4, 1])
            out_size_x = np.array([1.5/m2,1,0.1])*m2
            out_size_y = np.array([1,1.5/m2,0.1])*m2
    
            self.viewer.add_marker(pos=np.array([0,1.5-m2,0.00000000]),\
                                        rgba=out_color,\
                                        size=out_size_x)
    
            self.viewer.add_marker(pos=np.array([0,-1.5+m2,0.00000000]),\
                                        rgba=out_color,\
                                        size=out_size_x)
    
            self.viewer.add_marker(pos=np.array([-1.5+m2,0,0.00000000]),\
                                        rgba=out_color,\
                                        size=out_size_y)
    
            self.viewer.add_marker(pos=np.array([1.5-m2,0,0.00000000]),\
                                        rgba=out_color,\
                                        size=out_size_y)
    
        #Create marker visualization for goal pos
        goal_color = np.array([0.5,0.2,0.3,0.2])
        goal_pos = np.concatenate([self.goal_pos, [0]], axis=0)
        goal_thresh = self.config['goal_position_threshold']
        goal_size = np.array([goal_thresh, goal_thresh, 0.001])

        #Type=5 (Shape cylinder)
        self.viewer.add_marker(pos=goal_pos,\
                                rgba=goal_color,\
                                    size=goal_size, type=5)

    
    def _check_end_condition(self, timeout=False):
        """
        Checks whether block is out of inner square or near goal. Updates 
        self.done and self.success
        Args: None
        Returns: None

        """
        state, _ = self.get_state()
        block_pos = np.array(state[0])
        block_quat = np.array(state[1])
        buddy_pos = np.array(state[2])
        
        done = False
        success = False
        
        if not self.config['data_collection']:
            if (self._block_distance_to_goal() < self.config['goal_position_threshold']) and \
                (np.linalg.norm(self.goal_quat - block_quat ) < self.config['goal_rotation_threshold']):
                done = True
                success = True
        
        if not self.config['atomic_mode']:
            if (np.abs(block_pos) > 0.75).sum() or (np.abs(buddy_pos) > 1.5).sum():
                done = True

        if self.config['data_collection']:
            if np.linalg.norm(block_pos - buddy_pos) > self.config['dcol_max_dist_threshold']:
                done=True

        if timeout:
            done=True

        if done:
            self._compute_store_metrics(success)
            self.trial_done=True
            if self.trial_number >= self.config['num_trials'] -1:
                self. _store_run_metrics()
                self.run_done = True
            if self.run_number >= self.config['num_runs'] -1:
                self.done = True
   
    def _set_variational_rel_angle(self):
        try: 
            if self.rel_angle:
                pass
        except:
            n = self.config['num_trials']
            self.rel_angle = Cpy.J(Cpy.Uniform(-60, 60)).sample(n, rule="halton").round(3).tolist()
            
        rel_angle = Rotation.from_euler('xyz', [0.0,0.0]+\
                                        [self.rel_angle[self.trial_number]],\
                                            degrees=True)
        
        return self.start_block_quat+utils.flip_quat(rel_angle.as_quat())
        
   
    def _set_variational_start_end(self, same_loc=True):
        """
        Varies block pos and spawns mushr_car in front of it.
        Args: None
        Returns: None
        """
        try:
            if self.general_pos_dist and self.general_quat_dist:
                pass
        except AttributeError:
            n = self.config['num_runs']*2
            self.pos_idx = 0
            
            if self.config['atomic_mode']:
                _max_x, _max_y = self.config['goal_max_x'], self.config['goal_max_y']
                self.general_pos_dist = Cpy.J(Cpy.Uniform(-_max_x, _max_x),\
                                       Cpy.Uniform(-_max_y, _max_y)).sample(n, rule="halton").round(3).tolist()
                    
            else:
                self.general_pos_dist = Cpy.J(Cpy.Uniform(-0.75, 0.75),\
                                           Cpy.Uniform(-0.75, 0.75), Cpy.Uniform(-0.75, 0.75),\
                                        Cpy.Uniform(-0.75, 0.75)).sample(n, rule="halton").round(3).tolist()
    
            self.general_quat_dist = Cpy.J(Cpy.Uniform(0, 360), \
                                Cpy.Uniform(0, 360)).sample(n, rule="halton").round(3).tolist()
            
        while True:
            self.goal_pos = np.array([self.general_pos_dist[0][self.pos_idx],\
                                 self.general_pos_dist[1][self.pos_idx]])

            
            if not self.config['atomic_mode']:
                block_pos = [self.general_pos_dist[2][self.pos_idx], \
                         self.general_pos_dist[3][self.pos_idx]] \
                                                    + [0.00]
            else:
                block_pos = [0,0,0]

            if (self.pos_idx>=len(self.general_pos_dist[0])):
                raise Exception ("Need more samples in self._variational_start_end function. self.pos_idx: %d, n: %d"
                                    %(self.pos_idx, len(self.general_pos_dist[0])))

            if (np.linalg.norm(self.goal_pos - block_pos[:2]) < self.config['goal_position_threshold']):
                self.pos_idx+=1
                print ("Inside here...")
                continue
            else:
                self.pos_idx+=1
                break

        #Choosing rotation
        self.goal_quat = utils.flip_quat(Rotation.from_euler('xyz', \
                            [0.0,0.0]+\
                            [self.general_quat_dist[0][self.pos_idx]], \
                                degrees=True).as_quat())

        angle = Rotation.from_euler('xyz', [0.0,0.0]+\
                    [self.general_quat_dist[1][self.pos_idx]], degrees=True)
            
        buddy_quat = utils.flip_quat(angle.as_quat())
        
        block_quat = np.copy(buddy_quat)
        
        # Place block in front of car
        rel_pos = utils.rotate_vect([-0.330, 0, 0.0], angle.as_rotvec()[2])
        buddy_pos = [pose+rel for pose,rel in zip(block_pos, rel_pos)]
        block_pos[2] = 0.055
        
        buddy_pos = np.array(buddy_pos)
        block_pos = np.array(block_pos)

        # Set start pose for all trials
        self.start_block_pos = block_pos
        self.start_buddy_pos = buddy_pos
        self.start_block_quat = block_quat
        self.start_buddy_quat = buddy_quat
    
    def _block_distance_to_goal(self):
        '''
        Calculates block distance to goal 
        Args: None
        Returns (int): block's distance to goal 
        '''
        state, _ = self.get_state()
        block_pos = np.array(state[0])
        assert(block_pos[:2].shape==self.goal_pos.shape)
        return np.linalg.norm(block_pos[:2] - self.goal_pos)
    
    def _car_distance_to_block(self):
        '''
        Args: None
        Returns (float): distance from car to block 
        '''
        car_pos = np.array(self.sim.data.get_body_xpos("buddy"))
        block_pos = np.array(self.sim.data.get_body_xpos("block"))       
        return np.lingalg.norm(car_pos - block_pos)

    def _set_pose_state(self, block_pose, buddy_pose):
        '''
        Sets block, car poses in internal state
        Args ([7,1] ndarray, [7,1] ndarray): block pose, car pose
        '''
        state = self.sim.get_state()
        block_qpos_idxs = self.qpos_idx_lookup['block']
        buddy_qpos_idxs = self.qpos_idx_lookup['buddy']
        for i in range(7):
            state.qpos[block_qpos_idxs[i]] = block_pose[i]
            state.qpos[buddy_qpos_idxs[i]] = buddy_pose[i]
        self.sim.set_state(state)
        self.sim.forward()

    def _compute_store_metrics(self, success):
        '''
        Computes all metrics from this trial and stores them
        '''

        metrics = _BLANK_TRIAL_METRIC.copy()
        metrics['total_time'] = self.sim.data.time - self.metric_start_time
        metrics['steps'] = self.metric_steps
        metrics['computation_rate'] = 1.0/(self.metric_comp_time / self.metric_steps)
        metrics['block distance to goal'] = self._block_distance_to_goal()
        metrics['success'] = success 
        metrics['block pose'] = self.sim.data.get_body_xquat('block')
        self.metric_trial_storage.append(metrics)
    
    def _store_run_metrics(self):
        '''
        Moves all the trial metrics from this run into a seperate data structure
        '''

        self.metric_run_storage.append(self.metric_trial_storage)
        self.metric_trial_storage = []
    
    def _get_qpos_idxs(self):
        '''
        Gets a mapping from body_name -> [pos x, pos y, pos z, rot x, rot y, rot z, rot w]
        Returns (dict[string -> [7,1] ndarray]): qpos indexes for bodies in the sim
        NOTE: we will assume every body has 1 free joint (and has 7 entries in qpos)
        '''
        qpos_idx_lookup = {}
        for body_name in self.model.body_names:
            body_ID = self.model.body_name2id(body_name)
            body_jntadr = self.model.body_jntadr[body_ID]
            body_qposadr = self.model.jnt_qposadr[body_jntadr]
            qpos_idx_lookup[body_name] = np.array([body_qposadr + i for i in range(7)])
        return qpos_idx_lookup
