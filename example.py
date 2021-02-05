#!/usr/bin/env python3

# Example usage of mushr_push_sim

import time
import numpy as np
import mujoco_py

# Outside of package you would instead use import mushr_push_sim as mps
import src.sim as mps 

def print_contact_info(sim):
    print ("Contact info\n")
    for c in sim.data.contact:
        print ('dist:\t\t', c.dist)
        print ('pos:\t\t', c.pos)
        print('frame:\t\t', c.frame)
        print('friction:\t\t', c.friction)
        print('dim:\t\t', c.dim)
        print('geom1:\t\t', c.geom1)
        print('geom2:\t\t', c.geom2)
        print ("\n\n")
        
def print_contact_valid(sim):
    print('\n\nnumber of contacts', sim.data.ncon)
    for i in range(sim.data.ncon):
        # Note that the contact array has more than `ncon` entries,
        # so be careful to only read the valid entries.
        contact = sim.data.contact[i]
        if sim.model.geom_id2name(contact.geom1)=="buddy_pusher":
            print('contact: ', i)
            print('dist: ', contact.dist)
            print('geom1: ', contact.geom1, sim.model.geom_id2name(contact.geom1))
            print('geom2: ', contact.geom2, sim.model.geom_id2name(contact.geom2))
            # There's more stuff in the data structure
            # See the mujoco documentation for more info!
            geom2_body = sim.model.geom_bodyid[sim.data.contact[i].geom2]
            print(' Contact force on geom2 body: ', sim.data.cfrc_ext[geom2_body])
            print('norm: ', np.sqrt(np.sum(np.square(sim.data.cfrc_ext[geom2_body]))))
            # Use internal functions to read out mj_contactForce
            c_array = np.zeros(6, dtype=np.float64)
            print('c_array: ', c_array)
            mujoco_py.functions.mj_contactForce(sim.model, sim.data, i, c_array)
            print('c_array: ', c_array)
    
    print('done\n\n')

def print_contact_geoms(sim):
    print ("\n\n")
    for c in sim.data.contact:
        print (sim.model.geom_id2name(c.geom1), sim.model.geom_id2name(c.geom2))
        
    print ("\n\n")

class BasicController:

    def __init__(self):
        # This is your mpc, RL agent, etc.
        # Ideally you import your controller separately and just make an instance of it in main
        pass

    def compute_control(self, state):
        # Just drive straight
        return np.array([0.5, 0.05])
        # return np.array([0.01,0.0])


if __name__=="__main__":

    # Initialize sim and controller(agent)
    print('Initializing sim & controller')
    sim = mps.MushrPushSim(filename='/config/config.yaml')
    controller = BasicController()

    print('Starting testing!')
    while not sim.done:
        sim.setup_next_run()
        print("Run:  {}".format(sim.run_number))
        while not sim.run_done: 
            sim.setup_next_trial()
            print("Trial:  {}".format(sim.trial_number))
            while not sim.trial_done: 
                state, sensor_data = sim.get_state()
                action = controller.compute_control(state)
                sim.apply_control(action)
                sim.render()
                
    m = sim.get_metrics()
    print(m)
    
    
    sim.reset_sim()
    print('Complete!')
