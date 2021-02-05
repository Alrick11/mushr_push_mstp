# Alrick Dsouza

import time
import copy
import numpy as np
from mujoco_py import MjViewer

class MjViewer_custom(MjViewer):
    
    def render(self):
        """
        Render the current simulation state to the screen or off-screen buffer.
        Call this in your main loop.
        """
        def render_inner_loop(self):
            render_start = time.time()

            self._overlay.clear()
            if not self._hide_overlay:
                for k, v in self._user_overlay.items():
                    self._overlay[k] = v
                self._create_full_overlay()

            super().render()
            frame=[]
            if self._record_video:
                frame = self._read_pixels_as_in_window()
            else:
                self._time_per_render = 0.9 * self._time_per_render + \
                    0.1 * (time.time() - render_start)
                    
            return frame

        self._user_overlay = copy.deepcopy(self._overlay)
        # Render the same frame if paused.
        f=[]
        if self._paused:
            while self._paused:
                render_inner_loop(self)
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            # inner_loop runs "_loop_count" times in expectation (where "_loop_count" is a float).
            # Therefore, frames are displayed in the real-time.
            self._loop_count += self.sim.model.opt.timestep * self.sim.nsubsteps / \
                (self._time_per_render * self._run_speed)
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                f = render_inner_loop(self)
                self._loop_count -= 1
        # Markers and overlay are regenerated in every pass.
        self._markers[:] = []
        self._overlay.clear()
        
        return f
