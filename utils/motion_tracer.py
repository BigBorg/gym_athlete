from functools import reduce
import numpy as np

class MotionTracer(object):
    def __init__(self, frame_shape, max_frame=5):
        self.internal_shape = [max_frame] + list(frame_shape)
        self.state_shape = (reduce(lambda x,y: x*y, self.internal_shape), )
        self.frames = np.zeros(self.internal_shape)

    def reset(self):
        self.frames = np.zeros(self.internal_shape)

    def add_frame(self, frame):
        self.frames = np.roll(self.frames, 1, 0)
        self.frames[0] = frame

    def get_state(self):
        return self.frames.flatten()
