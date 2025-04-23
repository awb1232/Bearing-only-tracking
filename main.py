from bearing_only_track.motion import *
from bearing_only_track.unit import *

ship = ship2d([0,0], 4, 1, 0, 1)
stage_time = [180,300,540,660]
stage_course = [60, 30, 0, 270]
stage_speed = [6,8,6,4]

dt = 0.1

times = np.arange(0, 1200, dt)
