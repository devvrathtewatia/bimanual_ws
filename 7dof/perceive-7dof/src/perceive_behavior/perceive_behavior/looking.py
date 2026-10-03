"""Where the arms go so the workspace camera can see - ROS-free, for the suite.

SHARED FILE, between perceive and navigate (the 'sense' group in SHARED_FILES.sha256): the
sections that look. The same pose clears the MAST camera's view too, above image row 330.
"""

# THE LOOK POSE. Parked, the hands sit at (0.16, +/-0.06, 0.40) - squarely between the
# workspace camera and the near half of the workspace - so a lying object in front loses an
# end behind the left hand in the image and its silhouette fit is 7 mm out; no camera position
# avoids it, because from anywhere above, the parked hands are in the way. So the arms move to
# LOOK: hands spread to (0.14, +/-0.24, 0.32), tool down. Chosen by search: every sight line
# from the camera to every object at radius 0.27-0.33, bearing +/-8 deg, lying at 60-120 deg or
# standing, clears both arms by 41 mm; the arms clear each other by 178 mm and the base and
# mast by 19 mm; and no joint moves more than 45 deg from parked, so the move there and back is
# a short joint-space line checked in full (test_the_look_pose_clears_the_camera_view). It
# reaches out further than parked, so the base never turns in it.
LOOK_LEFT = [0.1693, -1.9302, 0.7053, 1.9552, 0.2313, 1.4678, -1.5382]
LOOK_RIGHT = [-0.1693, -1.9302, -0.7053, 1.9552, -0.2313, 1.4678, -1.6034]
LOOK_SAMPLES = 8
