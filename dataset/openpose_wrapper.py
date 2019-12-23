import sys
sys.path.append('/home/icicle/openpose/build/python/')
from openpose import pyopenpose as op
import copy
import numpy as np


params = dict()
params["model_folder"] = "/home/icicle/openpose/models"
params['net_resolution'] = "-1x720"

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

def openpose_25_kp(image):
    datum = op.Datum()
    image_to_process = copy.copy(image)
    datum.cvInputData = image_to_process
    opWrapper.emplaceAndPop([datum])
    pose_key_points = datum.poseKeypoints
    return pose_key_points


def detect_cropped_image(croped_image, left_corner):
    xmin, ymin = left_corner

    pose_key_points = openpose_25_kp(croped_image)

    if isinstance(pose_key_points, np.ndarray) and pose_key_points.ndim == 3 and len(pose_key_points) >= 1:
        main_pose = max(pose_key_points, key=lambda x: sum(x[:, 2]))
        main_pose[:, 0] += xmin
        main_pose[:, 1] += ymin

        return True, main_pose
    else:
        return False, None
