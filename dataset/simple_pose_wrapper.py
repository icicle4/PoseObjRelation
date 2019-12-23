from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
import mxnet as mx
import numpy as np


detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
pose_net = model_zoo.get_model('simple_pose_resnet101_v1d', pretrained=True)

detector.reset_class(['person'], reuse_weights=['person'])

# x, img = data.transforms.presets.ssd.load_test('/home/icicle/Pictures/tip-heroes.jpg', short=512)
#
# class_IDs, scores, bounding_boxs = detector(x)
#
# print('class_IDs', class_IDs)
# print('scores', scores)
# print('bbox', bounding_boxs)
#
# pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)
#
# predicted_heatmap = pose_net(pose_input)
# pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
#
# print('pred_coords', pred_coords.shape)
# print('confidence', confidence.shape)
#
# ax = utils.viz.plot_keypoints(img, pred_coords, confidence,
#                               class_IDs, bounding_boxs, scores,
#                               box_thresh=0.5, keypoint_thresh=0.2)
# plt.show()


def detector_kp(image, bbox):

    class_IDs = mx.nd.array([[[0.]]], mx.gpu())
    scores = mx.nd.array([[[1.0]]], mx.gpu())
    bounding_boxs = mx.nd.array([[bbox]], mx.gpu())

    pose_input, upscale_bbox = detector_to_simple_pose(image, class_IDs, scores, bounding_boxs)

    predicted_heatmap = pose_net(pose_input)
    pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

    print('coord', pred_coords.asnumpy().shape)
    print('confidence', confidence.asnumpy().shape)

    kps = np.concatenate([pred_coords.asnumpy(), confidence.asnumpy()], axis=2)

    return True, np.squeeze(kps, axis=0)


