import cv2

def center_bbox(bbox):
    xmin_1, ymin_1, xmax_1, ymax_1 = bbox
    center = ((xmin_1 + xmax_1) / 2, (ymax_1 + ymin_1) / 2)
    return center


def area(bbox):
    xmin, ymin, xmax, ymax = bbox
    w, h = xmax - xmin, ymax - ymin
    return w * h


def point_in_box(p, box):
    x, y = p
    xmin, ymin, xmax, ymax = box
    if xmin < x < xmax and ymin < y < ymax:
        return True
    else:
        return False


def draw_box_in_frame(aa, box):
    xmin, ymin, xmax, ymax = box
    cv2.rectangle(
        aa, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
    return aa


def draw_skeleton_in_frame(aa, kp, idx=0, show_skeleton_labels=False):
    """
    :param aa: image
    :param kp: shape is (25, 2)
    :param idx:
    :param show_skeleton_labels:
    :return:
    """
    skeleton = [
        [0, 1], [1, 3], [0, 2], [2, 4],
        [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
        [5, 11], [6, 12], [11, 12],
        [11, 13], [12, 14], [13, 15], [14, 16]
    ]

    kp_names = [
        'nose', 'neck', 'r_shoulder', 'r_elbow', 'r_wrist',
        'l_shoulder', 'l_elbow', 'l_wrist', 'mid_hip', 'r_hip',
        'r_knee', 'r_ankle', 'l_hip', 'l_knee', 'l_ankle',
        'r_eye', 'l_eye', 'r_ear', 'l_ear', 'l_big_toe',
        'l_small_toe', 'l_heel', 'r_big_toe', 'r_small_toe', 'r_heel'
    ]

    colors = [(255, 215, 0), (0, 0, 255), (100, 149, 237), (139, 0, 139), (192, 192, 192)]

    for i, j in skeleton:
        if kp[i - 1][0] >= 0 and kp[i - 1][1] >= 0 and kp[j - 1][0] >= 0 and kp[j - 1][1] >= 0 and \
                (len(kp[i - 1]) <= 2 or (len(kp[i - 1]) > 2 and kp[i - 1][2] > 0.1 and kp[j - 1][2] > 0.1)):
            aa = cv2.line(aa, (int(kp[i - 1][0]), int(kp[i - 1][1])), (int(kp[j - 1][0]), int(kp[j - 1][1])),
                          (0, 255, 255), 5)

    for j in range(len(kp)):
        if kp[j][0] >= 0 and kp[j][1] >= 0:

            if len(kp[j]) <= 2 or (len(kp[j]) > 2 and kp[j][2] > 1.1):
                aa = cv2.circle(aa, (int(kp[j][0]), int(kp[j][1])), 5, colors[idx % 5], -1)
            elif len(kp[j]) <= 2 or (len(kp[j]) > 2 and kp[j][2] > 0.1):
                aa = cv2.circle(aa, (int(kp[j][0]), int(kp[j][1])), 5, colors[idx % 5], -1)

            if show_skeleton_labels and (len(kp[j]) <= 2 or (len(kp[j]) > 2 and kp[j][2] > 0.1)):
                aa = cv2.putText(aa, kp_names[j], tuple(kp[j][:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))
    return aa
