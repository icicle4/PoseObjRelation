import scipy.io as scio
import cv2
import os
import numpy as np
from itertools import combinations, groupby
# from openpose_wrapper import detect_cropped_image
from dataset.simple_pose_wrapper import detector_kp
import json

from util_tools.util import center_bbox, area, point_in_box


ho_relations = [
    'block',
    'carry',
    'catch',
    'dribble',
    'hit',
    'hold',
    'inspect',
    'kick',
    'pick_up',
    'serve',
    'sign',
    'spin',
    'throw',
    'no_interaction'
]

sport_ball_action_ids = list(range(489, 503))
sports_ball_annotations = list()

meaning_connection = 0
no_meaning_connection = 0


class MatchedBbox:
    def __init__(self, bbox, match_id):
        self.bbox = bbox
        self.matched_tracklet_index = set()
        self.label = -1
        self.match_id = match_id


class MergedBbox:
    def __init__(self, label, bbox, merged_inds):
        self.label = label
        self.bbox = bbox
        self.merged_inds = merged_inds


def filter_sports_ball_action(hois):
    sports_ball_hois = list()
    for hoi in hois:
        action_id, bbox_human, bbox_obj, connection, invis = hoi
        if action_id in sport_ball_action_ids and invis == 0:
            sports_ball_hois.append(hoi)
    return sports_ball_hois


def iou(src_box, target_box):
    s_xmin, s_ymin, s_xmax, s_ymax = src_box
    t_xmin, t_ymin, t_xmax, t_ymax = target_box

    s_width, s_height = s_xmax - s_xmin, s_ymax - s_ymin
    t_width, t_height = t_xmax - t_xmin, t_ymax - t_ymin

    source_area = s_width * s_height
    target_area = t_width * t_height

    cross_x = max((s_width + t_width) - (max(t_xmax, s_xmax) - min(t_xmin, s_xmin)), 0)
    cross_y = max((s_height + t_height) - (max(t_ymax, s_ymax) - min(t_ymin, s_ymin)), 0)
    cross_area = cross_x * cross_y
    return cross_area / (source_area + target_area - cross_area)


def same_box_associate_matrix(bboxs):
    N = len(bboxs)

    matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(i, N):
            if j == i:
                matrix[i, i] = 1.0
            else:
                if is_same_bbox(bboxs[i], bboxs[j]):
                    matrix[i, j] = matrix[j, i] = iou(bboxs[i], bboxs[j])
    return matrix


def cross_box(bbox1, bbox2):
    xmin_1, ymin_1, xmax_1, ymax_1 = bbox1
    xmin_2, ymin_2, xmax_2, ymax_2 = bbox2

    if max(xmin_1, xmin_2) < min(xmax_1, xmax_2) and max(ymin_1, ymin_2) < min(ymax_1, ymax_2):
        return True, (
            max(xmin_1, xmin_2),
            max(ymin_1, ymin_2),
            min(xmax_1, xmax_2),
            min(ymax_1, ymax_2)
        )
    else:
        return False, []


def is_same_bbox(bbox1, bbox2):
    center1 = center_bbox(bbox1)
    center2 = center_bbox(bbox2)
    area1 = area(bbox1)
    area2 = area(bbox2)

    if area1 == 0 or area2 == 0:
        return False

    if max(area2, area1) / min(area2, area1) > 2.5:
        return False

    is_cross, c_box = cross_box(bbox1, bbox2)

    if not is_cross:
        return False

    if not point_in_box(center1, c_box) and not point_in_box(center2, c_box):
        return False

    return True


def filter_and_flatten_associate_matrix(match_matrix, thresh=0.55):
    N = len(match_matrix)
    flatten_matrix = []
    for i in range(N):
        for j in range(i, N):
            if i == j:
                continue
            match_score = match_matrix[i, j]
            if match_score > thresh:
                flatten_matrix.append([(i, j), match_score])
    flatten_matrix = sorted(flatten_matrix, key=lambda x: x[1], reverse=True)
    return flatten_matrix


def circle_score_check(tracklet_indexs, match_matrix, thresh=0.55):
    for assoc_1, assoc_2 in combinations(list(tracklet_indexs), 2):
        if match_matrix[assoc_1, assoc_2] < thresh:
            return False
    return True


def relabel_match_bboxs(matched_bboxs, circle_match_bbox_indexs):
    circle_matched_bboxs = []
    remain_matched_bboxs = []
    not_matched_bboxs = []

    for i in range(len(matched_bboxs)):
        if i in circle_match_bbox_indexs:
            circle_matched_bboxs.append(matched_bboxs[i])
        else:
            if matched_bboxs[i].label == -1:
                not_matched_bboxs.append(matched_bboxs[i])
            else:
                remain_matched_bboxs.append(matched_bboxs[i])

    new_label = 0
    for k, v in groupby(remain_matched_bboxs, key=lambda x: x.label):
        for matched_bbox in list(v):
            matched_bbox.label = new_label
        new_label += 1

    for matched_bbox in circle_matched_bboxs:
        matched_bbox.label = new_label
        matched_bbox.matched_tracklet_index = circle_match_bbox_indexs - {matched_bbox.match_id}

    return new_label + 1


def update_associated_labels(match_matrix, thresh, matched_bboxs, sorted_match_matrix_pair, label):
    for match_matrix_pair in sorted_match_matrix_pair:
        (i, j), match_score = match_matrix_pair
        bbox1 = matched_bboxs[i]
        bbox2 = matched_bboxs[j]

        if j in bbox1.matched_tracklet_index or i in bbox2.matched_tracklet_index:
            continue

        circle_match_bbox_indexs = bbox1.matched_tracklet_index | bbox2.matched_tracklet_index | {i, j}

        if not circle_score_check(circle_match_bbox_indexs, match_matrix, thresh):
            continue
        else:
            # print('matched pair', (i, j))
            label = relabel_match_bboxs(matched_bboxs, circle_match_bbox_indexs)
    return label


def nearest_average_bbox(bboxs):
    areas = [area(bbox) for bbox in bboxs]
    mean_areas = sum(areas) / len(areas)
    best_index = np.argmin(np.abs(np.array(areas) - mean_areas))
    return bboxs[best_index]


def merge_matched_bboxs(matched_bboxs, label):
    indepent_bboxs, associated_bboxs = list(), list()
    merged_bboxs = list()
    for matched_bbox in matched_bboxs:
        if matched_bbox.label == -1:
            indepent_bboxs.append(matched_bbox)
        else:
            associated_bboxs.append(matched_bbox)

    for i in range(label):
        bboxs = []
        merged_inds = list()
        for j, matched_bbox in enumerate(associated_bboxs):
            if matched_bbox.label == i:
                bboxs.append(matched_bbox.bbox)
                merged_inds.append(matched_bbox.match_id)
        avg_bbox = nearest_average_bbox(bboxs)
        merged_bbox = MergedBbox(i, avg_bbox, merged_inds)
        merged_bboxs.append(merged_bbox)

    for i, matched_bbox in enumerate(indepent_bboxs):
        merged_bboxs.append(
            MergedBbox(i + label, matched_bbox.bbox, [matched_bbox.match_id])
        )
    return merged_bboxs


def merge_bboxs(bboxs):
    if not bboxs:
        return []
    associate_matrix = same_box_associate_matrix(bboxs)
    print('associate_matrix', associate_matrix)
    flatten_matrix = filter_and_flatten_associate_matrix(associate_matrix)
    matched_bboxs = [MatchedBbox(bbox, match_id) for match_id, bbox in enumerate(bboxs)]

    new_label = update_associated_labels(associate_matrix, 0.5, matched_bboxs, flatten_matrix, 0)

    print([matched_bbox.label for matched_bbox in matched_bboxs])

    merged_bboxs = merge_matched_bboxs(matched_bboxs, new_label)
    return merged_bboxs


def new_index(merged_bboxs, old_index):
    for i, merged_bbox in enumerate(merged_bboxs):
        if old_index in merged_bbox.merged_inds:
            return merged_bbox.label


def update_hois(hois, merged_human_bboxs, merged_obj_bboxs):
    new_connections_with_action_id = list()
    for hoi in hois:
        action_id, bbox_human, bbox_obj, connection, invis = hoi
        for c in connection:
            human_order, obj_order = c
            new_human_ind, new_obj_ind = new_index(merged_human_bboxs, human_order - 1), new_index(merged_obj_bboxs,
                                                                                                   obj_order - 1)
            new_connections_with_action_id.append(
                ((new_human_ind, new_obj_ind), int(action_id))
            )
    new_human_bboxs = [merged_human_bbox.bbox for merged_human_bbox in
                       sorted(merged_human_bboxs, key=lambda x: x.label)
                       ]

    new_obj_bboxs = [merged_obj_bbox.bbox for merged_obj_bbox in
                     sorted(merged_obj_bboxs, key=lambda x: x.label)
                     ]

    return new_human_bboxs, new_obj_bboxs, new_connections_with_action_id


def group_connection_with_action(connections_with_action_id):
    new_connections_with_action = list()
    for c, v in groupby(connections_with_action_id, key=lambda x: x[0]):
        action = ''
        for g in list(v):
            _, action_id = g
            print('action_id', action_id)
            action += ho_relations[int(action_id) - 489]

        new_connections_with_action.append(
            (c, action)
        )
    return new_connections_with_action


def vis_new_format(frame, new_human_bboxs, new_obj_bboxs, new_connections_with_action_id):
    new_connections_with_action = group_connection_with_action(new_connections_with_action_id)

    for human_bbox in new_human_bboxs:
        xmin, ymin, xmax, ymax = human_bbox

        cv2.rectangle(frame,
                      (int(xmin), int(ymin)),
                      (int(xmax), int(ymax)),
                      (0, 0, 255))

    for obj_bbox in new_obj_bboxs:
        xmin, ymin, xmax, ymax = obj_bbox

        cv2.rectangle(frame,
                      (int(xmin), int(ymin)),
                      (int(xmax), int(ymax)),
                      (0, 255, 0))

    for c, action in new_connections_with_action:
        human_ind, obj_ind = c

        human_bbox = new_human_bboxs[human_ind]
        obj_bbox = new_obj_bboxs[obj_ind]

        human_center = center_bbox(human_bbox)
        obj_center = center_bbox(obj_bbox)

        cv2.line(frame,
                 (int(human_center[0]), int(human_center[1])),
                 (int(obj_center[0]), int(obj_center[1])),
                 (255, 0, 0)
                 )

        cv2.putText(frame, action,
                    (int((human_center[0] + obj_center[0]) / 2), int((human_center[1] + obj_center[1]) / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255)
                    )
        cv2.imshow('res', frame)
        cv2.waitKey(0)



def transform_hoi_to_small_json_dataset(annotation_path):

    annotation = scio.loadmat(annotation_path)

    parts = ['train', 'test']

    for part in parts:
        sport_ball_json_dataset_path = '/home/icicle/Documents/Datasets/hico_20160224_det/sports_ball_action_{}.json'.format(part)
        actions = annotation.get('list_action')
        part_annotation = annotation.get('bbox_{}'.format(part))

        image_dir = '/home/icicle/Documents/Datasets/hico_20160224_det/images/{}2015'.format(part)

        sport_ball_action_annos = {

        }
        for ann in part_annotation[0]:
            file_name, size, hois = ann
            file_name = file_name[0]

            image_path = os.path.join(image_dir, file_name)
            image = cv2.imread(image_path)
            hois = hois[0]

            sport_ball_action_hois = filter_sports_ball_action(hois)

            if not sport_ball_action_hois:
                continue

            human_bboxs = []
            obj_bboxs = []

            for hoi in sport_ball_action_hois:
                action_id, bbox_human, bbox_obj, connection, invis = hoi
                bbox_human, bbox_obj = bbox_human[0], bbox_obj[0]

                for bbox in bbox_human:
                    xmin, xmax, ymin, ymax = map(int, bbox)
                    human_bboxs.append([xmin, ymin, xmax, ymax])

                for obj_box in bbox_obj:
                    xmin, xmax, ymin, ymax = map(int, obj_box)
                    obj_bboxs.append([xmin, ymin, xmax, ymax])

            merged_human_bboxs = merge_bboxs(human_bboxs)
            merged_obj_bboxs = merge_bboxs(obj_bboxs)

            print('merged_human_bboxs', [merged_human_bbox.label for merged_human_bbox in merged_human_bboxs])
            print('merged_obj_bboxs', [merged_obj_bbox.label for merged_obj_bbox in merged_obj_bboxs])

            new_human_bboxs, new_obj_bboxs, new_connections_with_action_id = update_hois(sport_ball_action_hois,
                                                                                         merged_human_bboxs,
                                                                                         merged_obj_bboxs)

            print('new_human_bboxs', new_human_bboxs)
            print('new_obj_bboxs', new_obj_bboxs)
            print('new_connections&actions', new_connections_with_action_id)

            # vis_new_format(image, new_human_bboxs, new_obj_bboxs, new_connections_with_action_id)

            kps = []

            for human_box in new_human_bboxs:
                xmin, ymin, xmax, ymax = human_box
                is_detected_people, kp = detector_kp(image, human_box)

                if is_detected_people:
                    kps.append(kp.tolist())
                else:
                    kps.append(kp)

            # ==============
            # type of each element
            # kps: list of ndarray/None
            # human_box and obj_box: list of [xmin, ymin, xmax, ymax]
            # connections_with_action_id: list of

            sport_ball_action_annos[image_path] = {
                'kps': kps,
                'human_boxs': new_human_bboxs,
                'obj_boxs': new_obj_bboxs,
                'connection_with_action': new_connections_with_action_id
            }

        with open(sport_ball_json_dataset_path, 'w') as f:
            json.dump(sport_ball_action_annos, f)

if __name__ == '__main__':
    annotation_path = '/home/icicle/Documents/Datasets/hico_20160224_det/anno_bbox.mat'
    transform_hoi_to_small_json_dataset(annotation_path)