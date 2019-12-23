import json
from itertools import groupby
import cv2
import os
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader

from util_tools.util import center_bbox, draw_skeleton_in_frame, draw_box_in_frame, area
import random


def relation_mask_visualization(image, related_mask, kp, human_box):
    emphasis_image_part = cv2.bitwise_and(
        image, image, mask=related_mask.astype(np.uint8)
    )
    image = cv2.addWeighted(image, 0.5, emphasis_image_part, 0.5, 1)
    image = draw_skeleton_in_frame(image, np.array(kp)[:, :2])
    image = draw_box_in_frame(image, human_box)
    return image


def return_dataset(cfg):
    train_dataset = PoseBallRelationDataset(os.path.join(cfg.data_path,
                                                         'sports_ball_action_{}.json'.format('train'))).datas
    test_dataset = PoseBallRelationDataset(os.path.join(cfg.data_path,
                                                        'sports_ball_action_{}.json'.format('test'))).datas

    print('train sample: {}'.format(len(train_dataset)))
    print('test sample: {}'.format(len(test_dataset)))
    return train_dataset, test_dataset


class PoseBallRelationDataset:
    def __init__(self, json_path, stride=4):
        self.stride = stride
        self.json_path = json_path
        self.load_json()
        self.transform_to_possible_format()

    def load_json(self):
        with open(self.json_path, 'r') as f:
            annotations = json.load(f)
        self.annotations = annotations

    def fill_mask(self, mask, box, method):
        if method == 'gaussian':
            center = center_bbox(box)
            pass
        elif method == 'fill':
            xmin, ymin, xmax, ymax = box
            mask[ymin: ymax + 1, xmin: xmax + 1] = 1.0

        else:
            raise NotImplementedError('coming soon')
        return mask

    def group_same_connection(self, connections_with_action_id):
        new_connections = list()
        for c, v in groupby(connections_with_action_id, key=lambda x: x[0]):
            new_connections.append(
                c
            )
        return new_connections

    def related_vec(self, kp, human_box, related_pos):
        kp = np.asarray(kp, dtype=np.float32)
        human_area = area(human_box)
        human_radius = human_area ** 0.5
        related_x, related_y = related_pos
        kp[:, 0] -= related_x
        kp[:, 1] -= related_y
        kp[:, :2] /= human_radius
        return kp

    def transform_to_relate_vec(self, related_mask, stride):
        height, width = related_mask.shape[:2]

        positive_positions, negative_positions = list(), list()

        for h in range(0, height, stride):
            for w in range(0, width, stride):

                if related_mask[h, w] == 1.0:
                    positive_positions.append((w, h))
                else:
                    negative_positions.append((w, h))
        return positive_positions, negative_positions

    def balance_vecs(self, positive_vecs, negative_vecs):
        if len(negative_vecs) > 1.8 * len(positive_vecs):
            N = len(positive_vecs)
            M = len(negative_vecs)
            sample_inds = random.sample(list(range(0, M)), N)
            sampled_negative_vecs = [
                negative_vecs[i] for i in sample_inds
            ]
            return positive_vecs, sampled_negative_vecs

        if len(positive_vecs) > 1.8 * len(negative_vecs):
            N = len(positive_vecs)
            M = len(negative_vecs)
            sample_inds = random.sample(list(range(0, N)), M)
            sampled_positive_vecs = [
                positive_vecs[i] for i in sample_inds
            ]
            return sampled_positive_vecs, negative_vecs
        return positive_vecs, negative_vecs

    def graph_data_handle(self, vecs, class_id):
        x = torch.from_numpy(vecs).float()
        y = torch.tensor([class_id]).long()
        edge_index = torch.tensor(
            [[0, 1, 0, 2, 5, 5, 7, 6, 8, 5, 6, 11, 11, 12, 13, 14,
              1, 3, 2, 4, 6, 7, 9, 8, 10, 11, 12, 12, 13, 14, 15, 16],
             [1, 3, 2, 4, 6, 7, 9, 8, 10, 11, 12, 12, 13, 14, 15, 16,
              0, 1, 0, 2, 5, 5, 7, 6, 8, 5, 6, 11, 11, 12, 13, 14]
             ], dtype=torch.long
        )
        return Data(x=x, edge_index=edge_index, y=y)

    def transform_to_possible_format(self):

        all_positive_datas = []
        all_negative_datas = []

        for file_name, ann in self.annotations.items():
            kps = ann['kps']
            objs = ann['obj_boxs']
            human_boxs = ann['human_boxs']

            for i, kp in enumerate(kps):
                if kp is None:
                    continue
                else:
                    image = cv2.imread(file_name)
                    height, width = image.shape[:2]
                    related_mask = np.zeros((height, width), dtype=np.float32)

                    connection_with_action = ann['connection_with_action']

                    new_connections = self.group_same_connection(connection_with_action)
                    human_box = human_boxs[i]

                    for c in new_connections:
                        human_ind, obj_ind = c

                        if human_ind == i:
                            obj_box = objs[obj_ind]
                            related_mask = self.fill_mask(related_mask, obj_box, method='fill')

                    # image = relation_mask_visualization(image, related_mask, kp, human_box)
                    # cv2.imshow('res', image)
                    # cv2.waitKey(0)

                    positive_positions, negative_positions = self.transform_to_relate_vec(related_mask, self.stride)
                    positive_positions, negative_positions = self.balance_vecs(positive_positions, negative_positions)

                    positive_data = [
                        self.graph_data_handle(self.related_vec(kp, human_box, pos), 1) for pos in positive_positions
                    ]

                    negative_data = [
                        self.graph_data_handle(self.related_vec(kp, human_box, pos), 0) for pos in negative_positions
                    ]

                    all_positive_datas.extend(positive_data)
                    all_negative_datas.extend(negative_data)
        datas = all_positive_datas + all_negative_datas

        random.shuffle(datas)
        self.datas = datas
