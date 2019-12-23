import scipy.io as scio
import pandas as pd
import cv2
import os

annotation_path = '/home/icicle/Documents/Datasets/hico_20160224_det/anno_bbox.mat'
annotation = scio.loadmat(annotation_path)

actions = annotation.get('list_action')

train_annotation = annotation.get('bbox_train')
test_annotation = annotation.get('bbox_test')

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
print(len(ho_relations))

train_image_dir = '/home/icicle/Documents/Datasets/hico_20160224_det/images/train2015'
test_image_dir = '/home/icicle/Documents/Datasets/hico_20160224_det/images/test2015'

print(len(train_annotation[0]))

sport_ball_action_ids = list(range(489, 503))
sports_ball_annotations = list()

meaning_connection = 0
no_meaning_connection = 0

for ann in train_annotation[0]:
    file_name, size, hois = ann
    file_name = file_name[0]

    W, H, C = size[0][0]
    W, H, C = W[0][0], H[0][0], C[0][0]

    hois = hois[0]

    for hoi in hois:
        action_id, bbox_human, bbox_obj, connection, invis = hoi
        if action_id in sport_ball_action_ids:

            image_path = os.path.join(train_image_dir, file_name)
            image = cv2.imread(image_path)
            small_action_id = int(action_id) - 489

            if action_id == 502:
                meaning_connection += len(connection)
            else:
                no_meaning_connection += len(connection)

            if len(connection) == 0:
                continue

            print(bbox_human)
            bbox_human, bbox_obj = bbox_human[0], bbox_obj[0]
            print('==========invis: {}=========='.format(int(invis)))

            print(image_path, len(bbox_human))
            print(bbox_human)

            for bbox in bbox_human:
                xmin, xmax, ymin, ymax = map(int, bbox)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0))

            # for c in connection:
            #     human_order, obj_order = c
            #     human_box, obj_box = bbox_human[human_order-1], bbox_obj[obj_order-1]
            #     h_xmin, h_xmax, h_ymin, h_ymax = map(int, human_box)
            #     o_xmin, o_xmax, o_ymin, o_ymax = map(int, obj_box)
            #
            #     interaction_name = ho_relations[small_action_id]
            #
            #     print('human box: {}'.format([h_xmin, h_ymin, h_xmax, h_ymax]))
            #     print('obj box: {}'.format([o_xmin, o_ymin, o_xmax, o_ymax]))
            #     print(interaction_name)

                # cv2.rectangle(image, (h_xmin, h_ymin), (h_xmax, h_ymax), (0, 255, 0))
                # cv2.rectangle(image, (o_xmin, o_ymin),  (o_xmax, o_ymax), (255, 0, 0))
                # cv2.putText(image, interaction_name, (h_xmin, h_ymin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255))

            cv2.imshow('img', image)
            cv2.waitKey(0)


print('meaning nums: {}'.format(meaning_connection))
print('no_meaning nums: {}'.format(no_meaning_connection))




