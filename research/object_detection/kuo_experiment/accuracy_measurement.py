import os
import sys
from sklearn.metrics import f1_score
import json
import numpy as np
class Accuracy_Measurement():
    @staticmethod
    def compute_IOU(box,gt_box):
        # compute IoU of two bounding box
        ymin,xmin,ymax,xmax = box
        gt_ymin,gt_xmin,gt_ymax,gt_xmax = gt_box
        if xmax<=gt_xmin or gt_xmax <=xmin or gt_ymin >=ymax or ymin>=gt_ymax:
            return 0
        intersection = (min(xmax,gt_xmax)-max(xmin,gt_xmin))*(min(ymax,gt_ymax)
                                                              -max(ymin,gt_ymin))
        union = (xmax-xmin)*(ymax-ymin)+(gt_xmax-gt_xmin)*(gt_ymax-gt_ymin)-intersection
        return intersection/union

    @staticmethod
    def compute_single_image_F1_score(
            bb,
            det_class,
            gt_bb,
            gt_det_class,
            matching_iou_threshd=0.75):
        '''
        compute F1 score of a single image give ground truth
        :param bb:
        :param det_class:
        :param gt_bb:
        :param gt_det_class:
        :param matching_iou_threshd:
        :return: global F1 score and F1 score computed by taking average of all class
                    refer to this for details:
                    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
        '''
        num_pred =len(bb)
        num_gt=len(gt_bb)

        if num_gt >= num_pred:
            y_true = gt_det_class
            y_pred = []
            pred_left_set = set(range(num_pred))
            for true_box in gt_bb:
                idx_preBox_IoU_tupleList = [(idx, pred_box,Accuracy_Measurement.compute_IOU(pred_box, true_box))
                                            for idx, pred_box in enumerate(bb)]
                candidate_matching_bx = [i for i in idx_preBox_IoU_tupleList if i[2] > matching_iou_threshd]
                if len(candidate_matching_bx) == 0:
                    y_pred.append(0)  # predict result is background:0
                elif len(candidate_matching_bx) == 1:
                    if candidate_matching_bx[0][0] in pred_left_set:
                        y_pred.append(det_class[candidate_matching_bx[0][0]])
                        pred_left_set.remove(candidate_matching_bx[0][0])
                    else:
                        y_pred.append(0)
                else:
                    # more than one pred box match current true bd box
                    # select the one the with highest score
                    tuple_with_max_iou = max(candidate_matching_bx, key=lambda x: x[2])
                    if tuple_with_max_iou[0] in pred_left_set:
                        y_pred.append(det_class[tuple_with_max_iou[0]])
                        pred_left_set.remove(tuple_with_max_iou[0])
                    else:
                        y_pred.append(0)
                if not pred_left_set:
                    # each pred result can match a gt result at most one time
                    break
            # if the number of gt is different than the number of pred detection,
            # set the detection result to 0 to make sure the `y_true` and `y_pred` have equal length
            curr_len =len(y_pred)
            y_pred += [0] * (num_gt - curr_len)
            #print(y_true,y_pred)
            #exit()
        else:
            y_true = gt_det_class
            y_pred = []
            pred_left_set = set(range(num_pred))
            for true_box in gt_bb:
                idx_preBox_IoU_tupleList = [(idx, pred_box,Accuracy_Measurement.compute_IOU(pred_box, true_box))
                                            for idx, pred_box in enumerate(bb)]
                candidate_matching_bx = [i for i in idx_preBox_IoU_tupleList if i[2] > matching_iou_threshd]
                if len(candidate_matching_bx) == 0:
                    y_pred.append(0)  # predict result is background:0
                elif len(candidate_matching_bx) == 1:
                    if candidate_matching_bx[0][0] in pred_left_set:
                        y_pred.append(det_class[candidate_matching_bx[0][0]])
                        pred_left_set.remove(candidate_matching_bx[0][0])
                    else:
                        y_pred.append(0)
                else:
                    # more than one pred box match current true bd box
                    # select the one the with highest score
                    tuple_with_max_iou = max(candidate_matching_bx, key=lambda x: x[2])
                    if tuple_with_max_iou[0] in pred_left_set:
                        y_pred.append(det_class[tuple_with_max_iou[0]])
                        pred_left_set.remove(tuple_with_max_iou[0])
                    else:
                        y_pred.append(0)
            y_pred+=list(pred_left_set)
            y_pred = y_pred[:num_pred]
            y_true+=[0]*(num_pred-num_gt)

        # compute F1
        f1_score_global = f1_score(y_true, y_pred, average='micro')
        f1_score_mean_over_each_class = f1_score(y_true, y_pred, average='macro')
        return f1_score_global, f1_score_mean_over_each_class
    @staticmethod
    def compute_average_F1(gt_dir,pred_dir):
        json_file_names = sorted(os.listdir(gt_dir))
        f1_lst=[]
        f1_cls_aware_list=[]
        for i in json_file_names:
            with open(os.path.join(gt_dir,i)) as f:
                gt_dict = json.load(f)
            with open(os.path.join(pred_dir,i)) as f:
                pred_dict=json.load(f)
            f1,f1_cls_aware= Accuracy_Measurement.compute_single_image_F1_score(pred_dict['detection_boxes'],
                                                                   pred_dict['detection_classes'],
                                                                   gt_dict['detection_boxes'],
                                                                   gt_dict['detection_classes']
                                                                   )
            f1_lst.append(f1)
            f1_cls_aware_list.append(f1_cls_aware)
        return np.mean(f1_lst),np.mean(f1_cls_aware_list),len(f1_lst)


    @staticmethod
    def get_final_detection_result(out_dict,
                                    min_score_threshold = 0.2):
        '''
        filter out any detection whose score is less than `min_score_threshold`
        all np.array type vars are converted to list type

        :param out_dict: output_dict returned by sess.run
        :param min_score_threshold:
        :return: final modified output_dict
        '''
        out_dict['detection_scores']=\
            np.array([i for i in out_dict['detection_scores'] if i>= min_score_threshold]).tolist()
        final_num_detection = len(out_dict['detection_scores'])
        out_dict['detection_boxes']=out_dict['detection_boxes'][:final_num_detection].tolist()
        out_dict['detection_classes']=out_dict['detection_classes'][:final_num_detection].tolist()
        out_dict['num_detections']=final_num_detection

        if final_num_detection != out_dict['num_detections']:
            print('[INFO]: final number of detection NOT equal to original number of detections')
        out_dict['num_detections']=final_num_detection

    def crowd_sourcing_ground_truth(self,lst_bdbox,lst_det):
        pass

if __name__=='__main__':
    if len(sys.argv)<3:
        print('Usage: python3 script.py [path_gt] [path_pred]')
        exit()
    gt_dir = sys.argv[1]
    pred_dir = sys.argv[2]
    res = Accuracy_Measurement.compute_average_F1(gt_dir,pred_dir)
    print(res)





















