
import os
import argparse
import json
import xml.etree.ElementTree as ET
from collections import namedtuple
from collections import defaultdict
from collections import OrderedDict
from accuracy_measurement import Accuracy_Measurement
import numpy as np
import ILSVRC_coco_map
import copy
import importlib
from pathlib import Path
from PIL import Image
from itertools import cycle
import matplotlib.pyplot as plt
from plot import Plot

count_gt_obj_not_in_coco_ILSVRC_intersection=0

parser = argparse.ArgumentParser()
parser.add_argument('-sel',nargs=3,metavar=('Category_file_path','category_pointer_file_dir',
                        'train_set_dir'),help="To select subset from training set")
# used only with -sel option
parser.add_argument('-output_sel_snippet_file',nargs=1,default='selected_snippets.txt',
                    help='File name for store all selected snippets')
parser.add_argument('-single_acc',nargs=2,metavar=('res_path','gt_path'),
                    help='Compute single frame accuracy')
parser.add_argument('-ILSVRC_EVA', nargs=1,metavar=('config_file'),
                    help='Evaluate detection results on ILSVRC dataset')
parser.add_argument('-dataset_management',nargs=2,metavar=('all_snippets_file', 'dataset_root')
                    ,help='Report statistics of dataset:# of snippet in each video;# of video in each cat')
args = parser.parse_args()
if args.ILSVRC_EVA:
    config_file_woExtension = args.ILSVRC_EVA[0].replace('.py', '')  # remove '.py'
    config = importlib.import_module(config_file_woExtension)
else:
    config=None


def select_subset_from_train_set(category_file,
                                 category_pointer_file_dir,
                                 train_set_dir,
                                 selected_snippet_dir_file):
    '''
    select subset of training snippet to ensure that the object in each frame is in coco-defined categories.

    :param category_file: path to a file which specifies which categories to use,
                in the file, if there is more than one space separated fields, this category is excluded
    :param category_pointer_file_dir: the dir which contains category pointer file (`train_1.txt`)
    :param train_set_dir: the path of the dir contain training set:
                        e.g.,[THIS PATH]/ILSVRC2015_VID_train_0000
    :return: dict {category(int): [path_to_snippet]}, the snippet dir contains all the frames of that
            snippet
    '''
    category_set = set() # set storing the category ID (1-30)
    with open(category_file) as fp:
        count=1
        for line in fp:
            line = line.strip().split()
            if len(line)>1:
                count+=1
                continue
            else:
                category_set.add(count)
                count+=1

    cat_to_snippet_dict=defaultdict(list) #{categorty_id(int): [snippet_dir_path]}
    for cat in category_set:
        pointer_file =os.path.join(category_pointer_file_dir,'train_{}.txt'.format(cat))
        with open(pointer_file) as fp:
            for line in fp:
                sub_path=line.strip().split()[0]
                train_set,snippet_dir_id = sub_path.split(sep='/')
                cat_to_snippet_dict[cat].append(
                    os.path.join(train_set_dir,train_set,snippet_dir_id))
    #print(cat_to_snippet_dict[1])
    # write cat_to_snippet_dict to file
    with open(selected_snippet_dir_file,'w') as fp:
        for key,val in cat_to_snippet_dict.items():
            for item in cat_to_snippet_dict[key]:
                fp.write('{} {}\n'.format(item,key))
    return cat_to_snippet_dict

class ILSVRC_Evaluation:
    def __init__(self,
                 snippets_file_path,
                 dataset_root_path,
                 model_name,
                 purpose,
                 res_dir_name,
                 evaluation_res_dir,
                 threshds_lst,
                 min_detection_score,
                 write_or_append):
        self.Metric_Tuple = namedtuple('metric', ['snippet_path','accuracies'] )
        self.global_FP= 0
        self.global_TP= 0
        self.global_FN= 0
        self.cat_intersection = ILSVRC_coco_map.ILSVRC_TO_COCO.values()# view object
        self.threshds=threshds_lst
        self.min_detection_score=min_detection_score # min detection score for evaluation
        self.snippets_path=snippets_file_path
        self.dataset_root_path=dataset_root_path
        self.model_name=model_name
        self.purpose=purpose
        self.res_dir_name=res_dir_name
        self.evaluation_res_dir=evaluation_res_dir
        # since this class is specific for ILSVRC dataset, hardcode path component names
        self.dataset_training_root_path = os.path.join(self.dataset_root_path,'Data')
        self.dataset_annotation_root_path = os.path.join(self.dataset_root_path,'Annotations')
        self.dataset_result_root_path = os.path.join(self.dataset_root_path,self.res_dir_name,self.purpose,
                                                     self.model_name,'Data')
        self.write_or_append =write_or_append
        # variables for various statistics
        self.metric=[] # [ Metric_Tuple ], namely [(snippet_path_id,[])]
        # {category:
        #   {videoID:
        #       {snippet:
        #           [accuracy_single_frame]
        #       }
        #   }
        #  }
        self.cat_video_accuracs_dict=OrderedDict()
        global count_gt_obj_not_in_coco_ILSVRC_intersection
        # populate results only once
        self.cat_video_speeds_dict=OrderedDict()
        self.cat_video_snippetCount_dict =OrderedDict()
        self.compute_accuracy_all()
        self.compute_speed()
    def _find_best_matched_gtBbox(self,gt_bboxes,pred_bboxes,min_threshd):
        '''
        find best matched gt bbox for each pred bbox.
        :param gt_bboxes:
        :param pred_bboxes:
        :param min_threshd:
        :return: return the index of matching bbox or None if no matching is found
        '''
        def _filter_list(lst,threshd):
            truncate_idx = None
            for i,idx_iou_tuple in enumerate(lst):
                if idx_iou_tuple[1]<threshd:
                    truncate_idx = i
                    break
            if truncate_idx is not None:
                del lst[truncate_idx:]

        already_matched =[False]*len(gt_bboxes)
        result=[] #size of result is len(pred_bbox),each entry is the index of matching gtbb or None
        matrix_ious = [[(gt_idx,Accuracy_Measurement.compute_IOU(box,gt_box))
                        for gt_idx,gt_box in enumerate(gt_bboxes)]
                       for box in pred_bboxes]

        # sort in place
        matrix_ious = [sorted(per_pred_ious,key=lambda k:k[1],reverse=True)
                       for per_pred_ious in matrix_ious]
        # remove tuples whoes iou is less than threshd
        for item in matrix_ious:
            _filter_list(item,min_threshd)
        # find the best matched gt bbox
        for ious in matrix_ious:
            # ious is empty or can not find a match ---> None
            find = False
            for iou in ious:
                if not already_matched[iou[0]]:
                    result.append(iou[0])
                    already_matched[iou[0]]=True
                    find =  True
                    break
            if not find:
                result.append(None)
        return result


    def compute_accuracy_single_frame(self,pred_path,gt_path):
        '''
        compute average single frame accuracy over different IoU thresholds,
        thresholds list is defined in `__init__()`
        :param pred_path:
        :param gt_path:
        :return: single frame accuracy (averaged over multiple IoU threshold)
        '''
        # read from frame result and ground truth files
        def filter_detection_by_score(pred_classes,pred_bbxes,pred_scores,min_score):
            res_cls=[]
            res_bbs=[]
            res_scr=[]
            for idx,scr in enumerate(pred_scores):
                if scr >= min_score:
                    res_cls.append(pred_classes[idx])
                    res_bbs.append(pred_bbxes[idx])
                    res_scr.append(pred_scores[idx])
            return res_cls,res_bbs,res_scr

        global count_gt_obj_not_in_coco_ILSVRC_intersection
        tree = ET.parse(gt_path)
        root = tree.getroot()
        gt_width = int(root[3][0].text)
        gt_height = int(root[3][1].text)
        gt_object_element=root.findall('object')
        l1=len(gt_object_element)
        # filter out all gt objects which are not in the intersection of COCO and ILSVRC
        gt_object_element =[i for i in gt_object_element
                            if i[1].text in ILSVRC_coco_map.ILSVRC_TO_COCO]
        l2=len(gt_object_element)
        count_gt_obj_not_in_coco_ILSVRC_intersection+=(l1-l2)
        if gt_object_element:
            gt_classes=[ILSVRC_coco_map.ILSVRC_TO_COCO[obj[1].text]
                        for obj in gt_object_element] # need convert ILSVRC class id to coco id
            # ymin xmin ymax xmax
            gt_bd=[ (int(obj[2][3].text)/gt_height,int(obj[2][1].text)/gt_width,
                 int(obj[2][2].text)/gt_height,int(obj[2][0].text)/gt_width)
                for obj in gt_object_element]
            gt_num_detection=len(gt_classes)
        with open(pred_path) as f:
            pred_out_dict = json.load(f)
        pred_bd = pred_out_dict['detection_boxes']
        pred_classes= pred_out_dict['detection_classes']
        pred_scores = pred_out_dict['detection_scores']
        pred_num_detection=len(pred_bd)
        # filter out detection whose score is less than min_score in EVALUATION
        pred_classes,pred_bd,pred_scores = filter_detection_by_score(pred_classes,pred_bd,
                                      pred_scores,
                                      self.min_detection_score)
        # IMPORTANT: filter out pred detections whose category are not in the intersection of coco and ILSVRC
        temp=[ (bd,cls) for bd,cls in zip(pred_bd,pred_classes)
               if cls in self.cat_intersection]
        if temp:
            pred_bd,pred_classes = list(zip(*temp))
        else:
            pred_bd=[]
            pred_classes=[]

        # compute accuracy
        if not pred_classes and not gt_object_element:
            return -1
        elif not pred_classes and gt_object_element:
            self.global_FN = self.global_FN+gt_num_detection
            return -1
        elif pred_classes and not gt_object_element:
            self.global_FP=self.global_FP+pred_num_detection
            # This need further consideration and determination
            #return -1
            return 0
        else:
            accuracies=[] # holding accuracy values at different threshds
            for threshd in self.threshds:
                FP=0
                TP=0
                # idx of gt bbox or None
                matching_result = self._find_best_matched_gtBbox(gt_bd,pred_bd,threshd)
                for idx,matching_idx in enumerate(matching_result):
                    if matching_idx is None:
                        FP+=1
                        self.global_FP+=1
                        continue
                    if pred_classes[idx] == gt_classes[matching_idx]:
                        TP+=1
                        self.global_TP+=1
                    else:
                        FP+=1
                        self.global_FP+=1
                accuracies.append(TP/(TP+FP))
            return np.average(accuracies)
    def compute_accuracy_all(self):
        # can only be called once on instance initialization
        with open(self.snippets_path) as fd:
            for line in fd:
                snippet_path,category_key = line.strip().split()
                snippet_path_component_shared = snippet_path.replace(self.dataset_training_root_path+'/','',1)
                video_id = snippet_path_component_shared[:-3]
                snippet_id_of_single_video = snippet_path_component_shared[-3:]
                self.metric.append(self.Metric_Tuple(snippet_path_component_shared,[]))

                if category_key not in self.cat_video_accuracs_dict:
                    self.cat_video_accuracs_dict[category_key] = OrderedDict()
                if video_id not in self.cat_video_accuracs_dict[category_key]:
                    self.cat_video_accuracs_dict[category_key][video_id]=OrderedDict()
                if snippet_id_of_single_video not in self.cat_video_accuracs_dict[category_key][video_id]:
                    self.cat_video_accuracs_dict[category_key][video_id][snippet_id_of_single_video]=[]

                frames = sorted(os.listdir(snippet_path))
                for frame in frames:
                    frame_name,_ = os.path.splitext(frame)
                    gt_path = os.path.join(self.dataset_annotation_root_path,
                                           snippet_path_component_shared,frame_name+'.xml')
                    pred_path =os.path.join(self.dataset_result_root_path,
                                            snippet_path_component_shared,frame_name+'.json')
                    acc = self.compute_accuracy_single_frame(pred_path,gt_path)
                    self.cat_video_accuracs_dict[category_key][video_id][snippet_id_of_single_video].append(acc)
                    self.metric[-1].accuracies.append(acc)
                print('[INFO:] Snippet:{} is done for computing acc'.format(snippet_path))
        print('[INFO:] Count of objects for which gt should not be given:{}'.format(
            count_gt_obj_not_in_coco_ILSVRC_intersection))
    def compute_speed(self):
        # can only be called once on instance initialization
        def _read_process_time(pred_path):
            with open(pred_path) as f:
                pred_out_dict = json.load(f)
            pred_response_time = pred_out_dict['response_time']
            pred_processing_time = pred_out_dict['process_time']
            return float(pred_response_time),float(pred_processing_time)
        # populate self.cat_video_speeds_dict and self.cat_video_snippetCount_dict
        with open(self.snippets_path) as fd:
            for line in fd:
                snippet_path, category_key = line.strip().split()
                snippet_path_component_shared = snippet_path.replace(self.dataset_training_root_path + '/', '', 1)
                video_id = snippet_path_component_shared[:-3]
                snippet_id_of_single_video = snippet_path_component_shared[-3:]
                if category_key not in self.cat_video_speeds_dict:
                    self.cat_video_speeds_dict[category_key] = OrderedDict()
                    self.cat_video_snippetCount_dict[category_key]=OrderedDict()
                if video_id not in self.cat_video_speeds_dict[category_key]:
                    self.cat_video_speeds_dict[category_key][video_id] = OrderedDict()
                    self.cat_video_snippetCount_dict[category_key][video_id]=OrderedDict()
                if snippet_id_of_single_video not in self.cat_video_speeds_dict[category_key][video_id]:
                    self.cat_video_speeds_dict[category_key][video_id][snippet_id_of_single_video] = []
                frames = sorted(os.listdir(snippet_path))
                self.cat_video_snippetCount_dict[category_key][video_id][snippet_id_of_single_video]=len(frames)
                for frame in frames:
                    frame_name, _ = os.path.splitext(frame)
                    pred_path = os.path.join(self.dataset_result_root_path,
                                             snippet_path_component_shared, frame_name + '.json')
                    rst,prt = _read_process_time(pred_path)
                    self.cat_video_speeds_dict[category_key][video_id][snippet_id_of_single_video].append((rst,prt))
                print('[INFO:] Snippet:{} is done for computing speed and frame counting'.format(snippet_path))

    def compute_global_accuracy(self):
        # global average accuracy
        per_snippet_acc = []
        per_snippet_acc_weight=[]
        for snippet in self.metric:
            effective_frame_accs =[i for i in snippet.accuracies if i !=-1]
            per_snippet_acc.append(np.average(effective_frame_accs))
            per_snippet_acc_weight.append(len(effective_frame_accs))
        global_acc = np.average(per_snippet_acc,weights=per_snippet_acc_weight)
        print('Per snippet accuracy:',per_snippet_acc)
        print('Global Average accuracy',global_acc)
    def compute_acc_per_video_per_snippet(self):
        '''
        write to file (both json and txt): [Cat] [VideoId] [Accuracy]
        AND
        write to file (both json and txt): [Cat] [VideoId] [Acc_snippet_1,Acc_snippet_2,...]
        :return:
        '''
        # construct output file path
        # Hardcode Name of the res file:
        file_name_per_video_acc = 'per_video_accuracy_min_score_{}.txt'.format(self.min_detection_score)
        file_name_per_video_acc_json ='per_video_accuracy_min_score_{}.json'.format(self.min_detection_score)
        file_name_per_snippet_acc ='per_snippet_accuracy_min_score_{}.txt'.format(self.min_detection_score)
        file_name_per_snippet_acc_json ='per_snippet_accuracy_min_score_{}.json'.format(self.min_detection_score)
        path_base = os.path.join(self.dataset_root_path,self.evaluation_res_dir,
                            self.purpose,self.model_name)
        Path(path_base).mkdir(parents=True,exist_ok=True)
        per_video_res_file_path = os.path.join(path_base,file_name_per_video_acc)
        per_video_res_file_path_json =os.path.join(path_base,file_name_per_video_acc_json)
        per_snippet_res_file_path = os.path.join(path_base,file_name_per_snippet_acc)
        per_snippet_res_file_path_json =os.path.join(path_base,file_name_per_snippet_acc_json)
        # Data structure holding the result
        per_video_dict = OrderedDict()#{video: average_acc of all snippets}
        per_snippet_dict =OrderedDict() #{video: [acc_snippet_1,acc_snippet_2,...]}
        fd_per_video = open(per_video_res_file_path,self.write_or_append)
        fd_per_snippet = open(per_snippet_res_file_path,self.write_or_append)
        for cat,vd_snippet_dict in self.cat_video_accuracs_dict.items():
            for vd,snippet_dic in vd_snippet_dict.items():
                snippet_average_acc = [] # [acc of snippets of a specific video]
                snippet_average_weight =[]
                for snippet,acc_lst in snippet_dic.items():
                    effective_frame_acc = [i for i in acc_lst if i!=-1]
                    if effective_frame_acc:
                        snippet_average_acc.append(np.average(effective_frame_acc))
                        snippet_average_weight.append(len(effective_frame_acc))
                    else:
                        snippet_average_acc.append(0)
                        # if all frames in the snippet do not have effective accuracy (all -1), ignore this
                        # snippet when computing average acc for the video to which the snippet belongs
                        snippet_average_weight.append(0)
                per_video_dict[vd]=np.average(snippet_average_acc,
                                              weights=snippet_average_weight)
                per_snippet_dict[vd]=snippet_average_acc
                fd_per_video.write('{} [{}] {}\n'.format(cat,vd,per_video_dict[vd]))
                fd_per_snippet.write(
                    '{} [{}] {}\n'.format(cat,vd,','.join(map(str,snippet_average_acc))))
        fd_per_video.close()
        fd_per_snippet.close()

        # fp_per_video_js=open(per_video_res_file_path_json,'w')
        # fp_per_snippet_js = open(per_snippet_res_file_path_json,'w')
        # json.dump(per_video_dict,fp_per_video_js)
        # json.dump(per_snippet_dict,fp_per_snippet_js)
        # fp_per_video_js.close()
        # fp_per_snippet_js.close()
    def compute_per_video_per_snippet_speed_and_count(self):
        '''write to file: [Cat] [VideoId] [speed_snippet_1,speed_snippet_2,...] [Ave speed]'''
        file_name_per_snippet_speed = 'speed_per_snippet.txt'
        #file_name_per_snippet_count = 'count_per_snippet.txt'
        path_base = os.path.join(self.dataset_root_path, self.evaluation_res_dir,
                                 self.purpose, self.model_name)
        Path(path_base).mkdir(parents=True, exist_ok=True)
        per_snippet_speed_path = os.path.join(path_base,file_name_per_snippet_speed)
        #per_snippet_count_path =os.path.join(path_base,file_name_per_snippet_count)
        fd_speed = open(per_snippet_speed_path,self.write_or_append)
        #fd_count = open(per_snippet_count_path,'w')
        #fd_speed.write('[Cat] [VideoId] [speed_snippet_1,speed_snippet_2,...] [Ave speed]\n')
        for cat,vd_snippet_dict in self.cat_video_speeds_dict.items():
            for vd,snippet_dict in vd_snippet_dict.items():
                snippet_average_speed =[]
                snippet_average_weight =[]
                for snippet,speed_lst in snippet_dict.items():
                    rst,prt = list(zip(*speed_lst)) # speed lst is tuple (response time,process time)
                    per_snippet_ave= np.average(rst)
                    snippet_average_speed.append(per_snippet_ave)
                    snippet_average_weight.append(len(rst))
                per_vd_ave =np.average(snippet_average_speed,weights=snippet_average_weight)
                fd_speed.write('{} {} [{}] ({})\n'.format(
                    cat,vd,','.join(map(str,snippet_average_speed)),per_vd_ave))
        fd_speed.close()
    def draw_per_frame_acc(self):
        model_name =self.model_name
        save_dir = os.path.join(self.dataset_root_path,'temp_per_frame_acc')
        Path(save_dir).mkdir(parents=True,exist_ok=True)
        #   [ {videoID:
        #       {snippet:
        #           [accuracy_single_frame]
        #       }
        #      }
        #   ]
        list_vds = self.cat_video_accuracs_dict.values()
        for item in list_vds:
            for vd, snippet_dict in item.items():
                frames_acc_list =[]
                for i in snippet_dict.values():
                    frames_acc_list=frames_acc_list+i
                frames_acc_list=[i for i in frames_acc_list if i!=-1]
                save_file_name = os.path.join(save_dir,model_name+'_'+vd.replace('/','_'))
                Plot.draw_per_frame_accuracy_for_single_video(vd,model_name,frames_acc_list,
                                                              save_file_name)
        plt.show()


class Dataset_Management_ILSVRC:
    def __init__(self,
                 all_snippets_file_path,# training dataset path
                 dataset_root_path):
        self.all_snippet_file=all_snippets_file_path
        self.dataset_root = dataset_root_path
        self.dataset_training_root_path = os.path.join(self.dataset_root,'Data')
        self.dataset_dict = {} #{ category:{videoID:[snippets]}}
        self.num_video_in_cat_dict = {} # {category: # of videos}
        self.num_snippets_in_video = {} # {category: {videoID: # of snippet}}
        # MUST MAKE SURE THAT EACH INSTANCE OF THIS OBJECT ONLY CALLS THE BUILD FUNCTION ONCE
        self.build_global_dataset_dict()
    def build_global_dataset_dict(self):
        with open(self.all_snippet_file) as fd:
            for line in fd:
                line = line.strip().split()
                snippet = line[0]
                category = line[1]
                if category not in self.dataset_dict:
                    self.dataset_dict[category]=OrderedDict()
                # get the video path component
                snippet_path_component_shared=snippet.replace(
                    self.dataset_training_root_path+'/','',1)
                video_id = snippet_path_component_shared[:-3]
                snippet_id_of_single_video = snippet_path_component_shared[-3:]
                if video_id not in self.dataset_dict[category]:
                    self.dataset_dict[category][video_id]=[]
                self.dataset_dict[category][video_id].append(snippet_id_of_single_video)
        print('[INFO:] Global Dataset Dict Builded')
        #print(self.dataset_dict['9']['VID/train/ILSVRC2015_VID_train_0001/ILSVRC2015_train_00171'])
        #print(list(self.dataset_dict['9'].keys()))
    def report_statistics(self, output_file_num_snippets_each_video='num_snippets_each_video.txt',
                          output_file_num_video_each_cat='num_video_each_category.txt'):
        # if global dataset is not built, build it
        for cat,videos_dict in self.dataset_dict.items():
            self.num_video_in_cat_dict[cat]=len(videos_dict.keys())
        self.num_snippets_in_video = copy.deepcopy(self.dataset_dict)
        for cat,videos_dict in self.num_snippets_in_video.items():
            for video in videos_dict:
                self.num_snippets_in_video[cat][video]=len(self.num_snippets_in_video[cat][video])
        with open(output_file_num_snippets_each_video,'w') as fd:
            fd.write('cat videoID numOfSnippet\n')
            for key, val in self.num_snippets_in_video.items():
                for video, num in val.items():
                    fd.write('{} {} {}\n'.format(key,video,num))
        with open(output_file_num_video_each_cat,'w') as fd:
            for cat,num in self.num_video_in_cat_dict.items():
                fd.write('{} {}\n'.format(cat,num))

    def generate_snippets_file_of_single_category(self,category,num_video=1000):
        '''
        generate snippets file which contain only snippets of `category`;
        :param category:
        :param num_video: how many video to use; if the num > available videos, return all available videos
        :return:
        '''
        output_file_name = 'category_{}_numVideo_{}.txt.temp'.format(category,num_video)
        count=0
        with open(output_file_name,'w') as fd:
            cat_dict = self.dataset_dict[category]
            for video,snippets in cat_dict.items():
                for snippet in snippets:
                    fd.write(
                        os.path.join(self.dataset_training_root_path,video+snippet)
                        +' '+category+'\n')
                count+=1
                if count == num_video:
                    break
    def generate_snippets_of_categories(self,cat_list):
        '''
        Generate all video snippets of categories in `cat_list`
        :param cat_list:
        :return:
        '''
        def list_to_string(l):
            # ['1','2','3'] ->_1_2_3_
            res=''
            for i in l:
                res=res+'_'+i
            return res

        with open('categories'+list_to_string(cat_list)+'.txt','w') as fd:
            with open(self.all_snippet_file) as f_read:
                for line in f_read:
                    line_split_arr = line.split()
                    if line_split_arr[1] in cat_list:
                        fd.write(line)
    def count_num_frames(self, snippets_file):
        '''
        counting the num of frames for all the snippets in snippets_file
        :param cat_lst:
        :return:
        '''
        with open(snippets_file) as fd:
            sum =0
            for line in fd:
                path = line.strip().split()[0]
                sum+= len(os.listdir(path))
        print('Total Number of frames: {}'.format(sum))
        return sum
class Sampled_Infer:
    # common used method, so make it static; maybe get changed flexibly
    @staticmethod
    def find_best_matched_gt_bb(gt_bboxes, pred_bboxes,min_threshd):
        '''
            find best matched gt bbox for each pred bbox.
            :param gt_bboxes:
            :param pred_bboxes:
            :param min_threshd:
            :return: return the index of matching bbox or None if no matching is found
        '''

        def _filter_list(lst, threshd):
            truncate_idx = None
            for i, idx_iou_tuple in enumerate(lst):
                if idx_iou_tuple[1] < threshd:
                    truncate_idx = i
                    break
            if truncate_idx is not None:
                del lst[truncate_idx:]

        already_matched = [False] * len(gt_bboxes)
        result = []  # size of result is len(pred_bbox),each entry is the index of matching gtbb or None
        matrix_ious = [[(gt_idx, Accuracy_Measurement.compute_IOU(box, gt_box))
                        for gt_idx, gt_box in enumerate(gt_bboxes)]
                       for box in pred_bboxes]

        # sort in place
        matrix_ious = [sorted(per_pred_ious, key=lambda k: k[1], reverse=True)
                       for per_pred_ious in matrix_ious]
        # remove tuples whoes iou is less than threshd
        for item in matrix_ious:
            _filter_list(item, min_threshd)
        # find the best matched gt bbox
        for ious in matrix_ious:
            # ious is empty or can not find a match ---> None
            find = False
            for iou in ious:
                if not already_matched[iou[0]]:
                    result.append(iou[0])
                    already_matched[iou[0]] = True
                    find = True
                    break
            if not find:
                result.append(None)
        return result
    @staticmethod
    def compute_acc_using_prev_result(
                                      last_detect_res, # coco integer ID
                                      IOU,
                                      min_threshd,
                                      gt_dict):        # classID should be converted to coco integer ID
        # filter out all detection results whose detection scores are less than min_threshd
        idx_all_kept_detection = [ idx for idx, i in enumerate(last_detect_res['detection_scores'])
                                   if i>min_threshd]
        num_detection_last = len(idx_all_kept_detection)
        last_bb=[last_detect_res['detection_boxes'][idx] for idx in idx_all_kept_detection]
        last_class=[last_detect_res['detection_class'][idx] for idx in idx_all_kept_detection]
        # filter out all gt objects the category of which is out of COCO and ILSVRC intersection
        gt_bb=[]
        gt_cls=[]
        for idx,i in enumerate(gt_dict['detection_class']):
            if i in ILSVRC_coco_map.ILSVRC_TO_COCO.values():
                gt_bb.append(gt_dict['detection_boxes'][idx])
                gt_cls.append(gt_dict['detection_classes'][idx])
        num_detection_gt=len(gt_bb)
        if not last_class and not gt_cls:
            return -1
        if not last_class and gt_cls:
            return -2
        if last_class and not gt_cls:
            return 0
        matching_res = Sampled_Infer.find_best_matched_gt_bb(gt_bb,last_bb,IOU)
        FP=0
        TP=0
        for idx, matched_idx in enumerate(matching_res):
            if matched_idx is None:
                FP+=1
                continue
            if last_class[idx] == gt_cls[matched_idx]:
                TP+=1
            else:
                FP+=1
        return TP/(TP+FP) # denominator will never be zero
class RL_Guided_Sample:
    def __init__(self):
        pass
    class Action:
        INFER =0
        SKIP = 1
    @staticmethod
    def similarity_frames(f1,f2):
        pass
    class state:
        def __init__(self,
                     all_frames_generator,
                     compare_difference_range):
            self.all_frames_generator = all_frames_generator # set only one time for each instance
            self.backcheck_range=compare_difference_range
            self.cycle_frames = cycle(self.all_frames_generator)
            #=================================================
            self.current_frame =next(self.cycle_frames)
            self.next_frame=next(self.cycle_frames)
            self.last_infered_image=None
            self.distance_to_last_infer=0 # in [0,1,2,3,4,5]
            # the difference between current frame and the i_frame_before frame
            self.similarity_consecutive_frames = [-1 for i in range(compare_difference_range)]
            self.previous_frames =[None for i in range(compare_difference_range)]
            self.video_start = 1 # 1 yes, 0 no
        def reset(self):
            self.distance_to_last_infer=0
            self.similarity_consecutive_frames=[-1 for i in range(self.backcheck_range)]
            self.previous_frames =[None for i in range(self.backcheck_range)]
            self.video_start =1
            self.cycle_frames=cycle(self.all_frames_generator)
            self.current_frame=next(self.cycle_frames)
            self.next_frame=next(self.cycle_frames)
            self.last_infered_image=None
        def is_first_frame(self,curr_frame,last_frame):
            pass
        def set_video_starting(self):
            pass
        def update_state(self,action):

            if action==RL_Guided_Sample.Action.INFER:
                self.last_infered_image=self.next_frame
                # if do the inference, next state current frame is inferred, so the distance is zero
                self.distance_to_last_infer=0
            else:
                self.distance_to_last_infer+=1
            self.previous_frames =[self.current_frame]+self.previous_frames[1:]
            self.current_frame = self.next_frame
            self.next_frame = next(self.cycle_frames)
            self.similarity_consecutive_frames=[RL_Guided_Sample.similarity_frames(self.current_frame,i)
                                                for i in self.previous_frames]
            if self.is_first_frame(self.current_frame,self.previous_frames[0]):
                self.set_video_starting()

    @staticmethod
    def generator_next_frame(snippets_file):
        with open(snippets_file) as fd:
            for line in fd:
                snippet_path = line.split()[0]
                list_frames_paths = [os.path.join(snippet_path, i)
                                     for i in sorted(os.listdir(snippet_path))]
                for frame_path in list_frames_paths:
                    yield frame_path
    class Env:
        def __init__(self,snippets_file):
            self.state = RL_Guided_Sample.state()
            self.all_frames =RL_Guided_Sample.generator_next_frame(snippets_file)
        def reset(self):
            pass
        def step(self,action):
            pass



















if __name__=='__main__':

    if args.sel:
        if not args.output_sel_snippet_file:
            print('[Error:] Missing arg, you must specify where to save the output selected snippets file')
            exit()
        select_subset_from_train_set(args.sel[0],args.sel[1],args.sel[2],args.output_sel_snippet_file[0])
    if args.single_acc:
        pass
    if args.ILSVRC_EVA:
        if not config:
            print('[Error:] Import config file failed')
            exit()
        evaluator = ILSVRC_Evaluation(snippets_file_path=config.EVALUATION['snippets_file_path'],
                                      dataset_root_path=config.EVALUATION['dataset_root_dir_name'],
                                      model_name=config.EVALUATION['model_name'],
                                      purpose=config.EVALUATION['purpose_name'],
                                      res_dir_name=config.EVALUATION['res_dir_name'],
                                      evaluation_res_dir=config.EVALUATION['evaluation_res_dir'],
                                      threshds_lst=config.EVALUATION['iou_thresholds'],
                                      min_detection_score=config.EVALUATION['min_detection_score'],
                                      write_or_append=config.EVALUATION['write_or_append'])
        evaluator.compute_acc_per_video_per_snippet()
        #evaluator.compute_per_video_per_snippet_speed_and_count()
        #evaluator.draw_per_frame_acc()
    if args.dataset_management:
        reportor = Dataset_Management_ILSVRC(args.dataset_management[0],
                                             args.dataset_management[1])
        #reportor.report_statistics()
        #reportor.generate_snippets_of_categories(['6','7'])
        reportor.generate_snippets_file_of_single_category('4',3)
        reportor.generate_snippets_file_of_single_category('5',3)
        reportor.generate_snippets_file_of_single_category('19',3)
        #reportor.count_num_frames('categories_6_7_9_numVideo_5.txt')
