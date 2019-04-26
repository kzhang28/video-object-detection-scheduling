
import os
import argparse
import json
import xml.etree.ElementTree as ET
from collections import namedtuple
from collections import defaultdict
from accuracy_measurement import Accuracy_Measurement
import numpy as np
import ILSVRC_coco_map
import config
import copy

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
                 threshds_lst):
        self.Metric_Tuple = namedtuple('metric', ['snippet_path','accuracies'] )
        self.global_FP= 0
        self.global_TP= 0
        self.global_FN= 0
        self.cat_intersection = ILSVRC_coco_map.ILSVRC_TO_COCO.values()# view object
        self.threshds=threshds_lst
        self.snippets_path=snippets_file_path
        self.dataset_root_path=dataset_root_path
        self.model_name=model_name
        self.purpose=purpose
        # since this class is specific for ILSVRC dataset, hardcode path component names
        self.dataset_training_root_path = os.path.join(self.dataset_root_path,'Data')
        self.dataset_annotation_root_path = os.path.join(self.dataset_root_path,'Annotations')
        self.dataset_result_root_path = os.path.join(self.dataset_root_path,'all-result',self.purpose,
                                                     self.model_name,'Data')
        self.metric=[] # [ Metric_Tuple ], namely [(snippet_path_id,[])]


    def _find_best_matched_gtBbox(self,gt_bboxes,pred_bboxes,min_threshd):
        '''
        find best matched gt bbox for each pred bbox.
        :param gt_bboxes:
        :param pred_bboxes:
        :param min_threshd:
        :return:
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
        compute average single frame accuracy over different IoU thresholds
        :param pred_path:
        :param gt_path:
        :return: single frame accuracy (averaged over multiple IoU threshold)
        '''
        # read from frame result and ground truth files
        tree = ET.parse(gt_path)
        root = tree.getroot()
        gt_width = int(root[3][0].text)
        gt_height = int(root[3][1].text)
        gt_object_element=root.findall('object')
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
        pred_num_detection=len(pred_bd)
        # filter out pred detections whose category are not in the intersection of coco and ILSVRC
        temp=[ (bd,cls) for bd,cls in zip(pred_bd,pred_classes)
               if cls in self.cat_intersection]
        if temp:
            pred_bd,pred_classes = list(zip(*temp))
        else:
            pred_bd=[]
            pred_classes=[]

        # Note be careful when use x,y=list(zip(*[_])), the return is single element tuple
        # compute accuracy
        if not pred_classes and not gt_object_element:
            return -1
        elif not pred_classes and gt_object_element:
            self.global_FN = self.global_FN+gt_num_detection
            return -1
        elif pred_classes and not gt_object_element:
            self.global_FP=self.global_FP+pred_num_detection
            return 0
        else:
            accuracies=[] # holding accuracy values at different threshd
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
        with open(self.snippets_path) as fd:
            for line in fd:
                snippet_path = line.strip().split()[0]
                snippet_path_component_shared = snippet_path.replace(
                    self.dataset_training_root_path+'/','',1)
                self.metric.append(self.Metric_Tuple(snippet_path_component_shared,[]))
                frames = sorted(os.listdir(snippet_path))
                for frame in frames:
                    frame_name,_ = os.path.splitext(frame)
                    gt_path = os.path.join(self.dataset_annotation_root_path,
                                           snippet_path_component_shared,frame_name+'.xml')
                    pred_path =os.path.join(self.dataset_result_root_path,
                                            snippet_path_component_shared,frame_name+'.json')
                    acc = self.compute_accuracy_single_frame(pred_path,gt_path)
                    self.metric[-1].accuracies.append(acc)
    def driver(self):
        # average accuracy
        self.compute_accuracy_all()
        per_snippet_acc = []
        for snippet in self.metric:
            sum = 0
            count =0
            for acc in snippet.accuracies:
                if acc != -1:
                    sum+=acc
                    count+=1
            per_snippet_acc.append(sum/count)
        global_acc = np.average(per_snippet_acc)
        print('Per snippet accuracy:',per_snippet_acc)
        print('Average accuracy',global_acc)


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
    def build_global_dataset_dict(self):
        with open(self.all_snippet_file) as fd:
            for line in fd:
                line = line.strip().split()
                snippet = line[0]
                category = line[1]
                if category not in self.dataset_dict:
                    self.dataset_dict[category]=defaultdict(list)
                # get the video path component
                snippet_path_component_shared=snippet.replace(self.dataset_training_root_path+'/','',1)
                video_id = snippet_path_component_shared[:-3]
                snippet_id_of_single_video = snippet_path_component_shared[-3:]
                self.dataset_dict[category][video_id].append(snippet_id_of_single_video)
        print('[INFO:] Global Dataset Dict Builded')
    def report_statistics(self, output_file_num_snippets_each_video='num_snippets_each_video.txt',
                          output_file_num_video_each_cat='num_video_each_category.txt'):
        # if global dataset is not built, build it
        if not self.dataset_dict:
            self.build_global_dataset_dict()
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
        output_file_name = 'category_{}_numVideo_{}.txt'.format(category,num_video)
        count=0
        with open(output_file_name,'w') as fd:
            cat_dict = self.dataset_dict[category]
            for video,snippets in cat_dict:
                for snippet in snippets:
                    fd.write(os.path.join(self.dataset_training_root_path,video,snippet)+'\n')
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
                        fd.write(line+'\n')













if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sel',nargs=3,metavar=('Category_file_path','category_pointer_file_dir',
                            'train_set_dir'),help="To select subset from training set")
    parser.add_argument('-output_sel_snippet_file',nargs=1,default='selected_snippets.txt',
                        help='File name for store all selected snippets')
    parser.add_argument('-single_acc',nargs=2,metavar=('res_path','gt_path'),
                        help='Compute single frame accuracy')
    parser.add_argument('-ILSVRC_EVA',action='store_true',
                        help='Evaluate detection results on ILSVRC dataset')
    parser.add_argument('-report_dataset_stat',nargs=2,metavar=('all_snippets_file', 'dataset_root')
                        ,help='Report statistics of dataset:# of snippet in each video;# of video in each cat')
    args = parser.parse_args()
    if args.sel:
        select_subset_from_train_set(args.sel[0],args.sel[1],args.sel[2],args.output_sel_snippet_file[0])
    if args.single_acc:
        pass
    if args.ILSVRC_EVA:
        evaluator = ILSVRC_Evaluation(snippets_file_path=config.EVALUATION['snippets_file_path'],
                                      dataset_root_path=config.EVALUATION['dataset_root_dir_name'],
                                      model_name=config.EVALUATION['model_name'],
                                      purpose=config.EVALUATION['purpose_name'],
                                      threshds_lst=config.EVALUATION['iou_thresholds'])
        evaluator.driver()
    if args.report_dataset_stat:
        reportor = Dataset_Management_ILSVRC(args.report_dataset_stat[0],
                                             args.report_dataset_stat[1])
        #reportor.report_statistics()
        reportor.generate_snippets_of_categories(['6','7'])

