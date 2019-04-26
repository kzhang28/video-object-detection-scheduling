# global
import logging
import os
model_dict = {
    1: 'faster_rcnn_inception_v2_coco_2018_01_28',
    2: 'ssd_mobilenet_v1_coco_2017_11_17',
    3: 'ssd_inception_v2_coco_2018_01_28',
    4: 'faster_rcnn_resnet50_coco_2018_01_28',
    5: 'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
}
# =============================INPUT OUTPUT CONFIG=======================================#
INPUT_OUTPUT = {
    'MODEL_NAME': model_dict[3],
    # File containing the absolute path to each testing snippet
    'SNIPPETS_FILE_PATH': '/home/kuo/Documents/research_experiment/edge-dnn-serving-video-analytics'
                          '/docker-image-factory/obj-detection-tensorflow/basic-tf-obj-detection'
                          '/models/research/object_detection/kuo_experiment/selected_snippets.10line.txt',
    # Dataset Root Dir. This is used to separate each snippet's path such that output result file
    # would employ the same dir structure and name rooted from `DATA_SET_ROOT_DIR`
    'DATASET_ROOT_DIR': '/home/kuo/Documents/research_experiment/edge-dnn-serving-video-analytics'
                        '/docker-image-factory/obj-detection-tensorflow/ILSVRC2015-image',
    'RESULT_DIR_NAME': 'all-result',
    'PURPOSE_NAME':'acc_speed_tradeoff'
}

# =============================RUN TIME SYSTEM CONFIG=======================================#
RUNTIME_PARA = {
    # CPU related para
    'INTRA_OP_PARALLELISM_THREADS': 4,
    'INTER_OP_PARALLELSIM_THREADS': 2,
    # GPU-related para
    'ALLOW_GROWTH': True,
    'LOG_DEVICE_PLACEMENT': True,
    # Whether enable configs
    'USE_CPU_CONFIG': False,  # false is to use default sess config
    'USE_GPU_CONFIG': False  # if USE_SESS_CONFIG is false, this para will not take effect
}
LOGGING={
    'handler':'file',#stream or file
    'level':logging.INFO,
    'log_file_path':os.path.join(INPUT_OUTPUT['DATASET_ROOT_DIR'],'all-log',
                                 INPUT_OUTPUT['MODEL_NAME']+'_log.txt') # if handler is file, this must be set
}
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
#======================================Performance Metric=================================#
# the detection result lower than this minimum score threshold will be filtered out
MIN_THRESHOLD_TO_KEEP=0.2
# Whether dump result
WHETHER_DUMP_DETECTION_RESULT=True
WHETHER_RECORD_RESOURCE_USAGE=True

EVALUATION={
    'dataset':'',
    'snippets_file_path':'/home/kuo/Documents/research_experiment/edge-dnn-serving-video-analytics'
                         '/docker-image-factory/obj-detection-tensorflow/basic-tf-obj-detection'
                         '/models/research/object_detection/kuo_experiment/selected_snippets.10line.txt',
    'dataset_root_dir_name':'/home/kuo/Documents/research_experiment/edge-dnn-serving-video-analytics'
                            '/docker-image-factory/obj-detection-tensorflow/ILSVRC2015-image',
    'res_dir_name':'all-result', # res dir holds all result
    'model_name':model_dict[3],
    'purpose_name':'acc_speed_tradeoff',
    'iou_thresholds':[0.5,0.75,0.95]
}



