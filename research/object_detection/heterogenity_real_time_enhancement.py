import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
#import zipfile
from pathlib import Path
from distutils.version import StrictVersion
#from collections import defaultdict
#from io import StringIO
#from matplotlib import pyplot as plt
from PIL import Image
sys.path.append("..")
from object_detection.utils import ops as utils_ops
import time
if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')
from utils import label_map_util
#from utils import visualization_utils as vis_util
import json
import re
import logging
from kuo_experiment.accuracy_measurement import Accuracy_Measurement
import importlib
import psutil
import argparse
import subprocess
#====================================================================================================
# Command line args parse
parser = argparse.ArgumentParser()
parser.add_argument('-config_file',nargs=1,
                    metavar=('Config file to use by this script (w/o .py)'),help="Specify config file")
args = parser.parse_args()
if not args.config_file:
  print('[Error:] Must Specify Config File')
  exit()
config_file_woExtension=args.config_file[0].replace('.py','') # remove '.py'
config = importlib.import_module('kuo_experiment.{}'.format(config_file_woExtension))
#====================================================================================================
logger=logging.getLogger(__name__)
if config.LOGGING['handler']=='file':
  log_file_dir= os.path.dirname(config.LOGGING['log_file_path'])
  Path(log_file_dir).mkdir(parents=True,exist_ok=True)
  handler = logging.FileHandler(config.LOGGING['log_file_path'],mode=config.LOGGING['write_or_append'])
elif config.LOGGING['handler']=='stream':
  handler=logging.StreamHandler()
else:
  handler=None
formatter=logging.Formatter()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(config.LOGGING['level'])

PS_Process = psutil.Process()

PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

pattern = re.compile('^/')
# remove possible leading slash
rm_leading_slash=lambda s:pattern.sub('',s)
def record_resource_usage_cpu():
  cpu_percent=PS_Process.cpu_percent() # float
  num_threads=PS_Process.num_threads()
  logger.info('CPU_Percent:{}|Num_Threads:{}'.format(cpu_percent,num_threads))
def record_resource_usage_gpu():
    gpu_stat = subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu', '--format=csv'],
                              stdout=subprocess.PIPE).stdout.decode('utf-8')
    lines=gpu_stat.split(os.linesep)
    gpu_percent = float(lines[1][:-1])
    logger.info('{}'.format(gpu_percent))


def download_model(model_name):
  DOWNLOAD_BASE=config.DOWNLOAD_BASE
  MODEL_FILE=model_name+'.tar.gz'
  if not os.path.exists(MODEL_FILE):
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
  tar_file = tarfile.open(MODEL_FILE)
  for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
      tar_file.extract(file, os.getcwd())

# load a frozen TF model in memory
def load_frozen_graph(path_to_frozen_graph):
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
  return detection_graph


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_sess(image,sess,tensor_dict):
  # prepare work
  image_tensor=sess.graph.get_tensor_by_name('image_tensor:0')
  if 'detection_masks' in tensor_dict:
    # The following processing is only for single image
    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
      detection_masks, detection_boxes, image.shape[0], image.shape[1])
    detection_masks_reframed = tf.cast(
      tf.greater(detection_masks_reframed, 0.5), tf.uint8)
    # Follow the convention by adding back the batch dimension
    tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
  # Run inference
  tt1=time.process_time()
  ttt1 = time.perf_counter()
  output_dict = sess.run(tensor_dict,
                         feed_dict={image_tensor: np.expand_dims(image, 0)})
  ttt2 = time.perf_counter()
  tt2=time.process_time()
  logger.info('Response Time:{:8.3f}|Process Time:{:8.3f}'.format(
    ttt2-ttt1,tt2-tt1))
  # all outputs are float32 numpy arrays, so convert types as appropriate
  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict[
    'detection_classes'][0].astype(np.uint8)
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
  output_dict['detection_scores'] = output_dict['detection_scores'][0]
  if 'detection_masks' in output_dict:
    output_dict['detection_masks'] = output_dict['detection_masks'][0]
  # run time metric
  output_dict['response_time']= '{:8.3f}'.format(ttt2-ttt1)
  output_dict['process_time']= '{:8.3f}'.format(tt2-tt1)
  return output_dict

def create_dir(base_dir,purpose,model_name,dir_path):
  '''
  create dir for holding inference result

  :param base_dir:
  :param model_name:
  :param dir_path:
  :return: the path of result snippet dir
  '''
  full_path = os.path.join(base_dir,purpose,model_name,rm_leading_slash(dir_path))
  Path(full_path).mkdir(parents=True,exist_ok=True)
  return full_path
def create_sess(graph):
  with graph.as_default():
    sess_config = None
    if config.RUNTIME_PARA['USE_CPU_CONFIG']:
      sess_config = tf.ConfigProto()
      sess_config.intra_op_parallelism_threads=config.RUNTIME_PARA['INTRA_OP_PARALLELISM_THREADS']
      sess_config.inter_op_parallelism_threads=config.RUNTIME_PARA['INTER_OP_PARALLELSIM_THREADS']
    if config.RUNTIME_PARA['USE_GPU_CONFIG']:
        if not config.RUNTIME_PARA['USE_CPU_CONFIG']:
          sess_config = tf.ConfigProto()
        sess_config.log_device_placement = config.RUNTIME_PARA['LOG_DEVICE_PLACEMENT']
        sess_config.gpu_options.allow_growth = config.RUNTIME_PARA['ALLOW_GROWTH']
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
      tensor_name = key + ':0'
      if tensor_name in all_tensor_names:
        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
    #image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
  return tf.Session(graph=graph,config=sess_config), tensor_dict

def inference_main():
  model = config.INPUT_OUTPUT['MODEL_NAME']
  logger.info('[INFO]:Using model: {}'.format(model))
  snippets_file_path=config.INPUT_OUTPUT['SNIPPETS_FILE_PATH']
  dataset_root=config.INPUT_OUTPUT['DATASET_ROOT_DIR']
  res_dir_name=config.INPUT_OUTPUT['RESULT_DIR_NAME']
  purpose =config.INPUT_OUTPUT['PURPOSE_NAME']
  download_model(model)
  path_frozen_graph = model + '/frozen_inference_graph.pb'
  loaded_graph = load_frozen_graph(path_frozen_graph)
  sess,tensor_dict = create_sess(loaded_graph)
  with open(snippets_file_path) as f:
    for line in f:
      snippet_path = line.split()[0]
      snippet_path_component_shared = snippet_path.replace(dataset_root,'',1)
      result_dir_base = os.path.join(dataset_root,res_dir_name)
      # create a dir to hold the detection result for a single snippet
      res_snippet_path=create_dir(result_dir_base,purpose,model,snippet_path_component_shared)
      list_frames_paths=[os.path.join(snippet_path,i) for i in sorted(os.listdir(snippet_path))]
      for frame_path in list_frames_paths:
        image = Image.open(frame_path)
        image_np = load_image_into_numpy_array(image)
        output_dict = run_sess(image_np,sess,tensor_dict)
        # filter out low score detection
        Accuracy_Measurement.get_final_detection_result(output_dict,
                                                        min_score_threshold=config.MIN_THRESHOLD_TO_KEEP)
        if config.WHETHER_RECORD_RESOURCE_USAGE:
          record_resource_usage_cpu()
        if config.WHETHER_RECORD_GPU:
          record_resource_usage_gpu()
        if config.WHETHER_DUMP_DETECTION_RESULT:
          image_name = os.path.basename(frame_path)
          with open(os.path.join(res_snippet_path,
                                 os.path.splitext(image_name)[0]+'.json'),'w') as f:
            json.dump(output_dict,f)
  logger.info('Done of detection task!')

if __name__=="__main__":
    # output config.py content to log file
    with open('./kuo_experiment/{}'.format(args.config_file[0])) as f:
      config_content = f.read()
    logger.info(config_content)
    # begin object detection task
    inference_main()

