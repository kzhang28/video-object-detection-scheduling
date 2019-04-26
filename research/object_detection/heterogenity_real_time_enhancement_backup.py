import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
#import zipfile
import itertools
from pathlib import Path
from distutils.version import StrictVersion
from collections import defaultdict
#from io import StringIO
#from matplotlib import pyplot as plt
from PIL import Image
sys.path.append("..")
from object_detection.utils import ops as utils_ops
import time
if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')
from utils import label_map_util
from utils import visualization_utils as vis_util
import json
from kuo_experiment.accuracy_measurement import Accuracy_Measurement
from kuo_experiment import config


# runtime control
PRINT_IMAGE_SIZE= False
WHETHER_CYCLE=False
DUMP_RES = True # Whether write detection result to files
DRAW_BD_BOX = False
#====================================================================================================
# Separation Line: Above are all configurations
#====================================================================================================
#NUM_TEST_IMAGES = len(os.listdir(PATH_TO_TEST_IMAGES_DIR))-1 # exclude image_info.txt file
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
def write_single_detect_result_to_file(output_dict,image_file_path,outdir=OUTPUT_FILE_DIR):
  _,image_file_name = os.path.split(image_file_path)
  json_file_name= os.path.join(outdir,os.path.splitext(image_file_name)[0]+'.json')
  # filter out detections whose scores are less than 0.2, this method also converts all np.array to list
  output_dict = Accuracy_Measurement.get_final_detection_result(output_dict)
  with open(json_file_name, 'w') as f:
    json.dump(output_dict, f)

def download_model():
  DOWNLOAD_BASE=config.DOWNLOAD_BASE
  MODEL_FILE=config.INPUT_OUTPUT['MODEL_NAME']+'.tar.gz'
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
  image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
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
  #print('Exe Time of one image:{:8.3f}'.format(ttt2-ttt1))
  # all outputs are float32 numpy arrays, so convert types as appropriate
  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict[
    'detection_classes'][0].astype(np.uint8)
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
  output_dict['detection_scores'] = output_dict['detection_scores'][0]
  if 'detection_masks' in output_dict:
    output_dict['detection_masks'] = output_dict['detection_masks'][0]
  output_dict['response_time']= '{:8.3f}'.format(ttt2-ttt1)
  output_dict['process_time']= '{:8.3f}'.format(tt2-tt1)
  return output_dict

def run_inference(graph,test_image_path_list):
  '''
  run inference on frames in `test_image_path_list`

  :param graph: tf_graph
  :param test_image_path_list: list: [the path to a single image]
  :return:
  '''
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
    with tf.Session(config=sess_config) as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
            tensor_name)
      if WHETHER_CYCLE:
        test_image_path_list = itertools.cycle(test_image_path_list)
      for image_path in test_image_path_list:
        image = Image.open(image_path)
        image_np = load_image_into_numpy_array(image)
        if PRINT_IMAGE_SIZE:
          print(np.shape(image_np))
        image_np_expanded = np.expand_dims(image_np, axis=0)
        output_dict = run_sess(image_np,sess,tensor_dict)
        if DUMP_RES:
          write_single_detect_result_to_file(output_dict,image_path)
        if DRAW_BD_BOX:
          vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
def create_dir(base_dir,dir_path):
  '''
  Create a dir recursively: e.g., `base_dir`=/bar/foo, `dir_path`=/baz/qux, then this
  func will create a path /bar/foo/baz/quz

  :param base_dir: base dir
  :param dir_path: dir path starting from base
  :return:
  '''
  full_path = os.path.join(base_dir,dir_path)
  Path(full_path).mkdir(parents=True,exist_ok=True)
def inference_main():
  model = config.INPUT_OUTPUT['MODEL_NAME']
  print('[INFO]:Using model: {}'.format(model))
# Path to frozen detection graph. This is the actual model that is used for the object detection.
  snippet_dir_file=config.INPUT_OUTPUT['SNIPPETS_FILE_PATH']
  dataset_root=config.INPUT_OUTPUT['DATASET_ROOT_DIR']
  res_dir_name=config.INPUT_OUTPUT['RESULT_DIR_NAME']
  download_model()
  PATH_TO_FROZEN_GRAPH = model + '/frozen_inference_graph.pb'
  loaded_graph = load_frozen_graph(PATH_TO_FROZEN_GRAPH)

  with open(snippet_dir_file) as f:
    for line in f:
      snippet_path = line.split()[0]
      path_structure = snippet_path.replace(dataset_root,'',count=1)
      result_dir_base = os.path.join(dataset_root,res_dir_name)
      # create a dir to hold the detection result for a single snippet
      create_dir(result_dir_base,path_structure)
      test_frames_paths=[os.path.join(snippet_path,i)
                    for i in sorted(os.listdir(snippet_path))]
      run_inference(loaded_graph,test_frames_paths)





if __name__=="__main__":
    pass



