from collections import namedtuple
DetectionModel = namedtuple('DetectionModel', ['name', 'url', 'extract_dir'])

MODELS = {
    'ssd_mobilenet_v1_coco': DetectionModel(
        'ssd_mobilenet_v1_coco',
        'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz',
        'ssd_mobilenet_v1_coco_2018_01_28',
    ),
    'ssd_mobilenet_v2_coco': DetectionModel(
        'ssd_mobilenet_v2_coco',
        'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz',
        'ssd_mobilenet_v2_coco_2018_03_29',
    ),
    'ssd_inception_v2_coco': DetectionModel(
        'ssd_inception_v2_coco',
        'http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz',
        'ssd_inception_v2_coco_2018_01_28',
    ),
    'ssd_resnet_50_fpn_coco': DetectionModel(
        'ssd_resnet_50_fpn_coco',
        'http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz',
        'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03',
    ),
    'faster_rcnn_resnet50_coco': DetectionModel(
        'faster_rcnn_resnet50_coco',
        'http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz',
        'faster_rcnn_resnet50_coco_2018_01_28',
    ),
    'faster_rcnn_nas': DetectionModel(
        'faster_rcnn_nas',
        'http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz',
        'faster_rcnn_nas_coco_2018_01_28',
    ),
    'mask_rcnn_resnet50_atrous_coco': DetectionModel(
        'mask_rcnn_resnet50_atrous_coco',
        'http://download.tensorflow.org/models/object_detection/mask_rcnn_resnet50_atrous_coco_2018_01_28.tar.gz',
        'mask_rcnn_resnet50_atrous_coco_2018_01_28',
    )
}

MODEL = 'ssd_mobilenet_v1_coco'

import tensorflow.contrib.tensorrt as trt
import tensorflow as tf

# All the following functions are defined to prepare the neural network for the TRT Optimization

def make_const6(const6_name='const6'):
    graph = tf.Graph()
    with graph.as_default():
        tf_6 = tf.constant(dtype=tf.float32, value=6.0, name=const6_name)
    return graph.as_graph_def()


def make_relu6(output_name, input_name, const6_name='const6'):
    graph = tf.Graph()
    with graph.as_default():
        tf_x = tf.placeholder(tf.float32, [10, 10], name=input_name)
        tf_6 = tf.constant(dtype=tf.float32, value=6.0, name=const6_name)
        with tf.name_scope(output_name):
            tf_y1 = tf.nn.relu(tf_x, name='relu1')
            tf_y2 = tf.nn.relu(tf.subtract(tf_x, tf_6, name='sub1'), name='relu2')

            # tf_y = tf.nn.relu(tf.subtract(tf_6, tf.nn.relu(tf_x, name='relu1'), name='sub'), name='relu2')
        # tf_y = tf.subtract(tf_6, tf_y, name=output_name)
        tf_y = tf.subtract(tf_y1, tf_y2, name=output_name)

    graph_def = graph.as_graph_def()
    graph_def.node[-1].name = output_name

    # remove unused nodes
    for node in graph_def.node:
        if node.name == input_name:
            graph_def.node.remove(node)
    for node in graph_def.node:
        if node.name == const6_name:
            graph_def.node.remove(node)
    for node in graph_def.node:
        if node.op == '_Neg':
            node.op = 'Neg'

    return graph_def


def convert_relu6(graph_def, const6_name='const6'):
    # add constant 6
    has_const6 = False
    for node in graph_def.node:
        if node.name == const6_name:
            has_const6 = True
    if not has_const6:
        const6_graph_def = make_const6(const6_name=const6_name)
        graph_def.node.extend(const6_graph_def.node)

    for node in graph_def.node:
        if node.op == 'Relu6':
            input_name = node.input[0]
            output_name = node.name
            relu6_graph_def = make_relu6(output_name, input_name, const6_name=const6_name)
            graph_def.node.remove(node)
            graph_def.node.extend(relu6_graph_def.node)

    return graph_def


def remove_node(graph_def, node):
    for n in graph_def.node:
        if node.name in n.input:
            n.input.remove(node.name)
        ctrl_name = '^' + node.name
        if ctrl_name in n.input:
            n.input.remove(ctrl_name)
    graph_def.node.remove(node)


def remove_op(graph_def, op_name):
    matches = [node for node in graph_def.node if node.op == op_name]
    for match in matches:
        remove_node(graph_def, match)


def f_force_nms_cpu(frozen_graph):
    for node in frozen_graph.node:
        if 'NonMaxSuppression' in node.name:
            node.device = '/device:CPU:0'
    return frozen_graph


def f_replace_relu6(frozen_graph):
    return convert_relu6(frozen_graph)


def f_remove_assert(frozen_graph):
    remove_op(frozen_graph, 'Assert')
    return frozen_graph


from object_detection.protos import pipeline_pb2
from object_detection import exporter

import os
import subprocess

from google.protobuf import text_format

import tensorflow as tf

INPUT_NAME = 'image_tensor'
BOXES_NAME = 'detection_boxes'
CLASSES_NAME = 'detection_classes'
SCORES_NAME = 'detection_scores'
MASKS_NAME = 'detection_masks'
NUM_DETECTIONS_NAME = 'num_detections'
FROZEN_GRAPH_NAME = 'frozen_inference_graph.pb'
PIPELINE_CONFIG_NAME = 'pipeline.config'
CHECKPOINT_PREFIX = 'model.ckpt'


def get_input_names(model):
    return [INPUT_NAME]


def get_output_names(model):
    output_names = [BOXES_NAME, CLASSES_NAME, SCORES_NAME, NUM_DETECTIONS_NAME]
    if model == 'mask_rcnn_resnet50_atrous_coco':
        output_names.append(MASKS_NAME)
    return output_names


def download_detection_model(model, output_dir='.'):
    """Downloads a pre-trained object detection model"""
    global MODELS

    model_name = model

    model = MODELS[model_name]
    subprocess.call(['mkdir', '-p', output_dir])
    tar_file = os.path.join(output_dir, os.path.basename(model.url))

    config_path = os.path.join(output_dir, model.extract_dir, PIPELINE_CONFIG_NAME)
    checkpoint_path = os.path.join(output_dir, model.extract_dir, CHECKPOINT_PREFIX)

    if not os.path.exists(os.path.join(output_dir, model.extract_dir)):
        subprocess.call(['wget', model.url, '-O', tar_file])
        subprocess.call(['tar', '-xzf', tar_file, '-C', output_dir])

        # hack fix to handle mobilenet_v2 config bug
        subprocess.call(['sed', '-i', '/batch_norm_trainable/d', config_path])

    return config_path, checkpoint_path


def build_detection_graph(config, checkpoint,
                          batch_size=1,
                          score_threshold=None,
                          iou_threshold=None,
                          force_nms_cpu=True,
                          replace_relu6=True,
                          remove_assert=True,
                          input_shape=None,
                          output_dir='.generated_model'):
    """Builds a frozen graph for a pre-trained object detection model"""

    config_path = config
    checkpoint_path = checkpoint

    # parse config from file
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, 'r') as f:
        text_format.Merge(f.read(), config, allow_unknown_extension=True)

    # override some config parameters
    if config.model.HasField('ssd'):
        config.model.ssd.feature_extractor.override_base_feature_extractor_hyperparams = True
        if score_threshold is not None:
            config.model.ssd.post_processing.batch_non_max_suppression.score_threshold = score_threshold
        if iou_threshold is not None:
            config.model.ssd.post_processing.batch_non_max_suppression.iou_threshold = iou_threshold
        if input_shape is not None:
            config.model.ssd.image_resizer.fixed_shape_resizer.height = input_shape[0]
            config.model.ssd.image_resizer.fixed_shape_resizer.width = input_shape[1]
    elif config.model.HasField('faster_rcnn'):
        if score_threshold is not None:
            config.model.faster_rcnn.second_stage_post_processing.score_threshold = score_threshold
        if input_shape is not None:
            config.model.faster_rcnn.image_resizer.fixed_shape_resizer.height = input_shape[0]
            config.model.faster_rcnn.image_resizer.fixed_shape_resizer.width = input_shape[1]

    if os.path.isdir(output_dir):
        subprocess.call(['rm', '-rf', output_dir])

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # export inference graph to file (initial)
    with tf.Session(config=tf_config) as tf_sess:
        with tf.Graph().as_default() as tf_graph:
            exporter.export_inference_graph(
                'image_tensor',
                config,
                checkpoint_path,
                output_dir,
                input_shape=[batch_size, None, None, 3]
            )

    # read frozen graph from file
    frozen_graph = tf.GraphDef()
    with open(os.path.join(output_dir, FROZEN_GRAPH_NAME), 'rb') as f:
        frozen_graph.ParseFromString(f.read())

    # apply graph modifications
    if force_nms_cpu:
        frozen_graph = f_force_nms_cpu(frozen_graph)
    if replace_relu6:
        frozen_graph = f_replace_relu6(frozen_graph)
    if remove_assert:
        frozen_graph = f_remove_assert(frozen_graph)

    # get input names
    # TODO: handle mask_rcnn
    input_names = [INPUT_NAME]
    output_names = [BOXES_NAME, CLASSES_NAME, SCORES_NAME, NUM_DETECTIONS_NAME]

    # remove temporary directory
    subprocess.call(['rm', '-rf', output_dir])

    return frozen_graph, input_names, output_names


def optimize_model(frozen_graph,
                   use_trt=True,
                   force_nms_cpu=True,
                   replace_relu6=True,
                   remove_assert=True,
                   precision_mode='FP32',
                   minimum_segment_size=2,
                   max_workspace_size_bytes=1 << 32,
                   maximum_cached_engines=100,
                   calib_images_dir=None,
                   num_calib_images=None,
                   calib_batch_size=1,
                   calib_image_shape=None,
                   output_path=None):
    """Optimizes an object detection model using TensorRT
    Optimizes an object detection model using TensorRT.  This method also
    performs pre-tensorrt optimizations specific to the TensorFlow object
    detection API models.  Please see the list of arguments for other
    optimization parameters.
    Args
    ----
        config_path: A string representing the path of the object detection
            pipeline config file.
        checkpoint_path: A string representing the path of the object
            detection model checkpoint.
        use_trt: A boolean representing whether to optimize with TensorRT. If
            False, regular TensorFlow will be used but other optimizations
            (like NMS device placement) will still be applied.
        force_nms_cpu: A boolean indicating whether to place NMS operations on
            the CPU.
        replace_relu6: A boolean indicating whether to replace relu6(x)
            operations with relu(x) - relu(x-6).
        remove_assert: A boolean indicating whether to remove Assert
            operations from the graph.
        override_nms_score_threshold: An optional float representing
            a NMS score threshold to override that specified in the object
            detection configuration file.
        override_resizer_shape: An optional list/tuple of integers
            representing a fixed shape to override the default image resizer
            specified in the object detection configuration file.
        batch_size: An integer representing the batch size
        precision_mode: A string representing the precision mode to use for
            TensorRT optimization.  Must be one of 'FP32', 'FP16', or 'INT8'.
        minimum_segment_size: An integer representing the minimum segment size
            to use for TensorRT graph segmentation.
        max_workspace_size_bytes: An integer representing the max workspace
            size for TensorRT optimization.
        maximum_cached_engines: An integer represenging the number of TRT engines
            that can be stored in the cache.
        calib_images_dir: A string representing a directory containing images to
            use for int8 calibration.
        num_calib_images: An integer representing the number of calibration
            images to use.  If None, will use all images in directory.
        calib_batch_size: An integer representing the batch size to use for calibration.
        calib_image_shape: A tuple of integers representing the height,
            width that images will be resized to for calibration.
        output_path: An optional string representing the path to save the
            optimized GraphDef to.
        display_every: print log for calibration every display_every iteration
    Returns
    -------
        A GraphDef representing the optimized model.
    """
    # apply graph modifications
    if force_nms_cpu:
        frozen_graph = f_force_nms_cpu(frozen_graph)
    if replace_relu6:
        frozen_graph = f_replace_relu6(frozen_graph)
    if remove_assert:
        frozen_graph = f_remove_assert(frozen_graph)

    # get input names
    output_names = [BOXES_NAME, CLASSES_NAME, SCORES_NAME, NUM_DETECTIONS_NAME]

    # optionally perform TensorRT optimization
    if use_trt:
        graph_size = len(frozen_graph.SerializeToString())
        num_nodes = len(frozen_graph.node)
        start_time = time.time()

        converter = trt.TrtGraphConverter(
            input_graph_def=frozen_graph,
            nodes_blacklist=output_names,
            max_workspace_size_bytes=max_workspace_size_bytes,
            precision_mode=precision_mode,
            minimum_segment_size=minimum_segment_size,
            is_dynamic_op=True,
            maximum_cached_engines=maximum_cached_engines)
        frozen_graph = converter.convert()

        end_time = time.time()
        print("graph_size(MB)(native_tf): %.1f" % (float(graph_size)/(1<<20)))
        print("graph_size(MB)(trt): %.1f" %
            (float(len(frozen_graph.SerializeToString()))/(1<<20)))
        print("num_nodes(native_tf): %d" % num_nodes)
        print("num_nodes(tftrt_total): %d" % len(frozen_graph.node))
        print("num_nodes(trt_only): %d" % len([1 for n in frozen_graph.node if str(n.op)=='TRTEngineOp']))
        print("time(s) (trt_conversion): %.4f" % (end_time - start_time))

        # perform calibration for int8 precision
        if precision_mode == 'INT8':
            # get calibration images
            if calib_images_dir is None:
                raise ValueError('calib_images_dir must be provided for INT8 optimization.')
            image_paths = glob.glob(os.path.join(calib_images_dir, '*.jpg'))
            if len(image_paths) == 0:
                raise ValueError('No images were found in calib_images_dir for INT8 calibration.')
            image_paths = image_paths[:num_calib_images]
            num_batches = len(image_paths) // calib_batch_size

            def feed_dict_fn():
                # read batch of images
                batch_images = []
                for image_path in image_paths[feed_dict_fn.index:feed_dict_fn.index+calib_batch_size]:
                    image = _read_image(image_path, calib_image_shape)
                    batch_images.append(image)
                feed_dict_fn.index += calib_batch_size
                return {INPUT_NAME+':0': np.array(batch_images)}
            feed_dict_fn.index = 0

            print('Calibrating INT8...')
            start_time = time.time()
            frozen_graph = converter.calibrate(
                fetch_names=[x + ':0' for x in output_names],
                num_runs=num_batches,
                feed_dict_fn=feed_dict_fn)
            calibration_time = time.time() - start_time
            print("time(s) (trt_calibration): %.4f" % calibration_time)

    # write optimized model to disk
    if output_path is not None:
        with open(output_path, 'wb') as f:
            f.write(frozen_graph.SerializeToString())

    return frozen_graph


def main():
    # From the training, you need the config file and the checkpoint files
    config_path = 'frozen_graph/pipeline.config'
    checkpoint_path = 'frozen_graph/model.ckpt'
    # Create frozen graph
    frozen_graph, input_names, output_names = build_detection_graph(
        config=config_path,
        checkpoint=checkpoint_path,
        score_threshold=0.3,
        iou_threshold=0.5,
        batch_size=1
    )
    # Create trt-optimized graph
    print(output_names)
    trt_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph,
        outputs=output_names,
        max_batch_size=1,
        max_workspace_size_bytes=1 << 25,
        precision_mode='FP16',
        minimum_segment_size=50
    )
    with open('trt_graph_ssd_coco_vehicle_90K.pb', 'wb') as f:
        f.write(trt_graph.SerializeToString())

if __name__== "__main__":
  main()