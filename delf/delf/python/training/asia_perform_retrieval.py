# Lint as: python3
# Copyright 2020 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Performs DELG-based image retrieval on Revisited Oxford/Paris datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import pandas as pd

from delf import datum_io
from delf.python.detect_to_retrieve import dataset
from delf.python.detect_to_retrieve import image_reranking

FLAGS = flags.FLAGS

flags.DEFINE_string('query_file_path', '/tmp/query.csv','CSV file for Asia Dataset')
flags.DEFINE_string('index_file_path', '/tmp/index.csv','index file for Asia Dataset')
flags.DEFINE_string('query_features_dir', '/tmp/features/query',
                    'Directory where query DELG features are located.')
flags.DEFINE_string('index_features_dir', '/tmp/features/index',
                    'Directory where index DELG features are located.')
flags.DEFINE_boolean(
    'use_geometric_verification', False,
    'If True, performs re-ranking using local feature-based geometric '
    'verification.')
flags.DEFINE_float(
    'local_descriptor_matching_threshold', 1.0,
    'Optional, only used if `use_geometric_verification` is True. '
    'Threshold below which a pair of local descriptors is considered '
    'a potential match, and will be fed into RANSAC.')
flags.DEFINE_float(
    'ransac_residual_threshold', 20.0,
    'Optional, only used if `use_geometric_verification` is True. '
    'Residual error threshold for considering matches as inliers, used in '
    'RANSAC algorithm.')
flags.DEFINE_boolean(
    'use_ratio_test', False,
    'Optional, only used if `use_geometric_verification` is True. '
    'Whether to use ratio test for local feature matching.')
flags.DEFINE_string(
    'output_dir', '/tmp/retrieval',
    'Directory where retrieval output will be written to. A file containing '
    "metrics for this run is saved therein, with file name 'metrics.txt'.")

# Extensions.
_DELG_GLOBAL_EXTENSION = '.delg_global'
_DELG_LOCAL_EXTENSION = '.delg_local'

# Precision-recall ranks to use in metric computation.
_PR_RANKS = (1, 5, 10)

# Pace to log.
_STATUS_CHECK_LOAD_ITERATIONS = 50

# Output file names.
_METRICS_BF_FILENAME = 'metrics_bf.txt'
_METRICS_AF_FILENAME = 'metrics_af.txt'

def ComputeMetrics(sorted_index_ids, ground_truth, desired_pr_ranks):
  """Computes metrics for retrieval results on the Revisited datasets.

  If there are no valid ground-truth index images for a given query, the metric
  results for the given query (`average_precisions`, `precisions` and `recalls`)
  are set to NaN, and they are not taken into account when computing the
  aggregated metrics (`mean_average_precision`, `mean_precisions` and
  `mean_recalls`) over all queries.

  Args:
    sorted_index_ids: Integer NumPy array of shape [#queries, #index_images].
      For each query, contains an array denoting the most relevant index images,
      sorted from most to least relevant.
    ground_truth: List containing ground-truth information for dataset. Each
      entry is a dict corresponding to the ground-truth information for a query.
      The dict has keys 'ok' and 'junk', mapping to a NumPy array of integers.
    desired_pr_ranks: List of integers containing the desired precision/recall
      ranks to be reported. Eg, if precision@1/recall@1 and
      precision@10/recall@10 are desired, this should be set to [1, 10]. The
      largest item should be <= #index_images.

  Returns:
    mean_average_precision: Mean average precision (float).
    mean_precisions: Mean precision @ `desired_pr_ranks` (NumPy array of
      floats, with shape [len(desired_pr_ranks)]).
    mean_recalls: Mean recall @ `desired_pr_ranks` (NumPy array of floats, with
      shape [len(desired_pr_ranks)]).
    average_precisions: Average precision for each query (NumPy array of floats,
      with shape [#queries]).
    precisions: Precision @ `desired_pr_ranks`, for each query (NumPy array of
      floats, with shape [#queries, len(desired_pr_ranks)]).
    recalls: Recall @ `desired_pr_ranks`, for each query (NumPy array of
      floats, with shape [#queries, len(desired_pr_ranks)]).

  Raises:
    ValueError: If largest desired PR rank in `desired_pr_ranks` >
      #index_images.
  """
  num_queries, num_index_images = sorted_index_ids.shape
  num_desired_pr_ranks = len(desired_pr_ranks)

  sorted_desired_pr_ranks = sorted(desired_pr_ranks)

  if sorted_desired_pr_ranks[-1] > num_index_images:
    raise ValueError(
        'Requested PR ranks up to %d, however there are only %d images' %
        (sorted_desired_pr_ranks[-1], num_index_images))

  # Instantiate all outputs, then loop over each query and gather metrics.
  mean_average_precision = 0.0
  mean_precisions = np.zeros([num_desired_pr_ranks])
  mean_recalls = np.zeros([num_desired_pr_ranks])
  average_precisions = np.zeros([num_queries])
  precisions = np.zeros([num_queries, num_desired_pr_ranks])
  recalls = np.zeros([num_queries, num_desired_pr_ranks])
  num_empty_gt_queries = 0
  for i in range(num_queries):
    index_ground_truth = np.array(ground_truth[i])

    if not index_ground_truth.size:
      average_precisions[i] = float('nan')
      precisions[i, :] = float('nan')
      recalls[i, :] = float('nan')
      num_empty_gt_queries += 1
      continue

    positive_ranks = np.arange(num_index_images)[np.in1d(sorted_index_ids[i], 
                                                        index_ground_truth)]

    average_precisions[i] = dataset.ComputeAveragePrecision(positive_ranks)
    precisions[i, :], recalls[i, :] = dataset.ComputePRAtRanks(positive_ranks, desired_pr_ranks)

    mean_average_precision += average_precisions[i]
    mean_precisions += precisions[i, :]
    mean_recalls += recalls[i, :]

  # Normalize aggregated metrics by number of queries.
  num_valid_queries = num_queries - num_empty_gt_queries
  mean_average_precision /= num_valid_queries
  mean_precisions /= num_valid_queries
  mean_recalls /= num_valid_queries

  return (mean_average_precision, mean_precisions, mean_recalls,
          average_precisions, precisions, recalls)

def _ReadDelgGlobalDescriptors(input_dir, image_list):
  """Reads DELG global features.

  Args:
    input_dir: Directory where features are located.
    image_list: List of image names for which to load features.

  Returns:
    global_descriptors: NumPy array of shape (len(image_list), D), where D
      corresponds to the global descriptor dimensionality.
  """
  num_images = len(image_list)
  global_descriptors = []
  print('Starting to collect global descriptors for %d images...' % num_images)
  start = time.time()
  for i in range(num_images):
    if i > 0 and i % _STATUS_CHECK_LOAD_ITERATIONS == 0:
      elapsed = (time.time() - start)
      print('Reading global descriptors for image %d out of %d, last %d '
            'images took %f seconds' %
            (i, num_images, _STATUS_CHECK_LOAD_ITERATIONS, elapsed))
      start = time.time()

    descriptor_filename = image_list[i] + _DELG_GLOBAL_EXTENSION
    descriptor_fullpath = os.path.join(input_dir, descriptor_filename)
    global_descriptors.append(datum_io.ReadFromFile(descriptor_fullpath))

  return np.array(global_descriptors)


def _read_query_csv_file(dataset_file_path):
  query_list = []
  ground_truth_list = []
  with tf.io.gfile.GFile(dataset_file_path, 'rb') as csv_file:
    df = pd.read_csv(csv_file)
    for index, row in df.iterrows():
      query_name = row['id']
      ground_truth = row['result'].split(' ')
      query_list.append(query_name)
      ground_truth_list.append(ground_truth)

  return query_list, ground_truth_list

def _read_index_csv_file(dataset_file_path):
  index_list = []
  with tf.io.gfile.GFile(dataset_file_path, 'rb') as csv_file:
    df = pd.read_csv(csv_file)

    for index, row in df.iterrows():
      index_name = row['id']
      index_list.append(index_name)

  return index_list

def main(argv):
  if len(argv) > 1:
    raise RuntimeError('Too many command-line arguments.')

  # Parse dataset to obtain query/index images, and ground-truth.
  print('Parsing dataset...')
  query_list, ground_truth = _read_query_csv_file(FLAGS.query_file_path)
  index_list = _read_index_csv_file(FLAGS.index_file_path)
  num_query_images = len(query_list)
  num_index_images = len(index_list)
  
  id_ground_truth = []
  for i in range(num_query_images):
    id_gt = []
    for gt in ground_truth[i]:
      if  gt not in index_list:
        continue
      else:
        id_gt.append(index_list.index(gt))
    id_ground_truth.append(id_gt)

  id_ground_truth = np.array(id_ground_truth)
  print(type(id_ground_truth))

  print('Done! Found %d queries and %d index images' %
        (num_query_images, num_index_images))

  # Read global features.
  query_global_features = _ReadDelgGlobalDescriptors(FLAGS.query_features_dir, query_list)
  index_global_features = _ReadDelgGlobalDescriptors(FLAGS.index_features_dir, index_list)

  # Compute similarity between query and index images, potentially re-ranking
  # with geometric verification.
  ranks_before_gv = np.zeros([num_query_images, num_index_images], dtype='int32')
  if FLAGS.use_geometric_verification:
    ranks_after_gv = np.zeros([num_query_images, num_index_images], dtype='int32')

  total_query_time = 0
  for i in range(num_query_images):
    print('Performing retrieval with query %d (%s)...' % (i, query_list[i]))
    start = time.time()

    # Compute similarity between global descriptors.
    similarities = np.dot(index_global_features, query_global_features[i])
    ranks_before_gv[i] = np.argsort(-similarities)
    print(f'Global rank {ranks_before_gv[i][:20]}')
    # Re-rank using geometric verification.
    if FLAGS.use_geometric_verification:
      ranks_after_gv[i] = image_reranking.RerankByGeometricVerification(
          input_ranks=ranks_before_gv[i],
          initial_scores=similarities,
          query_name=query_list[i],
          index_names=index_list,
          query_features_dir=FLAGS.query_features_dir,
          index_features_dir=FLAGS.index_features_dir,
          junk_ids=[],
          local_feature_extension=_DELG_LOCAL_EXTENSION,
          ransac_seed=0,
          descriptor_matching_threshold=FLAGS.local_descriptor_matching_threshold,
          ransac_residual_threshold=FLAGS.ransac_residual_threshold,
          use_ratio_test=FLAGS.use_ratio_test)
      print(f'local_rank {ranks_after_gv[i][:20]}')
      

    elapsed = (time.time() - start)
    total_query_time += elapsed
    print('Done! Retrieval for query %d took %f seconds' % (i, elapsed))

  average_query_time = total_query_time / num_query_images
  # Create output directory if necessary.
  if not tf.io.gfile.exists(FLAGS.output_dir):
    tf.io.gfile.makedirs(FLAGS.output_dir)

  # global_path = os.path.join(FLAGS.output_dir, 'global_rank.txt')
  # np.savetxt(global_path, ranks_before_gv.astype(np.int32))
  # if FLAGS.use_geometric_verification:
  #   local_path = os.path.join(FLAGS.output_dir, 'local_rank.txt')
  #   np.savetxt(global_path, ranks_after_gv.astype(np.int32))

  # Write total time to query all query image set
  write_time_path = os.path.join(FLAGS.output_dir, 'query_time.txt')
  with tf.io.gfile.GFile(write_time_path, 'w') as f:
    f.write(f'Total query time: {total_query_time} seconds\n')
    f.write(f'Avarage query time: {average_query_time} seconds')

  # Write metric to file
  metric_bf_path = os.path.join(FLAGS.output_dir, _METRICS_BF_FILENAME)
  metrics = ComputeMetrics(ranks_before_gv, id_ground_truth, _PR_RANKS)
  with tf.io.gfile.GFile(metric_bf_path, 'w') as f:
      f.write('mAP={}\n  mP@k{} {}\n  mR@k{} {}\n'.format(
        np.around(metrics[0] * 100, decimals=2),
        np.array(_PR_RANKS), np.around(metrics[1] * 100, decimals=2),
        np.array(_PR_RANKS), np.around(metrics[2] * 100, decimals=2)))

  if FLAGS.use_geometric_verification:
    metrics_a = ComputeMetrics(ranks_after_gv, id_ground_truth, _PR_RANKS)
    metric_af_path = os.path.join(FLAGS.output_dir, _METRICS_AF_FILENAME)
    with tf.io.gfile.GFile(metric_af_path, 'w') as f:
      f.write('mAP={}\n  mP@k{} {}\n  mR@k{} {}\n'.format(
        np.around(metrics_a[0] * 100, decimals=2),
        np.array(_PR_RANKS), np.around(metrics_a[1] * 100, decimals=2),
        np.array(_PR_RANKS), np.around(metrics_a[2] * 100, decimals=2)))

if __name__ == '__main__':
  app.run(main)
