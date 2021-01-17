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

import io
import os
from os import listdir 
import os.path as osp
import time

from absl import app
from absl import flags
from statistics import *
import numpy as np
import tensorflow as tf
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
from skimage import feature
from skimage import measure
from skimage import transform

from delf import feature_io

from delf import datum_io
from delf.python.detect_to_retrieve import dataset
# from delf.python.detect_to_retrieve import image_reranking

FLAGS = flags.FLAGS

flags.DEFINE_string('query_file_path', '/tmp/query.csv','CSV file for Asia Dataset')
flags.DEFINE_string('index_file_path', '/tmp/index.csv','index file for Asia Dataset')
flags.DEFINE_string('query_features_dir', '/tmp/features/query',
                    'Directory where query DELG features are located.')
flags.DEFINE_string('index_features_dir', '/tmp/features/index',
                    'Directory where index DELG features are located.')
flags.DEFINE_string('query_sift_dir', '/tmp/features/query',
                    'Directory where query SIFT features are located.')
flags.DEFINE_string('index_sift_dir', '/tmp/features/query',
                    'Directory where index SIFT features are located.')
flags.DEFINE_boolean(
    'use_geometric_verification', False,
    'If True, performs re-ranking using local feature-based geometric '
    'verification.')
flags.DEFINE_boolean(
    'use_sift_verification', False,
    'If True, performs re-ranking using local sift '
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

# Extensions.
_DELF_EXTENSION = '.delf'

# Pace to log.
_STATUS_CHECK_GV_ITERATIONS = 10

# Re-ranking / geometric verification parameters.
_NUM_TO_RERANK = 20
_NUM_RANSAC_TRIALS = 1000
_MIN_RANSAC_SAMPLES = 3


def MatchFeatures(query_locations,
                  query_descriptors,
                  index_image_locations,
                  index_image_descriptors,
                  ransac_seed=None,
                  descriptor_matching_threshold=0.9,
                  ransac_residual_threshold=10.0,
                  query_im_array=None,
                  index_im_array=None,
                  query_im_scale_factors=None,
                  index_im_scale_factors=None,
                  use_ratio_test=False):
  """Matches local features using geometric verification.

  First, finds putative local feature matches by matching `query_descriptors`
  against a KD-tree from the `index_image_descriptors`. Then, attempts to fit an
  affine transformation between the putative feature corresponces using their
  locations.

  Args:
    query_locations: Locations of local features for query image. NumPy array of
      shape [#query_features, 2].
    query_descriptors: Descriptors of local features for query image. NumPy
      array of shape [#query_features, depth].
    index_image_locations: Locations of local features for index image. NumPy
      array of shape [#index_image_features, 2].
    index_image_descriptors: Descriptors of local features for index image.
      NumPy array of shape [#index_image_features, depth].
    ransac_seed: Seed used by RANSAC. If None (default), no seed is provided.
    descriptor_matching_threshold: Threshold below which a pair of local
      descriptors is considered a potential match, and will be fed into RANSAC.
      If use_ratio_test==False, this is a simple distance threshold. If
      use_ratio_test==True, this is Lowe's ratio test threshold.
    ransac_residual_threshold: Residual error threshold for considering matches
      as inliers, used in RANSAC algorithm.
    query_im_array: Optional. If not None, contains a NumPy array with the query
      image, used to produce match visualization, if there is a match.
    index_im_array: Optional. Same as `query_im_array`, but for index image.
    query_im_scale_factors: Optional. If not None, contains a NumPy array with
      the query image scales, used to produce match visualization, if there is a
      match. If None and a visualization will be produced, [1.0, 1.0] is used
      (ie, feature locations are not scaled).
    index_im_scale_factors: Optional. Same as `query_im_scale_factors`, but for
      index image.
    use_ratio_test: If True, descriptor matching is performed via ratio test,
      instead of distance-based threshold.

  Returns:
    score: Number of inliers of match. If no match is found, returns 0.
    match_viz_bytes: Encoded image bytes with visualization of the match, if
      there is one, and if `query_im_array` and `index_im_array` are properly
      set. Otherwise, it's an empty bytes string.

  Raises:
    ValueError: If local descriptors from query and index images have different
      dimensionalities.
  """
  num_features_query = query_locations.shape[0]
  num_features_index_image = index_image_locations.shape[0]
  if not num_features_query or not num_features_index_image:
    return 0, b''

  local_feature_dim = query_descriptors.shape[1]
  if index_image_descriptors.shape[1] != local_feature_dim:
    raise ValueError(
        'Local feature dimensionality is not consistent for query and index '
        'images.')

  # Construct KD-tree used to find nearest neighbors.
  index_image_tree = spatial.cKDTree(index_image_descriptors)
  if use_ratio_test:
    distances, indices = index_image_tree.query(
        query_descriptors, k=2, n_jobs=-1)
    query_locations_to_use = np.array([
        query_locations[i,]
        for i in range(num_features_query)
        if distances[i][0] < descriptor_matching_threshold * distances[i][1]
    ])
    index_image_locations_to_use = np.array([
        index_image_locations[indices[i][0],]
        for i in range(num_features_query)
        if distances[i][0] < descriptor_matching_threshold * distances[i][1]
    ])
  else:
    _, indices = index_image_tree.query(
        query_descriptors,
        distance_upper_bound=descriptor_matching_threshold,
        n_jobs=-1)

    # Select feature locations for putative matches.
    query_locations_to_use = np.array([
        query_locations[i,]
        for i in range(num_features_query)
        if indices[i] != num_features_index_image
    ])
    index_image_locations_to_use = np.array([
        index_image_locations[indices[i],]
        for i in range(num_features_query)
        if indices[i] != num_features_index_image
    ])

  # If there are not enough putative matches, early return 0.
  if query_locations_to_use.shape[0] <= _MIN_RANSAC_SAMPLES:
    return 0, b''

  # Perform geometric verification using RANSAC.
  _, inliers = measure.ransac(
      (index_image_locations_to_use, query_locations_to_use),
      transform.AffineTransform,
      min_samples=_MIN_RANSAC_SAMPLES,
      residual_threshold=ransac_residual_threshold,
      max_trials=_NUM_RANSAC_TRIALS,
      random_state=ransac_seed)
  match_viz_bytes = b''

  if inliers is None:
    inliers = []
  elif query_im_array is not None and index_im_array is not None:
    if query_im_scale_factors is None:
      query_im_scale_factors = [1.0, 1.0]
    if index_im_scale_factors is None:
      index_im_scale_factors = [1.0, 1.0]
    inlier_idxs = np.nonzero(inliers)[0]
    _, ax = plt.subplots()
    ax.axis('off')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    feature.plot_matches(
        ax,
        query_im_array,
        index_im_array,
        query_locations_to_use * query_im_scale_factors,
        index_image_locations_to_use * index_im_scale_factors,
        np.column_stack((inlier_idxs, inlier_idxs)),
        only_matches=True)

    match_viz_io = io.BytesIO()
    plt.savefig(match_viz_io, format='jpeg', bbox_inches='tight', pad_inches=0)
    match_viz_bytes = match_viz_io.getvalue()

  return sum(inliers), match_viz_bytes


def RerankByGeometricVerification(input_ranks,
                                  initial_scores,
                                  query_name,
                                  index_names,
                                  query_features_dir,
                                  index_features_dir,
                                  junk_ids,
                                  local_feature_type = 'delf',
                                  local_feature_extension=_DELF_EXTENSION,
                                  ransac_seed=None,
                                  descriptor_matching_threshold=0.9,
                                  ransac_residual_threshold=10.0,
                                  use_ratio_test=False):
  """Re-ranks retrieval results using geometric verification.

  Args:
    input_ranks: 1D NumPy array with indices of top-ranked index images, sorted
      from the most to the least similar.
    initial_scores: 1D NumPy array with initial similarity scores between query
      and index images. Entry i corresponds to score for image i.
    query_name: Name for query image (string).
    index_names: List of names for index images (strings).
    query_features_dir: Directory where query local feature file is located
      (string).
    index_features_dir: Directory where index local feature files are located
      (string).
    junk_ids: Set with indices of junk images which should not be considered
      during re-ranking.
    local_feature_extension: String, extension to use for loading local feature
      files.
    ransac_seed: Seed used by RANSAC. If None (default), no seed is provided.
    descriptor_matching_threshold: Threshold used for local descriptor matching.
    ransac_residual_threshold: Residual error threshold for considering matches
      as inliers, used in RANSAC algorithm.
    use_ratio_test: If True, descriptor matching is performed via ratio test,
      instead of distance-based threshold.

  Returns:
    output_ranks: 1D NumPy array with index image indices, sorted from the most
      to the least similar according to the geometric verification and initial
      scores.

  Raises:
    ValueError: If `input_ranks`, `initial_scores` and `index_names` do not have
      the same number of entries.
  """
  num_index_images = len(index_names)
  if len(input_ranks) != num_index_images:
    raise ValueError('input_ranks and index_names have different number of '
                     'elements: %d vs %d' %
                     (len(input_ranks), len(index_names)))
  if len(initial_scores) != num_index_images:
    raise ValueError('initial_scores and index_names have different number of '
                     'elements: %d vs %d' %
                     (len(initial_scores), len(index_names)))

  # Filter out junk images from list that will be re-ranked.
  input_ranks_for_gv = []
  for ind in input_ranks:
    if ind not in junk_ids:
      input_ranks_for_gv.append(ind)
  num_to_rerank = min(_NUM_TO_RERANK, len(input_ranks_for_gv))
  print("Rerank ",num_to_rerank," instances")

  # Load query image features.
  query_features_path = os.path.join(query_features_dir,
                                     query_name + local_feature_extension)
  if local_feature_type == 'delf':
    query_locations, _, query_descriptors, _, _ = feature_io.ReadFromFile(
      query_features_path)
    print("Shape of query_locations and query_descriptors", query_locations.shape, query_descriptors.shape)
  else:
    data = np.load(query_features_path)
    query_locations = data['keypoint'].reshape(data['keypoint'].shape[0],2)
    query_descriptors = data['feature']
    print("Shape of query_locations and query_descriptors", query_locations.shape, query_descriptors.shape)
    
  # Initialize list containing number of inliers and initial similarity scores.
  inliers_and_initial_scores = []
  for i in range(num_index_images):
    inliers_and_initial_scores.append([0, initial_scores[i]])

  # Loop over top-ranked images and get results.
  print('Starting to re-rank')
  for i in range(num_to_rerank):
    if i > 0 and i % _STATUS_CHECK_GV_ITERATIONS == 0:
      print('Re-ranking: i = %d out of %d' % (i, num_to_rerank))

    index_image_id = input_ranks_for_gv[i]

    # Load index image features.
    index_image_features_path = os.path.join(
        index_features_dir,
        index_names[index_image_id] + local_feature_extension)
    # (index_image_locations, _, index_image_descriptors, _,
    #  _) = feature_io.ReadFromFile(index_image_features_path)
    data = np.load(query_features_path)
    index_image_locations = data['keypoint'].reshape(data['keypoint'].shape[0],2)
    index_image_descriptors = data['feature']

    inliers_and_initial_scores[index_image_id][0], _ = MatchFeatures(
        query_locations,
        query_descriptors,
        index_image_locations,
        index_image_descriptors,
        ransac_seed=ransac_seed,
        descriptor_matching_threshold=descriptor_matching_threshold,
        ransac_residual_threshold=ransac_residual_threshold,
        use_ratio_test=use_ratio_test)

  # Sort based on (inliers_score, initial_score).
  def _InliersInitialScoresSorting(k):
    """Helper function to sort list based on two entries.

    Args:
      k: Index into `inliers_and_initial_scores`.

    Returns:
      Tuple containing inlier score and initial score.
    """
    return (inliers_and_initial_scores[k][0], inliers_and_initial_scores[k][1])

  output_ranks = sorted(
      range(num_index_images), key=_InliersInitialScoresSorting, reverse=True)

  return output_ranks


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

def _ReadGlobalFeature(input_dir):
  """Reads DELG global features.

  Args:
    input_dir: Directory where features are located.
    image_list: List of image names for which to load features.

  Returns:
    global_descriptors: NumPy array of shape (len(image_list), D), where D
      corresponds to the global descriptor dimensionality.
  """
  num_images = len(listdir(input_dir))
  global_descriptors = []
  print('Starting to collect global descriptors for %d images...' % num_images)
  start = time.time()
  for i, filename in enumerate(listdir(input_dir)):
    if i > 0 and i % _STATUS_CHECK_LOAD_ITERATIONS == 0:
      elapsed = (time.time() - start)
      print('Reading global descriptors for image %d out of %d, last %d '
            'images took %f seconds' %
            (i, num_images, _STATUS_CHECK_LOAD_ITERATIONS, elapsed))
      start = time.time()

    global_descriptors.append(np.load(osp.join(input_dir,filename)).flatten())

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
  # if len(argv) > 1:
  #   print(argv)
  #   raise RuntimeError('Too many command-line arguments.')

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
  # query_global_features = _ReadDelgGlobalDescriptors(FLAGS.query_features_dir, query_list)
  # index_global_features = _ReadDelgGlobalDescriptors(FLAGS.index_features_dir, index_list)
  query_global_features = _ReadGlobalFeature(FLAGS.query_features_dir)
  index_global_features = _ReadGlobalFeature(FLAGS.index_features_dir)

  # Compute similarity between query and index images, potentially re-ranking
  # with geometric verification.
  ranks_before_gv = np.zeros([num_query_images, num_index_images], dtype='int32')
  if FLAGS.use_geometric_verification or FLAGS.use_sift_verification:
    ranks_after_gv = np.zeros([num_query_images, num_index_images], dtype='int32')

  query_time = []
  for i in range(num_query_images):
    print('Performing retrieval with query %d (%s)...' % (i, query_list[i]))
    start = time.time()

    # Compute similarity between global descriptors.
    similarities = np.dot(index_global_features, query_global_features[i])
    ranks_before_gv[i] = np.argsort(-similarities)
    print(f'Global rank {ranks_before_gv[i][:20]}')
    # Re-rank using geometric verification.
    if FLAGS.use_geometric_verification:
      ranks_after_gv[i] = RerankByGeometricVerification(
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
    elif FLAGS.use_sift_verification:
      ranks_after_gv[i] = RerankByGeometricVerification(
          input_ranks=ranks_before_gv[i],
          initial_scores=similarities,
          query_name=query_list[i],
          index_names=index_list,
          query_features_dir=FLAGS.query_sift_dir,
          index_features_dir=FLAGS.index_sift_dir,
          junk_ids=[],
          local_feature_type = 'sift',
          local_feature_extension='.npz',
          ransac_seed=0,
          descriptor_matching_threshold=FLAGS.local_descriptor_matching_threshold,
          ransac_residual_threshold=FLAGS.ransac_residual_threshold,
          use_ratio_test=FLAGS.use_ratio_test)
      print(f'local_rank {ranks_after_gv[i][:20]}')
      

    elapsed = (time.time() - start)
    query_time.append(elapsed)
    print('Done! Retrieval for query %d took %f seconds' % (i, elapsed))

  print("Average time: {}    Standard derivation: {}".format(mean(query_time),stdev(query_time)))

  # Create output directory if necessary.
  if not tf.io.gfile.exists(FLAGS.output_dir):
    tf.io.gfile.makedirs(FLAGS.output_dir)

  # global_path = os.path.join(FLAGS.output_dir, 'global_rank.txt')
  # np.savetxt(global_path, ranks_before_gv.astype(np.int32))
  # if FLAGS.use_geometric_verification:
  #   local_path = os.path.join(FLAGS.output_dir, 'local_rank.txt')
  #   np.savetxt(global_path, ranks_after_gv.astype(np.int32))

  # Write total time to query all query image set
  # write_time_path = os.path.join(FLAGS.output_dir, 'query_time.txt')
  # with tf.io.gfile.GFile(write_time_path, 'w') as f:
  #   f.write(f'Total query time: {total_query_time} seconds\n')
  #   f.write(f'Avarage query time: {average_query_time} seconds')

  # Write metric to file
  metric_bf_path = os.path.join(FLAGS.output_dir, _METRICS_BF_FILENAME)
  metrics = ComputeMetrics(ranks_before_gv, id_ground_truth, _PR_RANKS)
  with tf.io.gfile.GFile(metric_bf_path, 'w') as f:
      f.write('mAP={}\n  mP@k{} {}\n  mR@k{} {}\n'.format(
        np.around(metrics[0] * 100, decimals=2),
        np.array(_PR_RANKS), np.around(metrics[1] * 100, decimals=2),
        np.array(_PR_RANKS), np.around(metrics[2] * 100, decimals=2)))

  if FLAGS.use_geometric_verification or FLAGS.use_sift_verification:
    metrics_a = ComputeMetrics(ranks_after_gv, id_ground_truth, _PR_RANKS)
    metric_af_path = os.path.join(FLAGS.output_dir, _METRICS_AF_FILENAME)
    with tf.io.gfile.GFile(metric_af_path, 'w') as f:
      f.write('mAP={}\n  mP@k{} {}\n  mR@k{} {}\n'.format(
        np.around(metrics_a[0] * 100, decimals=2),
        np.array(_PR_RANKS), np.around(metrics_a[1] * 100, decimals=2),
        np.array(_PR_RANKS), np.around(metrics_a[2] * 100, decimals=2)))

if __name__ == '__main__':
  app.run(main)
