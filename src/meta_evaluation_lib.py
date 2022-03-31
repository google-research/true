# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code for running meta-evaluation for factual consistency metrics.

The meta-evaluation includes scores and plots. The scores include the ROC area
under the curve of the binary decision - grounded or ungrounded, the accuracy of
the binary decision (given a score threshold), as well as ungrounded Precision
and Recall values (again, given a threshold). The plots include the ROC curve
and the grounded and ungrounded Precision vs. Recall curves, for a range of
possible thresholds. Additionally, this file includes code for computing
optimal thresholds for ungrounded text detection. Optimal thresholds are
computed according to the geometric mean (G-mean) of the true positive rate
(TPR) and 1 - the false positive rate (FPR).
"""

import dataclasses
import enum
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics

sns.set_theme(style='darkgrid', font_scale=1.1)


class Labels(enum.IntEnum):
  GROUNDED = 1
  UNGROUNDED = 0


@dataclasses.dataclass
class PrecisionRecallValues():
  """Class representing the grounded and ungrounded precision and recall."""
  grounded_precision: float
  grounded_recall: float
  grounded_f1: float
  ungrounded_precision: float
  ungrounded_recall: float
  ungrounded_f1: float


@dataclasses.dataclass
class MetaEvalValues():
  """Class representing the meta-evaluation measures."""
  metric: List[str]
  accuracy: List[float]
  roc_auc: List[float]
  grounded_precision: List[float]
  grounded_recall: List[float]
  grounded_f1: List[float]
  ungrounded_precision: List[float]
  ungrounded_recall: List[float]
  ungrounded_f1: List[float]


def get_predicted_labels(scores: np.ndarray, threshold: float) -> np.ndarray:
  """Determine grounding labels (grounded or ungrounded) given metric scores.

  The scores should be in range [0,1]. An example with a score smaller than or
  equals to the given threshold is marked as ungrounded, and marked as grounded
  otherwise.

  Args:
    scores: An array containing the metric scores, should be in range [0,1].
    threshold: A threshold for determining whether a given score corresponds
      with a grounded example or not.

  Returns:
    An array containing the predicted labels, 0 for ungrounded cases and 1
    otherwise.
  """
  return np.array([
      Labels.UNGROUNDED if score <= threshold else Labels.GROUNDED
      for score in scores
  ])


def get_optimal_gmeans_threshold(scores: np.ndarray,
                                 labels: np.ndarray) -> Tuple[float, float]:
  """Compute the optimal threshold for ungrounded classification.

  The optimal threshold is computed according to the geometric mean (G-mean) of
  the true positive rate (TPR) and 1 - the false positive rate (FPR).
  See details in:
  https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/

  Args:
    scores: An array containing the metric scores, should be in range [0,1].
    labels: The ground-truth binary grounded (1) or ungrounded (0) labels.

  Returns:
    A 2-tuple containing the optimal G-mean threshold and the corresponding
    G-mean value.
  """
  # Labels and scores are inverted to compute the FPR and TPR for ungrounded
  # detection. This way, the `positive class` in the roc_curve function will
  # be the ungrounded class.
  fpr, tpr, thresholds = metrics.roc_curve(
      y_true=(1 - labels), y_score=(1 - scores))
  gmeans = np.sqrt(tpr * (1 - fpr))
  max_gmeans_idx = np.argmax(gmeans)
  return (1 - thresholds[max_gmeans_idx], gmeans[max_gmeans_idx])


def compute_precision_recall_per_threshold(
    predictions: np.ndarray, labels: np.ndarray) -> PrecisionRecallValues:
  """Compute the grounded and ungrounded Precision-Recall values.

  Gets as input a metric's grounding evaluation predictions and calculates the
  grounded and ungrounded Precision-Recall values of the metric.

  Args:
    predictions: np.array containing the binary grounded or ungrounded
      predictions, according to the tested metric.
    labels: The ground-truth binary grounded (1) or ungrounded (0) labels.

  Returns:
    A dictionary containing the grounded and ungrounded Precision-Recall values.
  """
  grounded_precision = metrics.precision_score(
      y_true=labels, y_pred=predictions, pos_label=Labels.GROUNDED)
  grounded_recall = metrics.recall_score(
      y_true=labels, y_pred=predictions, pos_label=Labels.GROUNDED)
  grounded_f1 = metrics.f1_score(
      y_true=labels, y_pred=predictions, pos_label=Labels.GROUNDED)
  ungrounded_precision = metrics.precision_score(
      y_true=labels, y_pred=predictions, pos_label=Labels.UNGROUNDED)
  ungrounded_recall = metrics.recall_score(
      y_true=labels, y_pred=predictions, pos_label=Labels.UNGROUNDED)
  ungrounded_f1 = metrics.f1_score(
      y_true=labels, y_pred=predictions, pos_label=Labels.UNGROUNDED)
  result = PrecisionRecallValues(
      grounded_precision=grounded_precision,
      grounded_recall=grounded_recall,
      grounded_f1=grounded_f1,
      ungrounded_precision=ungrounded_precision,
      ungrounded_recall=ungrounded_recall,
      ungrounded_f1=ungrounded_f1)
  return result


def evaluate_metrics(data: pd.DataFrame, scores_columns: List[str],
                     thresholds: List[float]) -> pd.DataFrame:
  """Compute the meta-evaluation scores for given grounding evalution metrics.

  The meta-evaluation scores include the accuracy of the binary decision -
  grounded or ungrounded (given a score threshold), as well as the ROC area
  under the curve and the grounded and ungrounded Precision and Recall values.
  The scores are saved into a csv file and printed to the screen.

  Args:
    data: The input dataframe containing the different metrics scores. Should
      contain a column for each metric scores, as well as a binary `label`
      column, containing the ground-truth grounded or ungrounded labels.
    scores_columns: The names of the columns containing the different metrics
      scores in the input data.
    thresholds: A threshold for determining whether a given score corresponds
      with a grounded example or not.

  Returns:
    A dataframe containing all the meta-evaluation scores.
  """
  accuracies, roc_aucs = [], []
  grounded_precision_vals, grounded_recall_vals = [], []
  ungrounded_precision_vals, ungrounded_recall_vals = [], []
  grounded_f1_vals, ungrounded_f1_vals = [], []
  labels = data['label'].to_numpy()

  for i, metric_name in enumerate(scores_columns):
    print(metric_name)
    scores = data[metric_name].to_numpy()
    predictions = get_predicted_labels(scores, thresholds[i])
    accuracy = metrics.accuracy_score(y_true=labels, y_pred=predictions)
    accuracies.append(accuracy)

    fpr, tpr, _ = metrics.roc_curve(y_true=labels, y_score=scores)
    roc_aucs.append(metrics.auc(fpr, tpr))

    precision_recall_vals = compute_precision_recall_per_threshold(
        predictions, labels)
    grounded_precision_vals.append(precision_recall_vals.grounded_precision)
    grounded_recall_vals.append(precision_recall_vals.grounded_recall)
    grounded_f1_vals.append(precision_recall_vals.grounded_f1)
    ungrounded_precision_vals.append(precision_recall_vals.ungrounded_precision)
    ungrounded_recall_vals.append(precision_recall_vals.ungrounded_recall)
    ungrounded_f1_vals.append(precision_recall_vals.ungrounded_f1)
    print(f'For {metric_name}, accuracy: {accuracy}, {precision_recall_vals}')

  meta_eval_scores = MetaEvalValues(
      metric=scores_columns,
      accuracy=accuracies,
      roc_auc=roc_aucs,
      grounded_precision=grounded_precision_vals,
      grounded_recall=grounded_recall_vals,
      grounded_f1=grounded_f1_vals,
      ungrounded_precision=ungrounded_precision_vals,
      ungrounded_recall=ungrounded_recall_vals,
      ungrounded_f1=ungrounded_f1_vals)
  meta_eval_df = pd.DataFrame(dataclasses.asdict(meta_eval_scores))
  return meta_eval_df


def plot_precision_recall_comparison(data: pd.DataFrame,
                                     scores_columns: List[str],
                                     grounded_positive: bool,
                                     output_path: str) -> None:
  """Plot the Precision-Recall curves for several metrics.

  The plot shows the Precision-Recall trade-off for several input metrics,
  allowing comparison between the different metrics.

  Args:
    data: The input data containing the different metrics scores. Should contain
      a column for each metric scores, as well as a binary `label` column,
      containing the ground-truth grounded or ungrounded labels.
    scores_columns: The names of the columns containing the different metrics
      scores in the input data.
    grounded_positive: Whether to plot Precision-Recall values for determining
      if text is grounded or ungrounded. If true, plot the grounded case.
    output_path: Path of the output plot file.
  """
  all_metrics_precision = np.array([])
  all_metrics_recall = np.array([])
  metrics_names = []

  labels = data['label'].to_numpy()

  for metric_name in scores_columns:
    scores = data[metric_name].to_numpy()

    if grounded_positive:
      pos_label = 1
    else:
      # In the scikit learn implementation, examples are predicted with a
      # positive label if their scores is greater than, or equal to, the
      # threshold. Since in ungrounded detection we treat the ungrounded case as
      # the `positive` class aim at detecting, we take 1-score for labeling as
      # ungrounded.
      pos_label = 0
      scores = 1 - scores

    precision, recall, _ = metrics.precision_recall_curve(
        y_true=labels, probas_pred=scores, pos_label=pos_label)
    auc = metrics.auc(recall, precision)

    all_metrics_precision = np.append(all_metrics_precision, precision)
    all_metrics_recall = np.append(all_metrics_recall, recall)
    metrics_names.extend([f'{metric_name}, auc:{auc:.2f}'] * len(precision))

  plt.figure(figsize=(16, 9))
  precision_recall_values_df = pd.DataFrame({
      'Recall': all_metrics_recall,
      'Precision': all_metrics_precision,
      'Metric': metrics_names
  })

  plot_type = 'grounded' if grounded_positive else 'ungrounded'
  sns_plot = sns.lineplot(
      x='Recall', y='Precision', hue='Metric',
      data=precision_recall_values_df).set_title(
          f'Precision vs. Recall for detecting {plot_type} text')
  with open(output_path, 'wb') as f:
    sns_plot.figure.savefig(f)
    return


def plot_roc_comparison(data: pd.DataFrame, scores_columns: List[str],
                        grounded_positive: bool, output_path: str) -> None:
  """Plot an ROC curve for several metrics.

  The plot shows the true positive rate (TPR) vs. the false positive rate (FPR)
  trade-off for several input metrics, allowing comparison between the different
  metrics.

  Args:
    data: The input data containing the different metrics scores. Should contain
      a column for each metric scores, as well as a binary `label` column,
      containing the ground-truth grounded or ungrounded labels.
    scores_columns: The names of the columns containing the different metrics
      scores in the input data.
    grounded_positive: Whether to plot ROC curve for determining if text is
      grounded or ungrounded. If true, plot the grounded case.
    output_path: Path of the output plot file.
  """
  all_metrics_tpr = np.array([])
  all_metrics_fpr = np.array([])
  metrics_names = []

  labels = data['label'].to_numpy()

  for metric_name in scores_columns:
    scores = data[metric_name].to_numpy()

    if grounded_positive:
      pos_label = 1
    else:
      # In the scikit learn implementation, examples are predicted with a
      # positive label if their scores is greater than, or equal to, the
      # threshold. Since in unogrunded detection we treat the ungrounded case as
      # the `positive` class aim at detecting, we take 1-score for labeling as
      # ungrounded.
      pos_label = 0
      scores = 1 - scores

    fpr, tpr, _ = metrics.roc_curve(
        y_true=labels, y_score=scores, pos_label=pos_label)
    auc = metrics.auc(fpr, tpr)

    all_metrics_tpr = np.append(all_metrics_tpr, tpr)
    all_metrics_fpr = np.append(all_metrics_fpr, fpr)
    metrics_names.extend([f'{metric_name}, auc:{auc:.2f}'] * len(tpr))

  plt.figure(figsize=(16, 9))
  roc_values_df = pd.DataFrame({
      'FPR': all_metrics_fpr,
      'TPR': all_metrics_tpr,
      'Metric': metrics_names
  })

  plot_type = 'grounded' if grounded_positive else 'ungrounded'
  sns_plot = sns.lineplot(
      x='FPR', y='TPR', hue='Metric', data=roc_values_df).set_title(
          f'TPR vs. FPR for detecting {plot_type} text')
  with open(output_path, 'wb') as f:
    sns_plot.figure.savefig(f)
    return
