"""Compute meta-evaluation scores for factual consistency metrics.

The scores include the ROC area under the curve of the binary decision -
grounded or ungrounded, the accuracy of the binary decision (given a score
threshold), as well as ungrounded Precision and Recall values (again, given a
threshold).
"""

from typing import Any

from absl import app
from absl import flags
import pandas as pd

from google3.third_party.google_research.google_research.true import meta_evaluation_lib

_INPUT_PATH = flags.DEFINE_string(
    name='input_path',
    default=None,
    help='Path of the csv file containing metrics scores. Each metric should '
    'correspond to a column in the input file, and metrics scores should be in '
    'range [0,1]. Additionally, the file should have a `label` column, '
    'containing the gold-standard labels - 0 for inconsistent text and 1 for '
    'consistent text.',
    required=True)
_METRICS_SCORES_COLUMNS = flags.DEFINE_list(
    name='metrics_scores_columns',
    default=None,
    help='The names of the columns containing the different metrics scores.',
    required=True)
_THRESHOLDS = flags.DEFINE_list(
    name='thresholds',
    default=None,
    help='Thresholds for calculating accuracy, as well as grounded and '
    'ungrounded Precision and Recall. By default, the threshold is 0.5 for '
    'all metrics. When specifying thresholds, one threshold should be '
    'specified for each input metric to test.',
)
_OUTPUT_PATH = flags.DEFINE_string(
    name='output_path',
    default=None,
    help='Path to the output csv file, will contain all the meta-evaluation '
    'scores.',
    required=True)

_DEFAULT_THRESHOLD = 0.5


@flags.multi_flags_validator(
    ['metrics_scores_columns', 'thresholds'],
    message='If thresholds are provided, the number of thresholds must be '
    'equal to the number of metrics_scores_columns.')
def check_metrics_scores_columns_thresholds_flags(flags_dict: Any) -> bool:
  thresholds = flags_dict['thresholds']
  metrics_scores_columns = flags_dict['metrics_scores_columns']
  return not thresholds or len(thresholds) == len(metrics_scores_columns)


def main(argv: Any) -> None:
  del argv
  scores_columns = _METRICS_SCORES_COLUMNS.value
  if _THRESHOLDS.value:
    thresholds = _THRESHOLDS.value
  else:
    print('Computing results using a default threshold of 0.5. You can define '
          'other threshold values by setting --thresholds.')
    thresholds = [_DEFAULT_THRESHOLD] * len(scores_columns)
  df = pd.read_csv(_INPUT_PATH.value, encoding='utf8')
  meta_eval_report_df = meta_evaluation_lib.evaluate_metrics(
      df, scores_columns, thresholds)
  meta_eval_report_df.to_csv(_OUTPUT_PATH.value, index=False)


if __name__ == '__main__':
  app.run(main)
