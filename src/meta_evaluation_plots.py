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

"""Plot the ROC curve for comparing factual consistency evaluation metrics."""

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
_OUTPUT_PATH = flags.DEFINE_string(
    name='output_path',
    default=None,
    help='Path to the output directory, will contain the output plots.',
    required=True)


def main(argv: Any) -> None:
  del argv
  df = pd.read_csv(_INPUT_PATH.value, encoding='utf8')
  scores_columns = _METRICS_SCORES_COLUMNS.value
  output_path = _OUTPUT_PATH.value

  meta_evaluation_lib.plot_roc_comparison(df, scores_columns, False,
                                          output_path)


if __name__ == '__main__':
  app.run(main)
