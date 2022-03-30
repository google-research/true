"""Tests for meta_evaluation_lib."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd

from google3.third_party.google_research.google_research.true import meta_evaluation_lib

LABELS = [1, 1, 0, 0]
TEST_DF = pd.DataFrame({
    'label': LABELS,
    'metric_1': [1, 1, 0, 0],
    'metric_2': [1, 0.2, 0.2, 1]
})


class PrecisionRecallTest(parameterized.TestCase):

  def assert_precision_recall_almost_equal(self, d1, d2):
    self.assertIsInstance(d1, meta_evaluation_lib.PrecisionRecallValues,
                          (d1, d2))
    self.assertIsInstance(d2, meta_evaluation_lib.PrecisionRecallValues,
                          (d1, d2))
    self.assertAlmostEqual(d1.grounded_precision, d2.grounded_precision)
    self.assertAlmostEqual(d1.grounded_recall, d2.grounded_recall)
    self.assertAlmostEqual(d1.ungrounded_precision, d2.ungrounded_precision)
    self.assertAlmostEqual(d1.ungrounded_recall, d2.ungrounded_recall)
    self.assertAlmostEqual(d1.grounded_f1, d2.grounded_f1)
    self.assertAlmostEqual(d1.ungrounded_f1, d2.ungrounded_f1)

  @parameterized.named_parameters(
      dict(
          testcase_name='int_scores_perfect_gmean',
          scores=[1, 1, 0, 0],
          labels=LABELS,
          expected_threshold=0,
          expected_gmean=1),
      dict(
          testcase_name='float_scores_perfect_gmean',
          scores=[0.7, 1, 0.5, 0],
          labels=LABELS,
          expected_threshold=0.5,
          expected_gmean=1),
      dict(
          testcase_name='imperfect_gmean',
          scores=[0.5, 0.4, 0.7, 0.2],
          labels=LABELS,
          expected_threshold=0.2,
          expected_gmean=0.707106781),
  )
  def test_get_optimal_gmeans_threshold(self, scores, labels,
                                        expected_threshold, expected_gmean):
    threshold, gmean = meta_evaluation_lib.get_optimal_gmeans_threshold(
        np.array(scores), np.array(labels))
    self.assertAlmostEqual(expected_threshold, threshold)
    self.assertAlmostEqual(expected_gmean, gmean)

  @parameterized.named_parameters(
      dict(
          testcase_name='all_ones',
          scores=[1, 1, 0, 0],
          labels=LABELS,
          threshold=0.5,
          expected=meta_evaluation_lib.PrecisionRecallValues(
              grounded_precision=1,
              grounded_recall=1,
              grounded_f1=1,
              ungrounded_precision=1,
              ungrounded_recall=1,
              ungrounded_f1=1),
      ),
      dict(
          testcase_name='all_ones_different_threshold',
          scores=[0.8, 1, 0.5, 0],
          labels=LABELS,
          threshold=0.7,
          expected=meta_evaluation_lib.PrecisionRecallValues(
              grounded_precision=1,
              grounded_recall=1,
              grounded_f1=1,
              ungrounded_precision=1,
              ungrounded_recall=1,
              ungrounded_f1=1),
      ),
      dict(
          testcase_name='zero_grounded_values',
          scores=[0, 0, 0, 0],
          labels=LABELS,
          threshold=0.5,
          expected=meta_evaluation_lib.PrecisionRecallValues(
              grounded_precision=0,
              grounded_recall=0,
              grounded_f1=0,
              ungrounded_precision=0.5,
              ungrounded_recall=1,
              ungrounded_f1=0.666666666667),
      ),
      dict(
          testcase_name='zero_ungrounded_values',
          scores=[1, 1, 1, 1],
          labels=LABELS,
          threshold=0.5,
          expected=meta_evaluation_lib.PrecisionRecallValues(
              grounded_precision=0.5,
              grounded_recall=1,
              grounded_f1=0.666666666667,
              ungrounded_precision=0,
              ungrounded_recall=0,
              ungrounded_f1=0),
      ),
      dict(
          testcase_name='all_zeros',
          scores=[0, 0, 1, 1],
          labels=LABELS,
          threshold=0.5,
          expected=meta_evaluation_lib.PrecisionRecallValues(
              grounded_precision=0,
              grounded_recall=0,
              grounded_f1=0,
              ungrounded_precision=0,
              ungrounded_recall=0,
              ungrounded_f1=0),
      ),
      dict(
          testcase_name='scores_equal_to_threshold',
          scores=[1, 1, 0.5, 0.5],
          labels=LABELS,
          threshold=0.5,
          expected=meta_evaluation_lib.PrecisionRecallValues(
              grounded_precision=1,
              grounded_recall=1,
              grounded_f1=1,
              ungrounded_precision=1,
              ungrounded_recall=1,
              ungrounded_f1=1),
      ),
      dict(
          testcase_name='all_half',
          scores=[1, 0.2, 0.2, 1],
          labels=LABELS,
          threshold=0.5,
          expected=meta_evaluation_lib.PrecisionRecallValues(
              grounded_precision=0.5,
              grounded_recall=0.5,
              grounded_f1=0.5,
              ungrounded_precision=0.5,
              ungrounded_recall=0.5,
              ungrounded_f1=0.5),
      ),
      dict(
          testcase_name='float_values',
          scores=[0.88, 0, 0.42, 0.83, 0.6, 0],
          labels=[1, 0, 0, 0, 0, 1],
          threshold=0.5,
          expected=meta_evaluation_lib.PrecisionRecallValues(
              grounded_precision=0.333333333,
              grounded_recall=0.5,
              grounded_f1=0.39999999976,
              ungrounded_precision=0.666666666,
              ungrounded_recall=0.5,
              ungrounded_f1=0.57142857),
      ))
  def test_compute_precision_recall(self, scores, labels, threshold, expected):
    predictions = meta_evaluation_lib.get_predicted_labels(scores, threshold)
    result = meta_evaluation_lib.compute_precision_recall_per_threshold(
        predictions, labels)
    self.assert_precision_recall_almost_equal(expected, result)

  def test_evaluate_metrics(self):
    result = meta_evaluation_lib.evaluate_metrics(
        data=TEST_DF,
        scores_columns=['metric_1', 'metric_2'],
        thresholds=[0.5, 0.5])
    expected = pd.DataFrame({
        'metric': ['metric_1', 'metric_2'],
        'accuracy': [1, 0.5],
        'roc_auc': [1, 0.5],
        'grounded_precision': [1, 0.5],
        'grounded_recall': [1, 0.5],
        'grounded_f1': [1, 0.5],
        'ungrounded_precision': [1, 0.5],
        'ungrounded_recall': [1, 0.5],
        'ungrounded_f1': [1, 0.5]
    })
    pd.testing.assert_frame_equal(
        result, expected, check_dtype=False, check_exact=False)


if __name__ == '__main__':
  absltest.main()
