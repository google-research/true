# TRUE

Code and data accompanying the paper "TRUE: Re-evaluating Factual
Consistency Evaluation".

## Data

We provide a script that downloads all the 11 datasets used in TRUE and converts
them to a standardized binary scheme.
To download the datasets, please first download
[nli-fever](https://www.dropbox.com/s/hylbuaovqwo2zav/nli_fever.zip?dl=0)
and extract it. After extraction, the directory should contain an nli_fever
folder.

Note: The SummEval datatset requires pairing summaries with the original
CNN/DailyMail articles. Downloading the articles (via the
[Datasets](https://huggingface.co/docs/datasets/index)
library) might take a while.

To download the datasets, run:

```
python -m true.download_datasets.py
```

## Usage

To compute meta-evaluation scores for new metrics, you should have a csv file
with one column per each metric scores. Scores should be in range [0,1].

For example:


```
     ...  metric_1  metric_2    metric_3
0    ...  0         0.666666    0.42
1    ...  0         1           0.75
2    ...  1         0.5         0

```

run:

```
python -m true.meta_evaluation_scores.py \
       --input_path=data/"{INPUT_FILE}" \
       --metrics_scores_columns="{METRICS}" \
       --output_path="{OUTPUT_PATH}"
```

`metrics_scores_columns` should be a comma-separated list of metrics to test,
e.g.
```
metrics_scores_columns="metric_1,metric_2,metric_3"
```


The output file is a meta-evaluation report, containing ROC AUC scores, as well
as accuracy and grounded/ungrounded precision and recall. Please note that other
than the ROC AUC, all other measures require threshold tuning and use a default
threshold of 0.5, which is not necessarily suitable for all
metrics. To pass custom thresholds, use `--thresholds`.


Similarly, to create the ROC plot, run:

```
python -m true.meta_evaluation_plots.py \
       --input_path=data/"{INPUT_FILE}" \
       --metrics_scores_columns="{METRICS}" \
       --output_path="{OUTPUT_PATH}"
```
