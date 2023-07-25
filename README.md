# TRUE

Note: This is not an officially supported Google product.

This repository includes code and data accompanying our [NAACL 2022 paper 
"TRUE: Re-evaluating Factual Consistency Evaluation"](https://arxiv.org/pdf/2204.04991.pdf).

## Model

We open sourced an NLI model based on this work on [HuggingFace](https://huggingface.co/google/t5_xxl_true_nli_mixture).

## Data

We provide a script that downloads all 11 datasets used in TRUE and converts
them to a standardized binary scheme.
To download the datasets, first download the
[nli-fever](https://www.dropbox.com/s/hylbuaovqwo2zav/nli_fever.zip?dl=0)
directory and extract it. After extraction, the working directory should 
contain an nli_fever directory.

Note: The SummEval datatset requires pairing summaries with the original
CNN/DailyMail articles. Downloading these (via the HuggingFace
[Datasets](https://huggingface.co/docs/datasets/index)
library) might take a while.

To download and standardize the datasets for TRUE, run:

```
python true/src/download_datasets.py
```

For computing the TRUE scores, use the "grounding", "generated_text" and
"label" columns in the resulting csv files. The "label" column will contain 
the binary label to evaluate against.

## Usage

To compute meta-evaluation scores for new metrics, you should have a csv file
with one column per metric and a label column. Scores should be in range [0,1].

For example:


```
     ...  label  metric_1  metric_2    metric_3
0    ...  0      0.5       0.666666    0.42
1    ...  0      0.2       1           0.75
2    ...  1      0.8       0.5         0

```

run:

```
python true/src/meta_evaluation_scores.py \
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
python true/src/meta_evaluation_plots.py \
       --input_path=data/"{INPUT_FILE}" \
       --metrics_scores_columns="{METRICS}" \
       --output_path="{OUTPUT_PATH}"
```
