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

"""Download the TRUE datasets and convert them to a binary labels format."""
import collections
import io
import json
import tarfile
from typing import Any, Dict
import zipfile

from absl import app
from datasets import load_dataset
import pandas as pd
import requests

MNBM_WORKER_TO_IDX = {'wid_0': 0, 'wid_1': 1, 'wid_2': 2}
MNBM_LABEL_NAME_TO_NUM = {'extrinsic': 0, 'intrinsic': 0}
SUMMEVAL_NUM_ANNOTATORS = 3


def download_frank(data_split: str = 'valid') -> None:
  """Download the FRANK dataset.

  Args:
    data_split: The split to download, i.e.`valid` or `test`.
  """
  json_file = requests.get(
      'https://raw.githubusercontent.com/artidoro/frank/main/data/human_annotations_sentence.json'
  )
  dataset = json.loads(json_file.text)
  frank_data = []
  for example in dataset:
    if example['split'] == data_split:
      summary_annotations = example['summary_sentences_annotations']
      label = 1
      for sentence_annotation in summary_annotations:
        # If a sentence doesn't contain any inconsistency, it only has
        # one label, which is NoE (No Error).
        sentence_labels = [
            1 if annotation[0] == 'NoE' else 0
            for annotation in sentence_annotation.values()
        ]
        majority_label = int(round(sum(sentence_labels) / len(sentence_labels)))
        if not majority_label:
          # If one sentence is inconsistent, the entire summary is
          # considered inconsistent
          label = 0
      d = {
          'hash': example['hash'],
          'model': example['model_name'],
          'grounding': example['article'],
          'generated_text': example['summary'],
          'label': label
      }
      frank_data.append(d)

  df = pd.DataFrame.from_records(frank_data)
  df.to_csv(f'frank_{data_split}_download.csv')


def download_qags(data_source: str) -> None:
  """Download the QAGS dataset.

  Args:
    data_source: The articles source, i.e., cnndm or xsum.
  """
  jsonl_file = requests.get(
      f'https://raw.githubusercontent.com/W4ngatang/qags/master/data/mturk_{data_source}.jsonl'
  )
  dataset = []
  for line in jsonl_file.iter_lines():
    dataset.append(json.loads(line.decode()))
  qags_data = []
  for example in dataset:
    summary_sentences = example['summary_sentences']
    sentences_text = []
    label = 1
    for sentence in summary_sentences:
      sentences_text.append(sentence['sentence'])
      sentence_labels = [
          1 if annotation['response'] == 'yes' else 0
          for annotation in sentence['responses']
      ]
      majority_label = int(round(sum(sentence_labels) / len(sentence_labels)))
      if not majority_label:
        # If one or more sentences are ungrounded, we mark the entire summary
        # as ungrounded.
        label = 0
    d = {
        'grounding': example['article'],
        'generated_text': ' '.join(sentences_text),
        'label': label
    }
    qags_data.append(d)

  df = pd.DataFrame.from_records(qags_data)
  df.to_csv(f'qags_{data_source}_download.csv')


def download_begin(data_split: str = 'dev') -> None:
  """Download the BEGIN dataset.

  Args:
    data_split: The split to download, i.e.`dev` or `test`.
  """
  df = pd.read_csv(
      f'https://raw.githubusercontent.com/google/BEGIN-dataset/main/{data_split}_05_24_21.tsv',
      sep='\t')
  df = df.rename(columns={
      'evidence': 'grounding',
      'response': 'generated_text'
  })
  labels = []
  for _, row in df.iterrows():
    label = 1 if row['gold label'] == 'entailment' else 0
    labels.append(label)
  df['label'] = labels
  df.to_csv(f'begin_{data_split}_download.csv')


def download_dialfact(data_split: str = 'valid') -> None:
  """Download the DialFact dataset.

  Args:
    data_split: The split to download, i.e.`valid` or `test`.
  """
  jsonl_file = requests.get(
      f'https://raw.githubusercontent.com/salesforce/DialFact/master/data/{data_split}_split.jsonl'
  )
  dataset = []
  for line in jsonl_file.iter_lines():
    dataset.append(json.loads(line.decode()))
  dialfact_data = []
  for example in dataset:
    if example['type_label'] != 'factual':
      continue
    evidence_list = [evidence[2] for evidence in example['evidence_list']]
    evidence_joined = ' '.join(evidence_list)
    if evidence_joined:  # Avoid cases with empty evidence (no grounding)
      label = 1 if example['response_label'] == 'SUPPORTS' else 0
      d = {
          'grounding': evidence_joined,
          'generated_text': example['response'],
          'label': label
      }
      dialfact_data.append(d)

  df = pd.DataFrame.from_records(dialfact_data)
  df.to_csv(f'dialfact_{data_split}_download.csv')


def get_xsum_articles(data_split: str = 'test') -> Dict[str, str]:
  """Get a mapping of XSUM ids to the corresponding articles.

  Done using the XSUM dataset from HuggingFace Datasets.

  Args:
    data_split: Data split, i.e., `train`, `valid` or `test`.

  Returns:
    A dictionary mapping article ids to the corresponding article text.
  """
  id_to_document = {}
  xsum_articles = load_dataset('xsum', split=data_split)
  for example in xsum_articles:
    id_to_document[example['id']] = example['document'].replace('\n', ' ')
  return id_to_document


def download_mnbm() -> None:
  """Download the MNBM dataset."""
  xsum_id_to_document = get_xsum_articles('test')
  dataset = pd.read_csv(
      'https://github.com/google-research-datasets/xsum_hallucination_annotations/raw/master/hallucination_annotations_xsum_summaries.csv'
  )
  mnbm_data = []

  id_and_system_to_labels = collections.defaultdict(lambda: [1, 1, 1])
  id_and_system_to_summary = {}
  for _, row in dataset.iterrows():
    bbc_id = row['bbcid']
    system = row['system']
    worker_id = MNBM_WORKER_TO_IDX[row['worker_id']]
    id_and_system_to_labels[(bbc_id,
                             system)][worker_id] = MNBM_LABEL_NAME_TO_NUM.get(
                                 row['hallucination_type'], 1)
    id_and_system_to_summary[(bbc_id, system)] = row['summary']

  for key, labels in id_and_system_to_labels.items():
    label = int(round(sum(labels) / len(labels)))
    d = {
        'bbcid': key[0],
        'model': key[1],
        'grounding': xsum_id_to_document[str(key[0])],
        'generated_text': id_and_system_to_summary[key],
        'label': label
    }
    mnbm_data.append(d)

  df = pd.DataFrame.from_records(mnbm_data)
  df.to_csv('mnbm_download.csv')


def download_paws() -> None:
  """Download the PAWS dataset."""
  response = requests.get(
      'https://storage.googleapis.com/paws/english/paws_wiki_labeled_final.tar.gz',
      stream=True)
  with tarfile.open(
      fileobj=io.BytesIO(response.raw.read()), mode='r:gz') as tar_file:
    f = tar_file.extractfile('final/dev.tsv')
    df = pd.read_csv(f, sep='\t')
    df = df.rename(columns={
        'sentence1': 'grounding',
        'sentence2': 'generated_text'
    })
    df.to_csv('paws_download.csv')


def download_q2() -> None:
  """Download the Q2 dataset."""
  df = pd.read_csv(
      'https://raw.githubusercontent.com/orhonovich/q-squared/main/third_party/data/cross_annotation.csv'
  )
  q2_data = []
  for _, row in df.iterrows():
    for model in ['dodeca', 'memnet']:
      d = {
          'model': model,
          'grounding': row['knowledge'],
          'generated_text': row[f'{model}_response'],
          'label': 1 - int(row[f'{model}_label'])
      }
      q2_data.append(d)

  processed_df = pd.DataFrame.from_records(q2_data)
  processed_df.to_csv('q2_download.csv')


def download_vitc() -> None:
  """Download the VitaminC dataset."""
  dataset = []
  response = requests.get(
      'https://github.com/TalSchuster/talschuster.github.io/raw/master/static/vitaminc.zip',
      stream=True)
  with zipfile.ZipFile(io.BytesIO(response.raw.read())) as zip_file:
    jsonl_file = zip_file.extract('vitaminc/dev.jsonl')

  with open(jsonl_file, encoding='utf-8') as f:
    for line in f:
      dataset.append(json.loads(line))

  vitc_data = []
  for example in dataset:
    label = 1 if example['label'] == 'SUPPORTS' else 0
    d = {
        'id': example['unique_id'],
        'grounding': example['evidence'],
        'generated_text': example['claim'],
        'label': label
    }
    vitc_data.append(d)

  df = pd.DataFrame.from_records(vitc_data)
  df.to_csv('vitc_dev_download.csv')


def get_fever_labels(data_split: str = 'labelled_dev') -> Dict[int, str]:
  """Get a mapping of FEVER ids to labels.

  Done using the FEVER dataset from HuggingFace Datasets.

  Args:
    data_split: The data split to download.

  Returns:
   A dictionary mapping from FEVER ids to the corresponding labels.
  """
  id_to_label = {}
  fever_data = load_dataset('fever', 'v1.0', split=data_split)
  for example in fever_data:
    id_to_label[example['id']] = example['label']
  return id_to_label


def download_fever() -> None:
  """Download the FEVER dataset."""
  dataset = []
  fever_id_to_label = get_fever_labels()
  with open('nli_fever/dev_fitems.jsonl', encoding='utf-8') as f:
    for line in f:
      dataset.append(json.loads(line))

  fever_data = []
  for example in dataset:
    context = example['context']
    if not context:
      continue
    label = 1 if fever_id_to_label[int(example['cid'])] == 'SUPPORTS' else 0
    d = {
        'id': example['cid'],
        'grounding': example['context'],
        'generated_text': example['query'],
        'label': label
    }
    fever_data.append(d)

  df = pd.DataFrame.from_records(fever_data)
  df.to_csv('fever_dev_download.csv')


def get_cnndm_articles(data_split: str = 'test') -> Dict[str, str]:
  """Get a mapping of CNNDM ids to the corresponding articles.

  Done using the CNNDM dataset from HuggingFace Datasets.

  Args:
    data_split: Data split, i.e., `train`, `valid` or `test`.

  Returns:
    A dictionary mapping article ids to the corresponding article text.
  """
  id_to_document = {}
  cnndm_articles = load_dataset('cnn_dailymail', '3.0.0', split=data_split)
  for example in cnndm_articles:
    article = ' '.join(
        filter(None, [x.strip() for x in example['article'].split('\n')]))
    id_to_document[example['id']] = article
  return id_to_document


def download_summeval():
  """Download the SummEval dataset."""
  cnndm_id_to_document = get_cnndm_articles('test')
  jsonl_file = requests.get(
      'https://storage.googleapis.com/sfr-summarization-repo-research/model_annotations.aligned.jsonl'
  )
  dataset = []
  for line in jsonl_file.iter_lines():
    dataset.append(json.loads(line.decode()))
  summeval_data = []
  for example in dataset:
    expert_annotations = example['expert_annotations']
    assert len(expert_annotations) == SUMMEVAL_NUM_ANNOTATORS
    labels_sum = 0
    for i in range(SUMMEVAL_NUM_ANNOTATORS):
      label = expert_annotations[i]['consistency']
      labels_sum += label
    # we label as "consistent" if all experts gave a score of 5
    majority_label = 1 if labels_sum == 15 else 0
    cnndm_id = example['id'].replace('dm-test-', '').replace('cnn-test-', '')
    d = {
        'id': cnndm_id,
        'grounding': cnndm_id_to_document[cnndm_id],
        'generated_text': example['decoded'],
        'label': majority_label
    }
    summeval_data.append(d)

  df = pd.DataFrame.from_records(summeval_data)
  df.to_csv('summeval_download.csv')


def main(argv: Any) -> None:
  del argv
  download_frank()
  download_qags(data_source='xsum')
  download_qags(data_source='cnndm')
  download_mnbm()
  download_begin()
  download_q2()
  download_dialfact()
  download_vitc()
  download_paws()
  download_summeval()

  verify_fever_download = input('Did you download and extract nli_fever? Y/N: ')
  while verify_fever_download not in ['Y', 'y']:
    print('Please download nli-fever from '
          'https://www.dropbox.com/s/hylbuaovqwo2zav/nli_fever.zip?dl=0 '
          'and extract it.')
    print('After extraction, your working directory should contain an "nli_fever" '
          'folder.')
    verify_fever_download = input(
        'Did you download and extract nli_fever? Y/N: ')
  download_fever()


if __name__ == '__main__':
  app.run(main)
