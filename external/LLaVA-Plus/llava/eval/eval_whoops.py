from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from scipy.special import softmax
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

VOCAB_PATH = '/nfs/turbo/justincj-turbo/ancao/repos/ImprovingVLM/packages/LLaVA-Plus/llava/eval/vocab.txt'

vocab_table = tf.lookup.StaticVocabularyTable(
        tf.lookup.TextFileInitializer(
            filename=VOCAB_PATH,
            key_dtype=tf.string,
            key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64,
            value_index=tf.lookup.TextFileIndex.LINE_NUMBER
        ),
        num_oov_buckets=1)
cls_id, sep_id = vocab_table.lookup(tf.convert_to_tensor(['[CLS]', '[SEP]']))
tokenizer = text.BertTokenizer(vocab_lookup_table=vocab_table,
                               token_out_type=tf.int64,
                               preserve_unused_token=True,
                               lower_case=True)

def bertify_example(q, a, gt):
  question = tokenizer.tokenize(q).merge_dims(1, 2)
  reference = tokenizer.tokenize(a).merge_dims(1, 2)
  candidate = tokenizer.tokenize(gt).merge_dims(1, 2)

  input_ids, segment_ids = text.combine_segments(
      (candidate, reference, question), cls_id, sep_id)

  return {'input_ids': input_ids.numpy(), 'segment_ids': segment_ids.numpy()}


def pad(a, length=512):
  return np.append(a, np.zeros(length - a.shape[-1], np.int32))


def bertify_examples(examples, gt):
  input_ids = []
  segment_ids = []
  for i in range(len(examples)):
    example_inputs = bertify_example(examples[i][0], examples[i][1], gt[i])
    input_ids.append(pad(example_inputs['input_ids']))
    segment_ids.append(pad(example_inputs['segment_ids']))

  return {'input_ids': np.stack(input_ids), 'segment_ids': np.stack(segment_ids)}


# Load BEM model.BLIP
with open("/nfs/turbo/justincj-turbo/tiangel/improvingVLM/eval_data/single_img_dataset/whoops/answers/llava-v1.5-13b.jsonl", 'r') as json_file:
    json_list = list(json_file)
bem = hub.load('https://tfhub.dev/google/answer_equivalence/bem/1')
bem_scores = []
for item in tqdm(json_list):
    result = json.loads(item)
    q_a_list = [result["question"], result['gt_answers']] 
    gt = result["gen_answers"]

    try:
        inputs = bertify_examples(q_a_list, gt)
        raw_outputs = bem(inputs)
        softmax_score = list(softmax(raw_outputs, axis=1)[:, 1])
        bem_scores.append(np.mean(softmax_score))
    except:
        continue

print(f'\nBEM score: {round(np.mean(bem_scores) * 100, 2)}')