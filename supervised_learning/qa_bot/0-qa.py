import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import numpy as np

def question_answer(question, reference):
    """Find answer to question in reference text using BERT."""
    model = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/4")
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )

    input_ids = tokenizer.encode(
        question, reference, max_length=384, truncation=True, padding='max_length'
    )
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(tokenizer.encode(question)) + [1] * (
        len(input_ids) - len(tokenizer.encode(question))
    )

    input_ids = tf.constant([input_ids])
    input_mask = tf.constant([input_mask])
    segment_ids = tf.constant([segment_ids])

    outputs = model([input_ids, input_mask, segment_ids])

    start_logits = outputs['start_logits'][0]
    end_logits = outputs['end_logits'][0]

    start_index = tf.argmax(start_logits).numpy()
    end_index = tf.argmax(end_logits).numpy()

    if end_index < start_index:
        start_index, end_index = end_index, start_index

    answer = tokenizer.decode(input_ids[0][start_index:end_index+1])

    if not answer or answer.strip() in ['[CLS]', '[SEP]', '[PAD]']:
        return None

    return answer.strip()
