#!/usr/bin/env python3
'''
This script contains a function that performs
semantic search on a corpus of documents and answers
questions using BERT.
'''
from transformers import BertTokenizer, TFBertForQuestionAnswering
import tensorflow as tf
from sentence_transformers import SentenceTransformer, util
import os

def semantic_search(corpus_path, sentence):
    """
    Perform semantic search on a corpus of documents.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_embedding = model.encode(sentence, convert_to_tensor=True)

    max_similarity = -1
    most_similar_text = ""

    for filename in os.listdir(corpus_path):
        with open(os.path.join(corpus_path, filename), 'r') as file:
            document = file.read()
            doc_embedding = model.encode(document, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(sentence_embedding, doc_embedding)

            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_text = document

    return most_similar_text

def find_answer(question, reference):
    """
    Find answer to question in reference text using BERT.
    """
    model = TFBertForQuestionAnswering.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )

    inputs = tokenizer(question, reference, return_tensors="tf")
    outputs = model(inputs)

    answer_start = tf.argmax(outputs.start_logits, axis=1).numpy()[0]
    answer_end = tf.argmax(outputs.end_logits, axis=1).numpy()[0] + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end])
    )

    return answer if answer.strip() else None

def question_answer(corpus_path):
    """
    Run Q&A loop using semantic search on the corpus.
    """
    while True:
        user_input = input("Q: ").strip()
        
        if user_input.lower() in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break
        
        reference = semantic_search(corpus_path, user_input)
        answer = find_answer(user_input, reference)
        
        if answer:
            print(f"A: {answer}")
        else:
            print("A: Sorry, I do not understand your question.")
