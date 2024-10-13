#!/usr/bin/env python3
'''
This script contains a function that answers questions
from a reference text using BERT.
'''
from transformers import BertTokenizer, TFBertForQuestionAnswering
import tensorflow as tf

def question_answer(question, reference):
    """Find answer to question in reference text using BERT."""
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

def answer_loop(reference):
    """Run Q&A loop using the provided reference text."""
    while True:
        user_input = input("Q: ").strip()
        
        if user_input.lower() in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break
        
        answer = question_answer(user_input, reference)
        
        if answer:
            print(f"A: {answer}")
        else:
            print("A: Sorry, I do not understand your question.")
