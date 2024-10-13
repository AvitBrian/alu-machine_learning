'''
This module contains the Dataset class, which handles dataset loading,
tokenization, and encoding for Portuguese to English translation.
'''
import tensorflow as tf
import tensorflow_datasets as tfds

class Dataset:
    '''
    Handles dataset loading, tokenization, and encoding.
    '''
    def __init__(self):
        '''
        Initialize the dataset
        '''
        dataset, info = tfds.load('ted_hrlr_translate/pt_to_en',
                                  with_info=True,
                                  as_supervised=True)
        
        # Split the dataset
        self.data_train = dataset['train']
        self.data_valid = dataset['validation']
        
        # Tokenize the dataset
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)
        
        # Update data_train and data_valid by tokenizing the examples
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        '''
        Build tokenizers for Portuguese and English from the training data.
        '''
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2**15)
        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2**15)
        
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        '''
        Encode Portuguese and English sentences using the respective tokenizers.
        '''
        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]
        
        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size + 1]
        
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        '''
        Encode Portuguese and English sentences using TensorFlow operations.
        '''
        result_pt, result_en = tf.py_function(self.encode, [pt, en], [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return result_pt, result_en
