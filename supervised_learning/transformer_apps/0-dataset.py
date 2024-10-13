import tensorflow as tf
import tensorflow_datasets as tfds

class Dataset:
    def __init__(self):
        '''
        Initialize the dataset
        '''
        dataset, info = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
        
        # Split the dataset
        self.data_train = dataset['train']
        self.data_valid = dataset['validation']
        
        # Tokenize the dataset
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        '''
        Tokenize the dataset
        '''
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2**15)
        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2**15)
        
        return tokenizer_pt, tokenizer_en
