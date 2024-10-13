"""
This module contains the train_transformer function for creating and training
a transformer model for machine translation of Portuguese to English.
"""

import tensorflow as tf

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    Creates and trains a transformer model for machine translation.
    """
    # Load and prepare the dataset
    data = Dataset(batch_size, max_len)
    
    transformer = Transformer(N, dm, h, hidden, data.tokenizer_pt.vocab_size,
                              data.tokenizer_en.vocab_size, max_len, max_len)
    
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_function(real, pred):
        '''
        Define the loss function for the transformer model.
        '''
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    # Define learning rate schedule
    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, dm, warmup_steps=4000):
            '''
            Initialize the CustomSchedule.
            '''
            super(CustomSchedule, self).__init__()
            self.dm = dm
            self.dm = tf.cast(self.dm, tf.float32)
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            '''
            Calculate the learning rate for the given step.
            '''
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)
            return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)

    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9,
                                         beta_2=0.98, epsilon=1e-9)

    # Define training step
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    @tf.function
    def train_step(inp, tar):
        '''
        Training step for the transformer model.
        '''
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp, True, enc_padding_mask,
                                         combined_mask, dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients,
                                      transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_real, predictions)

    # Training loop
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(data.data_train):
            train_step(inp, tar)

            if batch % 50 == 0:
                print(f'Epoch {epoch + 1}, batch {batch}: '
                      f'loss {train_loss.result():.14f} '
                      f'accuracy {train_accuracy.result():.14f}')

        print(f'Epoch {epoch + 1}: '
              f'loss {train_loss.result():.14f} '
              f'accuracy {train_accuracy.result():.14f}')

    return transformer
