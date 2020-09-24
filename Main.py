#!/usr/bin/env python
# coding: utf-8

# In[5]:


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

tf.logging.set_verbosity

import tensorflow as tf
import model as ml
import data
import numpy as np
import os
import sys

from configs import DEFINES

DATA_OUT_PATH = './data_out/'

def main(self):
    data_out_path = os.path.join(os.getcwd(), DATA_OUT_PATH)
    os.makedirs(data_out_path, exist_ok=True)

    word2idx, idxword, vocabulary_length = data.load_vocabulary()
    train_input, train_label, eval_input, eval_label = data.load_data()
    train_input_enc, train_input_enc_length = data.enc_processing(train_input, word2idx)
    train_input_dec, train_input_dec_length = data.dec_input_processing(train_label, word2idx)
    train_target_dec = data.dec_target_processing(train_label, word2idx)

    eval_input_enc, eval_input_enc_length = data.enc_processing(eval_input, word2idx)
    eval_input_dec, eval_input_dec_length = data.dec_input_processing(eval_label, word2idx)
    eval_target_dec = data.dec_target_processing(eval_label, word2idx)

    check_point_path = os.path.join(os.getcwd(), DEFINES.check_point_path)
    os.makedirs(check_point_path, exist_ok=True)

    classifier = tf.estimator.Estimator(
        model_fn=ml.model,
        model_dir=DEFINES.check_point_path,
        params={
            'hidden_size': DEFINES.hidden_size,
            'layer_size': DEFINES.layer_size,
            'learning_rate': DEFINES.learning_rate,
            'vocabulary_length': vocabulary_length,
            'embedding_size': DEFINES.embedding_size,
            'embedding': DEFINES.embedding,
            'multilayer': DEFINES.multilayer,
        })

    classifier.train(input_fn=lambda:data.train_input_fn(
        train_input_enc, train_input_dec, train_target_dec, DEFINES.batch_size),
    steps=DEFINES.train_steps)

    eval_result = classifier.evaluate(input_fn=lambda:data.eval_input_fn(
        eval_input_enc, eval_input_dec, eval_target_dec, DEFINES.batch_size))
    print('\nEVAL set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    predic_input_enc, predic_input_enc_length = data.enc_processing(["가끔 궁금해"], word2idx)
    predic_input_dec, predic_input_decLength = data.dec_input_processing([""], word2idx)
    predic_target_dec = data.dec_target_processing([""], word2idx)

    predictions = classifier.predict(
    input_fn=lambda:data.eval_input_fn(predic_input_enc, predic_input_dec, predic_target_dec, DEFINES.batch_size))

    data.pred2string(predictions, idx2word)


# In[ ]:
