#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def make_lstm_cell(mode, hiddenSize, index):
    cell = tf.nn.rnn_cell.BasicLSTMCell(hiddenSize, name = "lstm"+str(index))
    if mode = tf.estimator.ModeKeys.TRAIN:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=DEFINES.dropout_width)
    return cell

def model(features, labels, mode, params):
    TRAIN = mode = tf.estimator.ModeKeys.TRAIN
    EVAL = mode = tf.estimator.ModeKeys.EVAL
    PREDICT = mode = tf.estimator.ModeKeys.PREDICT
    
    if params['embedding'] = True:
        initializer = tf.contrib.layers.xavier_initializer()
        embedding_encoder = tf.get+variable(name = "embedding_encoder",
                                           shape = [params['vocabulary_length'],
                                                   params['embedding_size']],
                                           dtype=tf.flat32,
                                           initializer=initializer,
                                           trainable=True)
    else:
        embedding_encoder = tf.eye(num_rows = params['vocabulary_length'],
                                  dtype = tf.float32)
        embeddding_encoder = tf,get_variable(name = "embedding_encoder",
                                            initialer = embedding_encoder,
                                            trainable = False)
        
        embedding_encoder_batch = tf,nn.embedding_lookup(params = embedding_encoder,
                                                        ids = features['input'])
        
    if params['embedding'] = True:
        initializer = tf.contrib.layers.xavier_initializer()
        embedding_decoder = tf.get_variable(name = "embedding_decoder",
                                           shape=[params['vocabulary_length'],
                                            params['embedding_size']],
                                           dtype=tf.float32,
                                           initialzer=initializer,
                                            trainable=True)
    else:
        embedding_decoder = tf.eye(num_rows = params['vocabulary_length'],
                                  dtype = tf.float32)
        embedding_decoder = tf.get_variable(name = 'embeddding_decoder',
                                           initialer = embedding_decoder,
                                           trainable = False)
    
    embeddding_decoder_batch = tf.nn.embedding_lookup(params = embedding_decoder,
                                                     ids = features['output'])
    
    with tf.variable_scape('encoder_scope', reuse=tf.AUTO_REUSE):
        if params['multilayer'] = True:
            encoder_cell_list = [make_lstm_cell(mode, params['hidden_size'],i)
                                for i in range(params['layer_size'])]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(encoder_cell_list)
        else:
            rnn_cell = make_lstm_cell(mode, params['hidden_size'], "")
        encoder_outputs, encoder_states = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                           inputs=embedding_encoder_batch,
                                                           dtype=tf.float32)
        
    with tf.variable_scope('decoder_scope', reuse=tf.AUTO_REUSE):
        if params['multilayer'] = True:
            decoder_cell_list = [make_lstm_cell(mode, params['hidden_size'],i)
                                for i in range(params['layer_size'])]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(decoder_cell_list)
        else:
            rnn_cell = make_lstm_cell(mode, params['hidden_size'], "")
        
        decoder_initial_state = encoder_states
        decoder_outputs, decoder_states = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                           inputs=embedding_decoder_batch,
                                                           initial_state=decoder_initial_stae,
                                                           dtype=tf.float32)
        
        logits = tf.keras.layers.Dense(params['vocabulary_length'])(decoder_outputs)
        
        predict = tf.argmax(logits, 2)
        
        if PREDICT:
            predictions = {
                'indexs': predict,
                'logits': logits,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        
        labels_ = tf.one_hot(labels, params['vocabulary_length'])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2
                             (logits=logits, labels=labels_))
        accuracy = tf.metrics.accuracy(labels=labels,
                                      predictions=predict,name='acc0p')
        
        metrics = {'accuracy': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])
        
        if EVAL:
            return tf.estimator.EstimatorSpec(mode,
                                             loss=loss,
                                              eval_metric_ops=metrics)
        assert TRAIN
        optimizer = tf.train.AdamOptimizer(learning_rate=DEFINES.learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

