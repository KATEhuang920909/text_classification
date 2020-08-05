#-*- coding:utf-8 -*-

import tensorflow as tf
from models.attention_bilstm import BiLSTM
from utils import InputHelper
import time
import os
import numpy as np
from gensim.models import Word2Vec
# Parameters
# =================================================
tf.flags.DEFINE_integer('embedding_size', 256, 'embedding dimension of tokens')
tf.flags.DEFINE_integer('rnn_size', 100, 'hidden units of RNN , as well as dimensionality of character embedding (default: 100)')
tf.flags.DEFINE_float('dropout_keep_prob', 0.7, 'Dropout keep probability (default : 0.5)')
tf.flags.DEFINE_integer('layer_size', 2, 'number of layers of RNN (default: 2)')# 双层rnn
tf.flags.DEFINE_integer('batch_size', 128, 'Batch Size (default : 32)')
tf.flags.DEFINE_integer('sequence_length', 300, 'Sequence length (default : 32)')
tf.flags.DEFINE_integer('attn_size', 200, 'attention layer size')
tf.flags.DEFINE_float('grad_clip', 5.0, 'clip gradients at this value')
tf.flags.DEFINE_integer("num_epochs", 10, 'Number of training epochs (default: 200)')
tf.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
tf.flags.DEFINE_string('data_dir', 'data', 'data directory')
tf.flags.DEFINE_string('save_dir', 'save', 'model saved directory')
tf.flags.DEFINE_string('logs_dir', 'log', 'log info directiory')
tf.flags.DEFINE_string('pre_trained_vec_path',
                       'D:/learning/text_classification_self_learning/pre_train_model',
                       'using pre trained word embeddings, npy file format')
tf.flags.DEFINE_string('init_from', None, 'continue training from saved model at this path')
tf.flags.DEFINE_integer('save_steps', 1000, 'num of train steps for saving model')
tf.flags.DEFINE_integer('vocab_size',None,'vocabulary size')
tf.flags.DEFINE_integer('n_classes',None,'the number of class')
tf.flags.DEFINE_integer('num_batches',None,'batch number')
FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
def train():
    data_loader = InputHelper()
    # 创建词典
    data_loader.load_file()
    data_loader.create_dictionary_v2(FLAGS.save_dir+'/')
    x_train=data_loader.data_token(data_loader.x_train)
    data_loader.create_batches(x_train,data_loader.y_train, FLAGS.batch_size, FLAGS.sequence_length)
    FLAGS.vocab_size = data_loader.vocab_size
    FLAGS.n_classes = data_loader.n_classes
    FLAGS.num_batches = data_loader.num_batches
    test_data_loader = InputHelper()
    test_data_loader.load_dictionary(FLAGS.save_dir+'/dictionary',data_loader.y_train)

    x_test = data_loader.data_token(data_loader.x_test)
    test_data_loader.create_batches(x_test,data_loader.y_test, 100, FLAGS.sequence_length)
    embeddings_reshape=None
    if FLAGS.pre_trained_vec_path:
        print('将原始的embedding矩阵重置')
        embeddings = np.load(FLAGS.pre_trained_vec_path+'/word2vec.model.wv.vectors.npy',allow_pickle=True)
        model = Word2Vec.load(FLAGS.pre_trained_vec_path+'/word2vec.model')
        embeddings_reshape=np.zeros(embeddings.shape)
        print('embeddings_shape:',embeddings_reshape.shape)
        dic=data_loader.token_dictionary
        print(len(dic))
        i=20
        for word in model.wv.index2word:
            tmp=dic[word]
            if tmp<i:
                i=tmp
                print(i)
            embeddings_reshape[tmp]=model.wv[word]
        #print(embeddings_reshape[0])
        #print(embeddings_reshape[dic['e850']])
        """
        embeddings_reshape = tf.get_variable(name="W", shape=embeddings_reshape.shape,
                            initializer=tf.constant_initializer(embeddings_reshape),
                            trainable=False)
        """

        print (embeddings_reshape.shape)
        FLAGS.vocab_size = embeddings_reshape.shape[0]
        FLAGS.embedding_size = embeddings_reshape.shape[1]
    '''
    if FLAGS.init_from is not None:#   fine tune condition
            #断言，传参数前捕获参数异常
            assert os.path.isdir(FLAGS.init_from), '{} must be a directory'.format(FLAGS.init_from)
            ckpt = tf.train.get_checkpoint_state(FLAGS.init_from)
            assert ckpt,'No checkpoint found'
            assert ckpt.model_checkpoint_path,'No model path found in checkpoint'
    '''

    print('create model...')
    # Define specified Model
    model = BiLSTM(embedding_size=FLAGS.embedding_size, rnn_size=FLAGS.rnn_size,
        vocab_size=FLAGS.vocab_size,  sequence_length=FLAGS.sequence_length,
        n_classes=FLAGS.n_classes, learning_rate=FLAGS.learning_rate,embedding_w=embeddings_reshape)

    # define value for tensorboard
    tf.summary.scalar('train_loss', model.loss)
    tf.summary.scalar('accuracy', model.accuracy)
    merged = tf.summary.merge_all()

    # 调整GPU内存分配方案
    #tf_config = tf.ConfigProto()
    #tf_config.gpu_options.allow_growth = True
    init=tf.global_variables_initializer()
    print('start training...')
    with tf.Session() as  sess:    #tf.Session(config=tf_config) as sess:
        train_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)


        saver = tf.train.Saver(tf.global_variables())

        # using pre trained embeddings
        # if FLAGS.pre_trained_vec_path:
        #     sess.run(model.embedding.assign(embeddings_reshape))#替换为embeddings
        #     del embeddings
        #     del embeddings_reshape

        # restore model
        sess.run(init)
        sess.run(tf.local_variables_initializer())
        total_steps = FLAGS.num_epochs * FLAGS.num_batches

        for e in range(FLAGS.num_epochs):
            data_loader.reset_batch()  # 重新洗牌
            for b in range(FLAGS.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                #print(x.shape,y.shape)
                #print(x[0],y[0])

                feed = {model.input_data:x, model.targets:y, model.output_keep_prob:FLAGS.dropout_keep_prob}
                train_loss, summary,  _,accuracy = sess.run([model.loss, merged, model.train_op,model.accuracy], feed_dict=feed)
                end = time.time()

                global_step = e * FLAGS.num_batches + b

                print('{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f},acc = {:.3f}'.format(global_step,
                            total_steps,
                            e, train_loss, end - start,accuracy))

                if global_step % 20 == 0:
                    train_writer.add_summary(summary, e * FLAGS.num_batches + b)

                if global_step % FLAGS.save_steps == 0:
                    checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=global_step)
                    print ('model saved to {}'.format(checkpoint_path))
            test_data_loader.reset_batch()
            test_accuracy = []
            for i in range(test_data_loader.num_batches):
                test_x, test_y = test_data_loader.next_batch()
                feed = {model.input_data: test_x, model.targets: test_y, model.output_keep_prob: 1.0}
                accuracy = sess.run(model.accuracy, feed_dict=feed)
                test_accuracy.append(accuracy)
            print(np.average(test_accuracy))





if __name__ == '__main__':
    train()

## accuracy :0.9373399