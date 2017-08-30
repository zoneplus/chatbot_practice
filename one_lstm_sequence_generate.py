#coding:utf-8
""" A neural chatbot using sequence to sequence model with
attentional decoder. 

This practice is based on Google Translate Tensorflow model and ChatBotCourse.

This file contains the code to process load files,training of models and prediction function.
"""
import sys
from tqdm import tqdm
import numpy as np
import math
import tflearn
import chardet
import numpy as np
import struct
import gensim
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn
from gensim.models import word2vec  
seq = []
max_w = 50
float_size = 4
word_vector_dict = {}
word_vec_dim = 300
max_seq_len = 16

def load_word_set(input):
    """加载分好词的源文件，将问句与答句分开，返回问答字典
    """
    file_object = open(input, 'r')
    while True:
        line = file_object.readline()
        if line:
            line_pair = line.split('|')
            line_question = line_pair[0]
            line_answer = line_pair[1]
            for word in line_question.decode('utf-8').split(' '):
                word_set[word] = 1
            for word in line_answer.decode('utf-8').split(' '):
                word_set[word] = 1
        else:
            break
    file_object.close()

def load_vectors(input):
    """从vectors.bin加载词向量，返回一个word_vector_dict的词典，key是词，value是200维的向量
    """
    print ("begin load vectors")
    f = open(input,"r")
    for line in tqdm(f):
     values = line.split()
     word = values[0]
     coefs = values[1].split(',')
     word_vector_dict[word] = coefs
    f.close()
    print( "load vectors finish")

def init_seq():
    """读取切好词的文本文件，加载全部词序列
    """
    file_object = open('chatbox/vector_seg_sub.txt', 'r')
    vocab_dict = {}
    while True:
        line = file_object.readline()
        if line:
            for word in line.split(' '):
              if word:
                if word in  word_vector_dict.keys():
                    tup = ()
                    tup = (word,word_vector_dict[word])
                    seq.append(tup)
  
        else:
            break
    file_object.close()

def vector_sqrtlen(vector):
    len = 0
    
    for item in vector:
        len += float(item) * float(item)
    len = math.sqrt(len)
    return len

def vector_cosine(v1, v2):
    if len(v1) != len(v2):
        sys.exit(1)
    sqrtlen1 = vector_sqrtlen(v1)
    sqrtlen2 = vector_sqrtlen(v2)
    value = 0
    for item1, item2 in zip(v1, v2):
        value += float(item1) * float(item2)
    return value / (sqrtlen1*sqrtlen2)


def vector2word(vector):
    max_cos = -10000
    match_word = ''
    for word in word_vector_dict:
        v = word_vector_dict[word]
        cosine = vector_cosine(vector, v)
        if cosine > max_cos:
            max_cos = cosine
            match_word = word
    return (match_word, max_cos)
class MySeq2Seq(object):
    """
    思路：输入输出序列一起作为input，然后通过slick和unpack切分
    完全按照论文说的编码器解码器来做
    输出的时候把解码器的输出按照词向量的200维展平，这样输出就是(?,seqlen*200)
    这样就可以通过regression来做回归计算了，输入的y也展平，保持一致
    """
    def __init__(self, max_seq_len = 16, word_vec_dim = 200):
        self.max_seq_len = max_seq_len
        self.word_vec_dim = word_vec_dim

    def generate_trainig_data_static(self):
        load_vectors("chatbox/vector_test_sub.txt")
        init_seq()
        xy_data = []
        y_data = []
        for i in range(30,40,10):
            # 问句、答句都是16字，所以取32个
            start = i*self.max_seq_len*2
            middle = i*self.max_seq_len*2 + self.max_seq_len
            end = (i+1)*self.max_seq_len*2
            sequence_xy =[s[1] for s in seq[start:end]]
            sequence_y =[ s[1] for s in  seq[middle:end]]
            print([s[0] for s in seq[start:end]],[s[0] for s in seq[middle:end]])
            print( "right answer")
            for w in sequence_y:
                (match_word, max_cos) = vector2word(w)
                print( match_word)
            sequence_y = [np.ones(self.word_vec_dim)] + sequence_y
            xy_data.append(sequence_xy)
            y_data.append(sequence_y)
        return np.array(xy_data), np.array(y_data)

    def generate_trainig_data_dynamic(self):
        load_word_set()
        load_vectors("chatbox/vector_test_sub.txt")
        init_seq(self.input_file)
        xy_data = []
        y_data = []
        for i in range(len(question_seqs)):
            question_seq = question_seqs[i]
            answer_seq = answer_seqs[i]
            if len(question_seq) < self.max_seq_len and len(answer_seq) < self.max_seq_len:
                sequence_xy = [np.zeros(self.word_vec_dim)] * (self.max_seq_len-len(question_seq)) + list(reversed(question_seq))
                sequence_y = answer_seq + [np.zeros(self.word_vec_dim)] * (self.max_seq_len-len(answer_seq))
                sequence_xy = sequence_xy + sequence_y
                sequence_y = [np.ones(self.word_vec_dim)] + sequence_y
                xy_data.append(sequence_xy)
                y_data.append(sequence_y)

        return np.array(xy_data), np.array(y_data)


    def model(self, feed_previous=False):
        # 通过输入的XY生成encoder_inputs和带GO头的decoder_inputs
        input_data = tflearn.input_data(shape=[None, self.max_seq_len*2, self.word_vec_dim], dtype=tf.float32, name = "XY")
        encoder_inputs = tf.slice(input_data, [0, 0, 0], [-1, self.max_seq_len, self.word_vec_dim], name="enc_in")
        decoder_inputs_tmp = tf.slice(input_data, [0, self.max_seq_len, 0], [-1, self.max_seq_len-1, self.word_vec_dim], name="dec_in_tmp")
        go_inputs = tf.ones_like(decoder_inputs_tmp)
        go_inputs = tf.slice(go_inputs, [0, 0, 0], [-1, 1, self.word_vec_dim])
        decoder_inputs = tf.concat(axis=1,values= [go_inputs, decoder_inputs_tmp], name="dec_in")

        # 编码器
        # 把encoder_inputs交给编码器，返回一个输出(预测序列的第一个值)和一个状态(传给解码器)
        (encoder_output_tensor, states) = tflearn.lstm(encoder_inputs, self.word_vec_dim, return_state=True, scope='encoder_lstm')
        encoder_output_sequence = tf.stack([encoder_output_tensor], axis=1)

        # 解码器
        # 预测过程用前一个时间序的输出作为下一个时间序的输入
        # 先用编码器的最后一个输出作为第一个输入
        if feed_previous:
            first_dec_input = go_inputs
        else:
            first_dec_input = tf.slice(decoder_inputs, [0, 0, 0], [-1, 1, self.word_vec_dim])
        decoder_output_tensor = tflearn.lstm(first_dec_input, self.word_vec_dim, initial_state=states, return_seq=False, reuse=False, scope='decoder_lstm')
        decoder_output_sequence_single = tf.stack([decoder_output_tensor], axis=1)
        decoder_output_sequence_list = [decoder_output_tensor]
        # 再用解码器的输出作为下一个时序的输入
        for i in range(self.max_seq_len-1):
            if feed_previous:
                next_dec_input = decoder_output_sequence_single
            else:
                next_dec_input = tf.slice(decoder_inputs, [0, i+1, 0], [-1, 1, self.word_vec_dim])
            decoder_output_tensor = tflearn.lstm(next_dec_input, self.word_vec_dim, return_seq=False, reuse=True, scope='decoder_lstm')
            decoder_output_sequence_single = tf.stack([decoder_output_tensor], axis=1)
            decoder_output_sequence_list.append(decoder_output_tensor)

        decoder_output_sequence = tf.stack(decoder_output_sequence_list, axis=1)
        real_output_sequence = tf.concat(axis=1, values=[encoder_output_sequence, decoder_output_sequence])

        net = tflearn.regression(real_output_sequence, optimizer='sgd', learning_rate=0.1, loss='mean_square')
        model = tflearn.DNN(net)
        return model

    def train(self):
        trainXY, trainY = self.generate_trainig_data()
        model = self.model(feed_previous=False)
        model.fit(trainXY, trainY, n_epoch=1000, snapshot_epoch=False)
        #model.save('./model')
        return model

    def load(self):
        model = self.model(feed_previous=True)
        model.load('./model')
        return model

if __name__ == '__main__':
    #phrase = sys.argv[0]
    my_seq2seq = MySeq2Seq(word_vec_dim=word_vec_dim, max_seq_len=max_seq_len)
    #if phrase == 'train':
    #    my_seq2seq.train()
    #else:
    model = my_seq2seq.train()
    trainXY, trainY = my_seq2seq.generate_trainig_data()
    #print(trainXY[0])
    predict = model.predict(trainXY)
    for sample in predict:
            print( "predict answer")
            for w in sample[1:]:
                (match_word, max_cos) = vector2word(w)
                print (match_word, max_cos)

 