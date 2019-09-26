
# coding: utf-8

# In[1]:


import collections

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


# In[2]:


#학습에 필요한 hyper parameter
sequence_length = 4
learning_rate = 0.01
total_epoch = 200

#데이터 파일
file = "text/_dataforpractice.txt"
sequence_file = "text/_total.txt"
test_file = "text/test.txt" #text 폴더에 있는 다른 test 파일도 여기서 경로 변경해 테스트 가능합니다.


# In[3]:


#파일을 읽은 후, 텍스트를 반환한다.
def read_data(filename):
  with open(filename, 'r') as f:
      data = f.read().split()
      return data

#빈도수 순서로 단어를 인덱싱해 단어집을 생성한다.
def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()

  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0

  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)

  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

#학습 데이터를 인덱싱한다.
def indexing(_words):
    _idx = list()
    for word in _words:
        _idx.append(dictionary.get(word))
    _idx2char = list(set(_idx))
    _char2idx = {c: i for i, c in enumerate(_idx2char)}
    
    return _idx, _idx2char, _char2idx

#학습 시퀀스 데이터를 읽는다.
def read_sequence(file, _idx, _sequence_length):
    
    _dataX = []
    _dataY = []
    
    for i in range(0, len(_idx) - _sequence_length):
        temp = _idx[i:i + _sequence_length+1]
        temp_x = temp[:-1]

        _dataX.append(temp_x)
        
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp_y = line[-2]
            _dataY.append(temp_y)
            
    return _dataX, _dataY


#RNN으로 학습과 테스트를 한다.
def train_test_rnn(_sequence_length, _dic_size, _rnn_hidden_size, _learning_rate, _dataX, _dataY, test_path, _dictionary):
    #RNN 학습
    
    X = tf.placeholder(tf.float32, [None, _sequence_length, _dic_size])
    Y = tf.placeholder(tf.int32, [None])
   
    with tf.name_scope("layer") as scope:
        W = tf.get_variable("W1", shape=[_rnn_hidden_size, 2],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.random_normal([2]))
    
        #W_hist = tf.summary.histogram("weight", W)
        #b_hist = tf.summary.histogram("bias", b)
    
    with tf.variable_scope('MultiRNNCell') as scope:
        cell1 = tf.nn.rnn_cell.BasicRNNCell(rnn_hidden_size)
        cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.8)
        cell2 = tf.nn.rnn_cell.BasicRNNCell(rnn_hidden_size)
    
        multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
    
        outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
    
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = outputs[-1]
        model = tf.matmul(outputs, W) + b
        #model_hist = tf.summary.histogram("model", model)

    with tf.name_scope('cost') as scope:
        cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))
        
        #cost_sum = tf.summary.scalar("cost", cost)
    
    with tf.name_scope('train') as scope:
        optimizer = tf.train.AdamOptimizer(_learning_rate).minimize(cost)
    
    prediction = tf.cast(tf.argmax(model, 1), tf.int32)
    prediction_check = tf.equal(prediction, Y)
    accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))
    
    #accuracy_sum = tf.summary.scalar("accuracy", accuracy)
    
    sess = tf.Session()
    
    #merged = tf.summary.merge_all()
    #writer = tf.summary.FileWriter('Anaconda3/envs/tensorflow/mygraph2', sess.graph)
    
    sess.run(tf.global_variables_initializer())

    input_batch = []
    for x in _dataX:
        input_batch.append(np.eye(_dic_size)[x])

    target_batch = []
    for y in _dataY:
        target_batch.append(y)

    print('학습 시작!')
    
    for epoch in range(total_epoch):
        #summary, _ = sess.run([merged, optimizer], feed_dict={X: input_batch, Y: target_batch})
        _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
        print('Epoch:', '%04d' % (epoch + 1),'cost =', '{:.6f}'.format(loss))
        #writer.add_summary(summary, global_step=epoch)

    print('최적화 완료!')
    
    
    #학습한 모델 저장
    #saver = tf.train.Saver()
    #save_path = saver.save(sess, "C:/Users/USER1/Desktop/capstone/recruitment.ckpt")

    #print("\nModel Saved", save_path)
    
    #테스트 시작
    test_words = read_data(test_path)
    test_idx = list()
    
    test_idx, test_idx2char, test_char2idx = indexing(test_words)
    
    test_dataX = []
    
    for i in range(0, len(test_words) - _sequence_length):
        for j in range(0,_sequence_length):
            if test_idx[i+j] == None:
                test_idx[i+j] = 0
        test_temp = test_idx[i:i + 4]
        test_dataX.append(test_temp)
        
    test_input_batch = []
    
    for x in test_dataX:
        temp = np.eye(_dic_size)[x]
        test_input_batch.append(temp)
    
    
    #print(test_input_batch)
    
    predict2 = sess.run([prediction], feed_dict={X:test_input_batch})
    
    idxx = -1
    print('\n기업명: ')
    for k in range(len(predict2[0])):
        if predict2[0][k] == 1:
            idxx = k
            print(reverse_dictionary[test_dataX[idxx][0]])


# In[4]:


words = read_data(file)
print('data size: ', len(words),' \n')

vocabulary_size = 50000

data, count, dictionary, reverse_dictionary = build_dataset(words)

print('Most common words (+UNK)', count[:10], '\n')
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]], '\n')

print("dictionary length: ", len(dictionary), '\n')

idx, idx2char, char2idx = indexing(words)

dic_size = len(dictionary)  # RNN input size (one hot size)
rnn_hidden_size = len(char2idx)  # RNN output size
num_classes = len(char2idx)

dataX, dataY = read_sequence(sequence_file, idx, sequence_length)

train_test_rnn(sequence_length, dic_size, rnn_hidden_size, learning_rate, dataX, dataY, test_file, dictionary)

