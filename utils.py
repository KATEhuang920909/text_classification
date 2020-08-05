#-*- coding:utf-8 -*-
"""
用于文本分类任务
train_file为已经分好词的文本 如 'token1 token2 ... \t label' 
token之间使用空格分开, 与label使用\t隔开
'''
Author: huangkai
date: 2020/1/10 15:29
'''
"""
from tqdm import tqdm
import numpy as np
import pickle
import tensorflow as tf
import jieba
class InputHelper():

	def __init__(self):
		self.stop_path=None
		self.def_path=None
		self.x_train=None
		self.y_train=None
		self.x_test=None
		self.y_test=None
	def load_file(self,):
		with open('data/x_train.txt', encoding='utf-8') as f:
			x_train = f.readlines()
			self.x_train = [k.strip() for k in x_train]  # .split('/t')
		with open('data/y_train.txt', encoding='utf-8') as f:
			y_train = f.readlines()
			self.y_train = [k.strip() for k in y_train]  # .split('/t')
		with open('data/y_test.txt', encoding='utf-8') as f:
			y_test = f.readlines()
			self.y_test = [k.strip() for k in y_test]  # .split('/t')
		with open('data/x_test.txt', encoding='utf-8') as f:
			x_test = f.readlines()
			self.x_test = [k.strip() for k in x_test]

		self.y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
		self.y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)
		print(len(self.x_train), len(self.y_train), len(self.x_test), len(self.y_test))
	def create_dictionary(self, train_file, save_dir):
		"""
		从原始文本文件中创建字典
		train_file : 原始训练数据文档
		save_dir : 词典保存路径
		"""
		token_dictionary = {}
		token_index = 0

		label_dictionary = {}
		label_index = 0

		labels = []

		for line in open(train_file):
			# 使用unicode编码
			text, label = line.rstrip().split('\t')
			tokens = text.split(' ')
			if label not in label_dictionary:
				label_dictionary[label] = label_index
				labels.append(label)
				label_index += 1

			for token in tokens:
				if token not in token_dictionary:
					token_dictionary[token] = token_index
					token_index += 1


		token_dictionary['</s>'] = token_index
		token_index += 1
		self.vocab_size = len(token_dictionary)
		self.n_classes = len(label_dictionary)
		print ('Corpus Vocabulary:{0}, Classes:{1}'.format(self.vocab_size, self.n_classes))

		with open(save_dir+'dictionary', 'wb') as f:
			pickle.dump((token_dictionary, label_dictionary), f)

		self.token_dictionary = token_dictionary
		self.label_dictionary = label_dictionary
		self.labels = labels

	def data_token(self,data):# [sent1,sent2..]

		if self.def_path:
			jieba.load_userdict(self.def_path)
		if self.stop_path:
			data_tokenize = []
			stop_words =open(self.stop_path,encoding='utf-8').readlines()

			for sentence in tqdm(data):
				tmp_sentence=[]
				for word in jieba.cut(sentence,cut_all=False):
					if word not in stop_words:
						tmp_sentence.append(word)
				data_tokenize.append(tmp_sentence)
			return  data_tokenize
		data_tokenize=[jieba.lcut(sent,cut_all=False) for sent in data]
		return data_tokenize
	def create_dictionary_v2(self,  save_dir):
		"""
		从原始文本文件中创建字典
		token_dictionary : {word:index}
		save_dir : 词典保存路径
		:param save_dir:
		:return:
		"""
		token_dictionary = {}
		token_index = 0



		texts = self.x_train+self.x_test#,self.y_train
		tokens = self.data_token(texts)

		for sent in tokens:
			for token in sent:
				if token not in token_dictionary:
					token_dictionary[token] = token_index
					token_index += 1


		#token_dictionary['</s>'] = token_index
		#token_index += 1
		#print('vocab_size:',len(token_dictionary))
		self.vocab_size = len(token_dictionary)
		self.n_classes = self.y_train.shape[1]

		print ('Corpus Vocabulary:{0}, Classes:{1}'.format(self.vocab_size, self.n_classes))
		#print(token_dictionary[0])
		with open(save_dir+'dictionary', 'wb') as f:
			pickle.dump(token_dictionary, f)

		self.token_dictionary = token_dictionary
	def load_dictionary(self, dictionary_file,y_train):
		print('load dictionary...')
		with open(dictionary_file,'rb') as f:
			self.token_dictionary = pickle.load(f)
			self.vocab_size = len(self.token_dictionary)
			self.n_classes = y_train.shape[1]
			print('vocab_size: {},classes_size: {}'.format(self.vocab_size,self.n_classes))



	def create_batches(self, x_data,y_data, batch_size, sequence_length):

		self.x_data = []
		padding_index = 0
		for i in range(len(x_data)):
			#line = line.decode('utf-8')
			seq_ids = [self.token_dictionary.get(token) for token in x_data[i] if self.token_dictionary.get(token) is not None]
			seq_ids = seq_ids[:sequence_length]
			for _ in range(len(seq_ids), sequence_length):
				seq_ids.insert(0,padding_index)#padding

			self.x_data.append(seq_ids)
			#self.y_data.append(self.label_dictionary.get(y_data[i]))#  label_index

		self.num_batches = int(len(self.x_data) / batch_size)
		self.x_data = self.x_data[:self.num_batches * batch_size]
		self.y_data = y_data[:self.num_batches * batch_size]

		self.x_data = np.array(self.x_data, dtype=int)
		# shape like: (num_batches,batch_size,sequence_length)
		self.x_batches = np.split(self.x_data.reshape(batch_size, -1), self.num_batches, 1)
		self.y_batches = np.split(self.y_data.reshape(batch_size, -1), self.num_batches, 1)
		self.pointer = 0

	def label_one_hot(self, label_id):

		y = [0] * self.n_classes
		y[int(label_id)] = 1.0

		return np.array(y)

	def next_batch(self):
		index = self.batch_index[self.pointer]
		self.pointer += 1		
		x_batch, y_batch = self.x_batches[index], self.y_batches[index]
		#y_batch = [self.label_one_hot(y) for y in y_batch]
		return x_batch, y_batch

	def reset_batch(self):
		self.batch_index = np.random.permutation(self.num_batches)# 打乱顺序
		self.pointer = 0

	def transform_raw(self, text, sequence_length):

		#if not isinstance(text, unicode):
		#	text = text.decode('utf-8')

		x = [self.token_dictionary.get(token) for token in text]
		x = x[:sequence_length]
		padding_index = self.vocab_size - 1
		for _ in range(len(x), sequence_length):
			x.append(padding_index)

		return x


if __name__ == '__main__':
	data_loader = InputHelper()