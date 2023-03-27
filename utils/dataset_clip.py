import os
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import h5py
import PIL
import csv
import clip
from nltk.corpus import wordnet as wn

class SemevalDataset(Dataset):
	def __init__(self, data_dir, model, preprocess, device, data='trial'):
		super(SemevalDataset, self).__init__()
		self.data_dir = data_dir
		self.img_dir = data_dir + data + '_v1/' + data + '_images_v1/'
		self.text_file = data_dir + data + '_v1/' + data + '.data.v1.txt'
		self.label_file = data_dir + data + '_v1/' + data + '.gold.v1.txt'
		self.definition_file = data_dir + data + '_v1/' + data + '.gpt.prompt1s.txt'

		self.model = model
		self.preprocess = preprocess
		self.device = device
		self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)


		with open(self.text_file) as f:
			reader = csv.reader(f, delimiter="\t")
			self.data = list(reader)

		with open(self.label_file) as f:
			reader = csv.reader(f, delimiter="\t")
			self.label = list(reader)

		with open(self.definition_file) as f:
			reader = csv.reader(f)
			self.definition = list(reader)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		"""
		:param index: the index of the item
		:return: a tuple of (text, image_list, label)

		"""
		text_1_t = self.data[index][0]
		text_2_t = self.data[index][1]
		text_4_t = ''.join(i for i in self.definition[index])
		# text_4_t = text_2_t

		text_1_def_t = [i.definition() for i in wn.synsets(text_1_t)]
		# print(text_1_def_t)
		text_1_def = [clip.tokenize([i]) for i in text_1_def_t]

		text_1 = clip.tokenize([text_1_t])
		text_2 = clip.tokenize([text_2_t])
		text_4 = clip.tokenize([text_4_t])

		text_features_1 = self.model.encode_text(text_1)
		text_features_2 = self.model.encode_text(text_2)
		text_features_4 = self.model.encode_text(text_4)

		text_features_2_rep = text_features_2.repeat(len(text_1_def), 1)
		text_features_1_def = [self.model.encode_text(i) for i in text_1_def]
		text_features_1_def = torch.stack(text_features_1_def)
		text_features_1_def = text_features_1_def.squeeze()

		cos_sim = self.cos_sim(text_features_2_rep, text_features_1_def)
		cos_sim = cos_sim.cpu().numpy()
		def_idx = np.argsort(cos_sim)[::-1]
		def_rank = [text_1_def_t[i] for i in def_idx]
		# print(def_rank[0])
		# print(text_2_t)

		text_feature_3 = text_features_1_def[0].unsqueeze(0)
		# text_feature_3 = torch.cat((text_features_1, text_features_2, text_feature_3), dim=1)

		image_list = self.data[index][2:]
		label = self.label[index][0]
		images = []
		image_id = []
		for i in image_list:
			image = self.preprocess(PIL.Image.open(self.img_dir + i)).unsqueeze(0)
			images.append(image)
			image_id.append(int(i.split('.')[1]))
		image_id = torch.Tensor(image_id)
		label_id = torch.Tensor([int(label.split('.')[1])])
		image_features = []
		for i in images:
			image_feature = self.model.encode_image(i)
			image_features.append(image_feature)

		image_features = torch.cat(image_features)

		return text_features_1, text_features_2, text_feature_3, text_features_4, image_features, image_id, label_id


def collate_fn_eval(batch):
	all_text_features_1, all_text_features_2, all_text_features_3, all_text_features_4, all_image_feature, all_image_list, all_label = [], [], [], [], [], [], []
	for i, (text_features_1, text_features_2, text_features_3, text_features_4, image_feature, image_list, label) in enumerate(batch):
		all_text_features_1.append(text_features_1)
		all_text_features_2.append(text_features_2)
		all_text_features_3.append(text_features_3)
		all_text_features_4.append(text_features_4)

		all_image_feature.append(image_feature[None])
		all_image_list.append(image_list[None])
		all_label.append(label[None])

	all_text_features_1 = torch.cat(all_text_features_1)
	all_text_features_2 = torch.cat(all_text_features_2)
	all_text_features_3 = torch.cat(all_text_features_3)
	all_text_features_4 = torch.cat(all_text_features_4)

	all_image_feature = torch.cat(all_image_feature)
	all_image_list = torch.cat(all_image_list)
	all_label = torch.cat(all_label)

	out = (all_text_features_1, all_text_features_2, all_text_features_3, all_text_features_4, all_image_feature, all_image_list, all_label)

	return out





