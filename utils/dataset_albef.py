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
from nltk.corpus import wordnet as wn
from collections import defaultdict

from lavis.models import load_model_and_preprocess
import openai
#openai.api_key = 'xxxxx'  # enter your openai api here
PIL.Image.MAX_IMAGE_PIXELS = 1000000000


class SemevalDataset(Dataset):
	def __init__(self, data_dir, model, vis_processor, txt_processor, device, language):
		super(SemevalDataset, self).__init__()
		self.data_dir = data_dir

		self.img_dir = data_dir + 'test_images/'
		self.text_file = data_dir + 'test.data.v1.1.gold/'+language+'.test.data.v1.1.txt'
		self.label_file = data_dir + 'test.data.v1.1.gold/'+language+'.test.gold.v1.1.txt'
		self.gpt_file_ab = data_dir + 'test.data.v1.1.gpt/' + language+'.gpt.ab.txt'
		self.wn_file_1 = data_dir + 'test.data.v1.1.wordnet/' + language+'.test.wn.v1.1.txt'
		self.wn_file_2 = data_dir + 'test.data.v1.1.wordnet/' + language+'.test.wn.v2.1.txt'

		self.model = model
		self.vis_processor = vis_processor
		self.txt_processor = txt_processor
		self.device = device
		self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

		with open(self.text_file) as f:
			reader = csv.reader(f, delimiter="\t")
			self.data = list(reader)

		with open(self.label_file) as f:
			reader = csv.reader(f, delimiter="\t")
			self.label = list(reader)

		with open(self.gpt_file_ab) as f:
			reader = csv.reader(f, quotechar=None)
			self.gpt = []
			for line in reader:
				self.gpt.append(line)

		with open(self.wn_file_1) as f:
			reader = csv.reader(f, quotechar=None)
			self.wn_1 = []
			for line in reader:
				self.wn_1.append(line)

		with open(self.wn_file_2) as f:
			reader = csv.reader(f, quotechar=None)
			self.wn_2 = []
			for line in reader:
				self.wn_2.append(line)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		"""
		:param index: the index of the item
		:return: a tuple of (text, image_list, label)

		"""

		# text_a_t = "A picture of " + self.data[index][0]+ '.' # angora
		# text_ab_t = "A picture of " + self.data[index][1]+ '.' # angora city

		text_a_t = self.data[index][0]  # angora
		text_ab_t = self.data[index][1] # angora city
		text_prompt_ab_t = "A picture of " + self.data[index][1] + '.'  # prompt + angora city

		text_wn_c_t = self.data[index][1] + '.' + ''.join(i for i in self.wn_2[index])  # prompt + angora city
		text_wn_t_t = self.data[index][1] + '.' + ''.join(i for i in self.wn_1[index])  # prompt + angora city
		text_wn_prompt_t_t = "A picture of " + self.data[index][1] + '.' +self.data[index][1] + '.' + ''.join(i for i in self.wn_1[index])  # prompt + angora city

		text_gpt_ab = ''.join(i for i in self.gpt[index])  # get description from file
		text_prompt_gpt_ab = "A picture of " + self.data[index][1] + '.' + ''.join(i for i in self.gpt[index])  # get description from file
		text_sample_a = self.txt_processor["eval"](text_a_t)
		text_sample_ab = self.txt_processor["eval"](text_ab_t)
		text_sample_prompt_ab = self.txt_processor["eval"](text_prompt_ab_t)
		text_sample_wn_c = self.txt_processor["eval"](text_wn_c_t)
		text_sample_wn_t = self.txt_processor["eval"](text_wn_t_t)
		text_sample_prompt_wn_t = self.txt_processor["eval"](text_wn_prompt_t_t)
		text_sample_gpt_ab = self.txt_processor["eval"](text_gpt_ab)
		text_sample_prompt_gpt_ab = self.txt_processor["eval"](text_prompt_gpt_ab)

		# text_phrase = self.data[index][1]
		# des_phrase = text_phrase.replace(self.data[index][0], '')
		# text_ba_t = "A picture of " + des_phrase + self.data[index][0] + '.'  # city angora
		# text_b_t = "A picture of " + des_phrase + '.'  # city
		# text_des_t_a = "A picture of " + self.data[index][0] + '.' + ''.join(
		# 	i for i in self.definition_a[index])  # get a description from file

		# text_des_t_b = "A picture of " + des_phrase + '.' + ''.join(
		# 	i for i in self.definition_b[index])  # get b description from file

		# text_des_t_ba = "A picture of " + des_phrase + ''.join(
		# 	i for i in self.definition_b[index]) + self.data[index][0] + '.' + ''.join(
		# 	i for i in self.definition_a[index])   # get ba description
		# text_des_t_a = text_des_t_ab
		# text_des_t_ba = text_des_t_ab
		# text_des_t_b = text_des_t_ab

		# text_sample_a = self.txt_processor["eval"](text_a_t)
		# text_sample_b = self.txt_processor["eval"](text_b_t)
		# text_sample_ab = self.txt_processor["eval"](text_ab_t)
		# text_sample_ba = self.txt_processor["eval"](text_ba_t)
		# text_sample_des_ab = self.txt_processor["eval"](text_des_t_ab)
		# text_sample_des_a = self.txt_processor["eval"](text_des_t_a)
		# text_sample_des_b = self.txt_processor["eval"](text_des_t_b)
		# text_sample_des_ba = self.txt_processor["eval"](text_des_t_ba)

		image_list = self.data[index][2:]
		label = self.label[index][0]
		# label = self.data[index][2]

		images = []
		image_id = []
		img_file_type = []

		for i in image_list:
			image = PIL.Image.open(self.img_dir + i).convert('RGB')
			images.append(image)
			image_id.append(int(i.split('.')[1]))
			img_file_type.append(i.split('.')[2])

		image_id = torch.Tensor(image_id)
		label_id = torch.Tensor([int(label.split('.')[1])])

		name_1 = self.data[index][0]
		name_2 = self.data[index][1]

		return text_sample_a, text_sample_ab,text_sample_prompt_ab, text_sample_wn_c,\
		       text_sample_wn_t, text_sample_prompt_wn_t, text_sample_gpt_ab, text_sample_prompt_gpt_ab,\
		       images, image_id, label_id, name_1, name_2, img_file_type


def collate_fn_eval(batch):

	all_image_list, all_label = [], []
	for i, (text_sample_a, text_sample_ab,text_sample_prompt_ab, text_sample_wn_c,\
		       text_sample_wn_t, text_sample_prompt_wn_t, text_sample_gpt_ab, text_sample_prompt_gpt_ab, images, image_id, label_id, name_1, name_2, img_file_type) in enumerate(batch):
		all_image_list.append(image_id[None])
		all_label.append(label_id[None])

	all_image_list = torch.cat(all_image_list)
	all_label = torch.cat(all_label)

	out = (text_sample_a, text_sample_ab,text_sample_prompt_ab, text_sample_wn_c,\
		       text_sample_wn_t, text_sample_prompt_wn_t, text_sample_gpt_ab, text_sample_prompt_gpt_ab,
	       images, all_image_list, all_label, name_1, name_2, img_file_type)

	return out
