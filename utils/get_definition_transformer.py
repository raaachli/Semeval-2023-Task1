import os
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torch.nn import functional as F
import numpy as np
import h5py
import PIL
import csv
import clip
from nltk.corpus import wordnet as wn
import pytorch_lightning as pl

from transformers import BertModel, BertTokenizer, TransfoXLTokenizer, TransfoXLModel
import re
from googletrans import Translator

def clean_punctuation(text, pattern=re.compile(r"[…!?,;.:\-*()\[\]{}<>\"]+")):
	text = re.sub(r'<.*?>', '', text)
	text = re.sub(pattern, " ", text)
	text = re.sub("\s+", " ", text)
	return text


def get_definition(data_dir, model, tokenizer, data='train'):
	data_dir = data_dir
	text_file = data_dir + '/test.data.v1.1.gold/'+'en.test.data.v1.1.txt'
	label_file = data_dir + '/test.data.v1.1.gold/'+'en.test.gold.v1.1.txt'
	cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
	save_file = data_dir + '/test.data.v1.1.wordnet/' + 'en.test.wn.v1.2.txt'
	translator = Translator()
	with open(text_file) as f:
		reader = csv.reader(f, delimiter="\t")
		data = list(reader)

	with open(label_file) as f:
		reader = csv.reader(f, delimiter="\t")
		label = list(reader)

	count = 0

	for index in range(len(data)):
		print(index)
		text_1_t = data[index][0].lower()
		text_2_t = data[index][1].lower()
		# text_1_t = translator.translate(text_1_t).text
		# text_2_t = translator.translate(text_2_t).text
		# text_2_t = "This is {}, a {}".format(text_1_t, text_2_t)
		# print(text_2_t)

		text_1_def_t = [clean_punctuation(i.definition().lower()) for i in wn.synsets(text_1_t)]

		text_features_1_def = []
		for i in range(len(text_1_def_t)):
			def_i = tokenizer(text_1_def_t[i], return_tensors='pt')
			outputs = model(**def_i, return_dict=True)
			logits = outputs["last_hidden_state"]
			pooled_logits = logits[:, -1, :]
			# sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
			text_features_1_def.append(pooled_logits.squeeze())

		text_2 = tokenizer(text_2_t, return_tensors='pt')

		text_features_2 = model(**text_2, return_dict=True)
		text_features_2 = text_features_2["last_hidden_state"]
		text_features_2 = text_features_2[:, -1, :]

		text_features_2_rep = text_features_2.repeat(len(text_1_def_t), 1)
		# text_features_1_def = [model.encode_text(i) for i in text_1_def]

		if len(text_features_1_def) > 0:
			text_features_1_def = torch.stack(text_features_1_def)
			# text_features_1_def = text_features_1_def.squeeze()

			cos_sim = cos(text_features_2_rep, text_features_1_def)

			ce_sim = []
			for i in range(len(text_1_def_t)):
				sim = F.cross_entropy(text_features_2_rep[i], text_features_1_def[i])
				ce_sim.append(-sim.item())

			ce_sim = np.array(ce_sim)
			cos_sim = cos_sim.detach().numpy()

			cos_def_idx = np.argsort(cos_sim)[::-1]
			ce_def_idx = np.argsort(ce_sim)[::-1]
			def_idx = cos_def_idx
			def_rank = [text_1_def_t[i] for i in def_idx]
			# print(def_idx)
			text_3_t = def_rank[0]

			# text_final_t = "This is a picture of {}, {}.".format(text_2_t, text_3_t)
			text_final_t = text_3_t
		else:
			count += 1
			# text_final_t = "This is a picture of {}.".format(text_2_t)
			text_final_t = text_2_t

		with open(save_file, 'a') as f:
			f.write(text_final_t + '\n')

	print(count)


if __name__ == '__main__':
	device = 'cpu'
	data_dir = '/media/kiki/971339f7-b775-448b-b7d8-f17bc1499e4d/Dataset/semeval-2023-test/'
	pl.seed_everything(0)
	torch.no_grad().__enter__()
	# max_position_embeddings=128
	# configuration = CLIPTextConfig(max_position_embeddings=max_position_embeddings)
	# model = CLIPTextModel(configuration)
	# configuration = model.config
	# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

	# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
	# model = BertModel.from_pretrained("bert-base-uncased")

	tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")
	model = TransfoXLModel.from_pretrained("transfo-xl-wt103")

	get_definition(data_dir, model, tokenizer, data='trial')
