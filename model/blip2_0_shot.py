import csv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
from utils.dataset_albef import SemevalDataset, collate_fn_eval
from lavis.models import load_model_and_preprocess
import pytorch_lightning as pl
import os
import statistics
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
language = 'en'
model_name = 'blip2'
save_file_name = '../result_add/'+model_name+'/'
os.makedirs(save_file_name, exist_ok=True)


def albef_zero_shot_eval():
    np.random.seed(0)
    device = 'cuda'
    dt_collate_fn = collate_fn_eval
    model, vis_processor, txt_processor = load_model_and_preprocess(name=model_name,
                                                                     model_type="coco", is_eval=True,
                                                                      device=device)

    languages = ['en', 'fa', 'it']
    mrr_list = []
    hit_list = []
    measure = []
    for language in languages:
        mrr = [[] for _ in range(1)]
        hit = [[] for _ in range(1)]
        dataset = SemevalDataset('/media/kiki/971339f7-b775-448b-b7d8-f17bc1499e4d/Dataset/semeval-2023-test/',
                                 model,
                                 vis_processor,
                                 txt_processor,
                                 device,
                                 language
                                 )

        # num_data = 12869
        # indices = list(range(num_data))
        # # np.random.shuffle(indices)
        # test_indice = indices[:800]
        # sampler = SubsetRandomSampler(test_indice)

        eval_loader_kwargs = {
            'batch_size': 1,
            'num_workers': 1,
            'shuffle': False,
            'drop_last': False,
            # 'sampler': sampler,
            'collate_fn': dt_collate_fn
        }

        data_loader = DataLoader(dataset, **eval_loader_kwargs)
        with torch.no_grad():
            with tqdm(data_loader, dynamic_ncols=True) as tqdmDataLoader:
                for batch in tqdmDataLoader:
                    text_sample_a, text_sample_ab, text_sample_prompt_ab, text_sample_wn_c, \
                    text_sample_wn_t, text_sample_prompt_wn_t, text_sample_gpt_ab, text_sample_prompt_gpt_ab, \
                    raw_images, image_list, label, name_1, name_2, img_file_type = batch

                    image_list = image_list.squeeze().cpu().numpy()
                    label = label.squeeze().cpu()

                    image_features = []
                    for image in raw_images:
                        i = vis_processor["eval"](image).unsqueeze(0).to(device)
                        sample_a = {"image": i, "text_input": [text_sample_a]}
                        sample_ab = {"image": i, "text_input": [text_sample_ab]}
                        sample_prompt_ab = {"image": i, "text_input": [text_sample_prompt_ab]}
                        sample_wn_c = {"image": i, "text_input": [text_sample_wn_c]}
                        sample_wn_t = {"image": i, "text_input": [text_sample_wn_t]}
                        sample_wn_prompt_t = {"image": i, "text_input": [text_sample_prompt_wn_t]}

                        sample_gpt_ab = {"image": i, "text_input": [text_sample_gpt_ab]}
                        sample_prompt_gpt_ab = {"image": i, "text_input": [text_sample_prompt_gpt_ab]}

                        image_feature = model.forward_image(i)[0]
                        image_features.append(image_feature)

                        # image_feature = model.extract_features(sample_a, mode='image')
                        # image_features.append(image_feature.image_embeds_proj)

                    image_features = torch.cat(image_features)
                    image_feature = image_features.squeeze()

                    # text_samples = [sample_a, sample_ab, sample_prompt_ab, sample_wn_c, sample_wn_t, sample_gpt_ab, sample_prompt_gpt_ab]
                    text_samples = [sample_wn_prompt_t]
                    test_method = 0
                    for sample in text_samples:
                        rank_val_a, rank_label_a, image_idx_a, probs_a, type_a = get_measure(sample['text_input'][0], model, image_feature,image_list,img_file_type,label)
                        mrr[test_method].append(rank_val_a)
                        hit[test_method].append(rank_label_a)
                        save_result(name_1, name_2, save_file_name + model_name + '_' + language + '_' + str(test_method), image_idx_a, type_a,
                                    label)
                        test_method += 1

        measure_l = []
        for i in range(1):
            measure_i = save_mrr_hit(mrr[i], hit[i], save_file_name + model_name + '_' +language+ '_' + str(i))
            measure_l.append(measure_i)
        measure.append(measure_l)

    for i in range(1):
        m_i = []
        h_i = []
        for l in range(3):
            m_i.append(measure[l][i][0])
            h_i.append(measure[l][i][1])
        ave_mrr = sum(m_i)/len(m_i)
        ave_hit = sum(h_i)/len(h_i)

        with open('../result/' + save_file_name + 'all_'+str(i)+'_measure.txt', 'a') as f:
            f.write('\t'.join(['mrr: ' + str(ave_mrr), 'hit: ' + str(ave_hit)]) + '\n')


def get_measure(text_feature, model, image_feature,image_list,img_file_type,label):
    # text_feature_a = model.extract_features(text_feature, mode='text')
    # text_feature_a = text_feature_a.text_embeds_proj

    text_feature_a = model.tokenizer(text_feature,
                                      padding="max_length",
                                      truncation=True,
                                      max_length=50,
                                      return_tensors="pt").to(model.device)
    text_feature_a = model.forward_text(text_feature_a)

    rank_val_a, rank_label_a, image_idx_a, probs_a, type_a = get_similarity(text_feature_a,
                                                                            image_feature,
                                                                            image_list,
                                                                            img_file_type,
                                                                            label)
    return rank_val_a, rank_label_a, image_idx_a, probs_a, type_a


def get_caption_similarity(captions, text_feature, probs, label, image_list):
    rank_idx = [sorted(probs).index(x) for x in probs]
    image_idx = np.argsort(rank_idx)[::-1]
    # top_n_image = image_idx[:5]
    top_n_image = image_idx

    top_n_captions = [captions[i] for i in top_n_image]

    cos_sim = []
    for cap_feature in top_n_captions:
        cos_i = get_cos_sim(cap_feature[:,0,:], text_feature[:,0,:])
        cos_sim.append(cos_i.item())

    cos_sim_raw = []
    for cap_feature in captions:
        cos_i = get_cos_sim(cap_feature[:,0,:], text_feature[:,0,:])
        cos_sim_raw.append(cos_i.item())
    print('cos')
    print(cos_sim_raw)
    print([sorted(cos_sim_raw, reverse=True).index(x) for x in cos_sim_raw])

    cos_rank_idx = [sorted(cos_sim).index(x) for x in cos_sim]
    cos_rank_image_idx = np.argsort(cos_rank_idx)[::-1]
    cos_rank_image_idx = [image_idx[i] for i in cos_rank_image_idx]

    out_image_idx = cos_rank_image_idx # + image_idx.tolist()[5:]

    out_image_idx = [image_list[i] for i in out_image_idx]
    rank_label = out_image_idx.index(label.item())+1
    rank_val = 1/rank_label
    return rank_val, rank_label, out_image_idx


def get_cos_sim(feature_1, feature_2):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_sim = cos(feature_1, feature_2)
    return cos_sim


def get_ranking_var(image_idx_1, image_idx_2, image_idx_3, label):
    result = []
    waitlist = []

    for it in range(3):
        prob_i = image_idx_3[it]
        if prob_i not in image_idx_1[:3] or prob_i not in image_idx_2[:3]:
            waitlist.append(prob_i)
        else:
            result.append(prob_i)

    if len(result) == 0:
        first = waitlist[0]
    else:
        first = result[0]

    image_idx_3.remove(first)
    image_idx_3.insert(0, first)

    rank_label = image_idx_3.index(label.item())+1
    rank_val = 1/rank_label
    return rank_val, rank_label, image_idx_3


def get_var_rank(prob_1, prob_2, prob_3, top_n, label, image_list):
    rank_idx_1 = [sorted(prob_1).index(x) for x in prob_1]
    rank_idx_2 = [sorted(prob_2).index(x) for x in prob_2]
    rank_idx_3 = [sorted(prob_3).index(x) for x in prob_3]
    image_idx = np.argsort(rank_idx_3)[::-1]

    var_list = []
    for i in range(len(prob_1)):
        var_i = statistics.variance([rank_idx_1[i], rank_idx_2[i], rank_idx_3[i]])
        var_list.append(var_i)

    var_list_ranked = [var_list[i] for i in image_idx]

    var_list_top_n = var_list_ranked[:top_n]
    image_idx_top_n = image_idx[:top_n]

    var_idx_sorted = np.argsort(var_list_top_n)
    image_idx_top_n_sorted = [image_idx_top_n[i] for i in var_idx_sorted]

    out_image_idx = image_idx_top_n_sorted + image_idx.tolist()[top_n:]

    out_image_idx = [image_list[i] for i in out_image_idx]
    rank_label = out_image_idx.index(label.item())+1
    rank_val = 1/rank_label

    return rank_val, rank_label, out_image_idx


def get_similarity(text_feature, image_feature, image_list, file_type_list, label):
    probs = []
    # print(image_feature.shape)
    # print(text_feature.shape)
    for ind in range(10):
        logits_per_image = (image_feature[ind, 0, :].unsqueeze(0) @ text_feature.t())
        probs.append(logits_per_image[0].squeeze().cpu().numpy())
    probs = np.array(probs)
    image_idx = np.argsort(probs)[::-1]
    image_idx_save = [image_list[i] for i in image_idx]
    file_type = [file_type_list[i] for i in image_idx]

    rank_label = image_idx_save.index(label.item()) + 1
    rank_val = 1 / rank_label
    return rank_val, rank_label, image_idx_save, probs, file_type


def save_mrr_hit(mrr, hit, save_name):
    print(sum(mrr)/len(mrr))
    print(hit.count(1)/len(hit))
    measure = [sum(mrr)/len(mrr), hit.count(1)/len(hit)]
    with open('../result/'+save_name+'_measure.txt', 'a') as f:
        f.write('\t'.join(['mrr: '+str(measure[0]), 'hit: '+str(measure[1])]) + '\n')
    return measure


def save_result(name_1, name_2, save_file_name, image_idx, type_idx, label):
    # with open('../result/'+save_file_name+'_compare.csv', 'a') as f:
    #     write = csv.writer(f)
    #     write.writerow(['image.' + str(int(label)) + "." + , name_1, name_2])
    #     write.writerow(['image.' + str(int(r)) + ".jpg" for r in image_idx])
    with open('../result/'+save_file_name+'_result.txt', 'a') as f:
        files = []
        for i in range(len(image_idx)):
            file_name = 'image.' + str(int(image_idx[i])) + "." + type_idx[i]
            files.append(file_name)
        f.write('\t'.join(files) + '\n')


if __name__ == '__main__':
    pl.seed_everything(0)
    torch.no_grad().__enter__()
    albef_zero_shot_eval()
    #     mrr_list.append(measure[0])
    #     hit_list.append(measure[1])
    #
    #
    #
    # with open('../result/' + save_file_name + '_all_measure.txt', 'a') as f:
    #     f.write('\t'.join(['mrr: ' + str(ave_mrr), 'hit: ' + str(ave_hit)]) + '\n')