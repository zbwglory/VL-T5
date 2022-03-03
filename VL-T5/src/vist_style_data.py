from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import random
from multiprocessing import Pool
import h5py
import pickle
import math
from tqdm import tqdm
import torch
import os
import numpy as np
from copy import deepcopy

from torch.utils.data.distributed import DistributedSampler

from transformers import T5TokenizerFast, BartTokenizer
from tokenization import VLT5TokenizerFast

vist_dir = '/home/bwzhang/data/'
vist_img_dir = os.path.join(vist_dir, 'images_448/')
vist_feature_dir = os.path.join(vist_dir, 'features_448')


class VISTStyleTransferDataset(Dataset):
    def __init__(self,
                 split='train',
                 raw_dataset=None,
                 rank=-1,
                 topk=-1,
                 verbose=True,
                 args=None,
                 mode='train'):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode

        # Loading datasets to data
        if self.verbose:
            print('Data source: ', split)


        if self.args.tokenizer is None:
            self.args.tokenizer = self.args.backbone

        if 't5' in self.args.tokenizer:
            if self.args.use_vision:
                self.tokenizer = VLT5TokenizerFast.from_pretrained(
                    args.backbone,
                    # max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)
            else:
                self.tokenizer = T5TokenizerFast.from_pretrained(
                    args.backbone,
                    # max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)
        elif 'bart' in self.args.tokenizer:
            self.tokenizer = BartTokenizer.from_pretrained(
                args.backbone,
                # max_length=self.args.max_text_length,
                do_lower_case=self.args.do_lower_case)

            additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                    [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
            special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)


        data_info_path = os.path.join(vist_dir, 'anno', 'story_line.json')
        with open(data_info_path) as f:
            vist_anno_data = json.load(f)


        n_image_lists = 0

        data = []
        split_data = vist_anno_data[split]
        for data_id in split_data:
            datum = split_data[data_id]
            check_flag = True
            for x in datum['flickr_id']:
                if x not in vist_anno_data['image2caption_original'][split]:
                    check_flag = False
                    break
            if check_flag:
                if split == 'train':
                    new_datum = {
                        'album_id': '---',
                        'img_list_id': datum['flickr_id'],
                        'descriptive_sentence': [vist_anno_data['image2caption_original'][split][x] for x in datum['flickr_id']],
                        'targets': datum['origin_text'].strip(),
                        'sent': datum['origin_text'].strip(),
                        'is_train': True,
                        'split': split,
                    }
                else:
                    new_datum = {
                        'album_id': datum['album_id'],
                        'img_list_id': datum['flickr_id'],
                        'descriptive_sentence': [vist_anno_data['image2caption_original'][split][x] for x in datum['flickr_id']],
                        'targets': datum['origin_text'].strip(),
                        'sent': datum['origin_text'].strip(),
                        'is_train': False,
                        'split': split,
                    }
                data.append(new_datum)

        n_image_lists += 1

        if self.verbose:
            print(f"{split} has {n_image_lists} image lists")
            print(f"Loaded {len(data)} data from", split)

        self.n_gpus = torch.cuda.device_count()

        self.rank = rank
        if self.topk > 0:
            data = data[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")

        self.data = data

        if self.verbose:
            print("# all stories:", len(self.data))

        self.source_to_h5_path = {}
        self.source_to_h5 = {}

        if self.args.max_n_boxes == 36:
            self.source_to_h5_path.update({
                'train': os.path.join(vist_feature_dir, 'train_boxes36.h5'),
                'val': os.path.join(vist_feature_dir, 'val_boxes36.h5'),
                'test': os.path.join(vist_feature_dir, 'test_boxes36.h5'),
            })


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]

        out_dict['album_id'] = datum['album_id']

        ###### Image ######
        if self.args.use_vision:
            out_dict['img_list_id'] = datum['img_list_id']

            # bwzhang@: This is to fulfill the collate function
            out_dict['img_id'] = ' '.join(datum['img_list_id'])  

            if datum['split'] not in self.source_to_h5:
                f = self.source_to_h5_path[datum['split']]
                f = h5py.File(f, 'r')
                self.source_to_h5[datum['split']] = f
            f = self.source_to_h5[datum['split']]

            out_dict['n_boxes'] = []
            out_dict['boxes'] = []
            out_dict['vis_feats'] = []
            for img_id in datum['img_list_id']:
                # Normalize the boxes (to 0 ~ 1)
                img_h = f[f'{img_id}/img_h'][()]
                img_w = f[f'{img_id}/img_w'][()]
                boxes = f[f'{img_id}/boxes'][()]  # (x1, y1, x2, y2)
                boxes[:, (0, 2)] /= img_w
                boxes[:, (1, 3)] /= img_h
                np.testing.assert_array_less(boxes, 1+1e-5)
                # np.testing.assert_array_less(boxes, 1+5e-2)
                np.testing.assert_array_less(-boxes, 0+1e-5)
                boxes = torch.from_numpy(boxes)

                boxes.clamp_(min=0.0, max=1.0)

                n_boxes = len(boxes)

                feats = np.zeros(shape=(n_boxes, 2048), dtype=np.float32)
                f[f'{img_id}/features'].read_direct(feats)
                feats = torch.from_numpy(feats)

                n_boxes = min(n_boxes, self.args.max_n_boxes)
                # if not self.args.BUTD100:
                boxes = boxes[:n_boxes]
                feats = feats[:n_boxes]
                out_dict['n_boxes'].append(n_boxes)
                out_dict['boxes'].append(boxes)
                out_dict['vis_feats'].append(feats)
            out_dict['n_boxes'] = np.sum(out_dict['n_boxes'])
            out_dict['boxes'] = torch.cat(out_dict['boxes']) 
            out_dict['vis_feats'] = torch.cat(out_dict['vis_feats'])
            

        ###### Text #####
        prefix = "@ The sentences are @"

        descriptive_sentences = [np.random.choice(x, 1)[0].split('|')[0] for x in datum['descriptive_sentence']]

        input_tokens = [prefix]

        input_tokens.extend(descriptive_sentences)

        input_tokens.append('# The story is #')

        input_text = ' '.join(input_tokens)

        if 't5' in self.args.tokenizer:
            input_ids = self.tokenizer.encode(
                input_text,
                max_length=self.args.max_text_length, truncation=True)
        elif 'bart' in self.args.tokenizer:
            input_ids = self.tokenizer.encode(
                input_text,
                max_length=self.args.max_text_length, truncation=True)
        else:
            input_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(input_text)[:self.args.max_text_length - 1] + ['[SEP]'])

        out_dict['input_text'] = input_text

        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        if datum['is_train']:
            target = datum['sent'].strip()
            if 't5' in self.args.tokenizer:
                target_ids = self.tokenizer.encode(target, max_length=self.args.gen_max_length, truncation=True)
            elif 'bart' in self.args.tokenizer:
                target_ids = self.tokenizer.encode(target, max_length=self.args.gen_max_length, truncation=True)

            assert len(target_ids) <= self.args.gen_max_length, len(target_ids)
            out_dict['sent'] = target 
            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)

        out_dict['targets'] = datum['targets']
        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        S_W_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        input_masks = torch.zeros(B, S_W_L, dtype=torch.long)

        if self.args.no_prefix:
            assert input_ids.size() == (B, 0)

        if self.args.use_vision:
            V_L = max(entry['n_boxes'] for entry in batch)
            # V_L = len(batch[0]['boxes'])
            feat_dim = batch[0]['vis_feats'].shape[-1]

            boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
            vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)
            vis_attention_mask = torch.zeros(B, V_L, dtype=torch.float)

        if 'target_ids' in batch[0]:
            T_W_L = max(entry['target_length'] for entry in batch)
            target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        # sentences = []

        targets = []
        img_ids = []
        img_paths = []
        input_text = []
        album_id = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            input_masks[i, :entry['input_length']] = 1.0

            if self.args.use_vision:
                n_boxes = entry['n_boxes']
                boxes[i, :n_boxes] = entry['boxes']
                vis_feats[i, :n_boxes] = entry['vis_feats']
                vis_attention_mask[i, :n_boxes] = 1
                img_ids.append(entry['img_id'])
                # img_paths.append(entry['img_path'])

            if 'target_ids' in entry:
                target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'input_text' in entry:
                input_text.append(entry['input_text'])

            if 'album_id' in entry:
                album_id.append(entry['album_id'])

            if 'targets' in entry:
                targets.append(entry['targets'])


        batch_entry['input_ids'] = input_ids
        batch_entry['attention_mask'] = input_masks
        if 'target_ids' in batch[0]:
            word_mask = target_ids != self.tokenizer.pad_token_id
            target_ids[~word_mask] = -100
            batch_entry['target_ids'] = target_ids

        if self.args.use_vision:
            batch_entry['boxes'] = boxes
            batch_entry['vis_feats'] = vis_feats
            batch_entry['vis_attention_mask'] = vis_attention_mask
            batch_entry['img_id'] = img_ids
            batch_entry['img_paths'] = img_paths

        batch_entry['input_text'] = input_text
        batch_entry['album_id'] = album_id

        batch_entry['targets'] = targets

        batch_entry['task'] = 'vist'

        return batch_entry


def get_loader(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1):

    verbose = (gpu == 0)

    dataset = VISTStyleTransferDataset(
        split,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        mode=mode)

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)

    loader.task = 'vist'
    loader.folder_path = vist_dir

    return loader


# Unit-test code for the vist dataloader
if __name__ == "__main__":
    class Object(object):
        pass
    args = Object()
    args.tokenizer = 't5-base'
    args.use_vision = False
    args.backbone = 't5-base'
    args.do_lower_case = True
    args.max_n_boxes = 36
    args.max_text_length = 200
    args.gen_max_length = 200
    data_loader = VISTStyleTransferDataset(args=args)
    data_loader.__getitem__(10)
