from vist_eval.album_eval import AlbumEvaluator

import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import collections
from pathlib import Path
from packaging import version

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import logging
import shutil
from pprint import pprint


from param import parse_args

from vist_style_data import get_loader
from utils import load_state_dict, LossMeter, set_global_logging_level
import dist_utils
#import wandb
import json
from pprint import pformat

set_global_logging_level(logging.ERROR, ["transformers"])

proj_dir = Path(__file__).resolve().parent.parent


_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

from trainer_base import TrainerBase

class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)

        split = ['val', 'test']
        self.ref_dir = "data/reference/"
        story_line = json.load(open(os.path.join(train_loader.folder_path, 'anno/story_line.json'),'r'))
        try:
            os.makedirs(self.ref_dir)
        except:
            pass
        for split in ['val', 'test']:
            if os.path.exists(os.path.join(self.ref_dir, '{}_reference.json'.format(split))):
                continue
            reference = {}
            for story in story_line[split].values():
                if story['album_id'] not in reference:
                    reference[story['album_id']] = [story['origin_text']]
                else:
                    reference[story['album_id']].append(story['origin_text'])
            with open(os.path.join(self.ref_dir, '{}_reference.json'.format(split)), 'w') as f:
                json.dump(reference, f)


        #self.wandb_initialized = False

        from vist_style_model import T5Story

        model_kwargs = {}
        if 't5' in args.backbone:
            model_class = T5Story
        elif 'bart' in args.backbone:
            raise NotImplemented

        config = self.create_config()
        self.tokenizer = self.create_tokenizer()
        if 'bart' in self.args.tokenizer:
            num_added_toks = 0
            if config.use_vis_order_embedding:
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                        [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
                special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
                num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

                config.default_obj_order_ids = self.tokenizer.convert_tokens_to_ids([f'<vis_extra_id_{i}>' for i in range(100)])

        self.model = self.create_model(model_class, config, **model_kwargs)

        if 't5' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        elif 'bart' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.model.model.shared.num_embeddings + num_added_toks)

        self.model.tokenizer = self.tokenizer

        # Load Checkpoint
        self.start_epoch = None
        if args.load!='':
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)

        if self.args.from_scratch:
            self.init_weights()

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            elif _use_apex:
                self.model, self.optim = amp.initialize(
                    self.model, self.optim, opt_level='O1', verbosity=self.verbose)

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True
                                 )
        if self.verbose:
            print(f'It took {time() - start:.1f}s')

    def train(self):
        if self.verbose:
            loss_meter = LossMeter()
            best_valid = 0.
            best_epoch = 0

            #if not self.wandb_initialized:

            #    if 't5' in self.args.backbone:
            #        project_name = "VLT5_VISTCaption"
            #    elif 'bart' in self.args.backbone:
            #        project_name = "VLBart_VISTCaption"

            #    wandb.init(project=project_name)
            #    wandb.run.name = self.args.run_name
            #    wandb.config.update(self.args)
            #    wandb.watch(self.model)

            #    src_dir = Path(__file__).resolve().parent
            #    base_path = str(src_dir.parent)
            #    src_dir = str(src_dir)
            #    wandb.save(os.path.join(src_dir + "/*.py"), base_path=base_path)

            #    self.wandb_initialized = True

        if self.args.distributed:
            dist.barrier()

        global_step = 0
        epochs = self.args.epochs

        for epoch in range(epochs):

            if self.start_epoch is not None:
                epoch += self.start_epoch
            self.model.train()
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)
            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=120)

            epoch_results = {
                'loss': 0.,

            }

            for step_i, batch in enumerate(self.train_loader):

                if self.args.fp16 and _use_native_amp:
                    with autocast():
                        if self.args.distributed:
                            results = self.model.module.train_step(batch)
                        else:
                            results = self.model.train_step(batch)
                else:
                    if self.args.distributed:
                        results = self.model.module.train_step(batch)
                    else:
                        results = self.model.train_step(batch)

                loss = results['loss']

                if self.args.fp16 and _use_native_amp:
                    self.scaler.scale(loss).backward()
                elif self.args.fp16 and _use_apex:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()


                loss = loss.detach()

                # Update Parameters
                if self.args.clip_grad_norm > 0:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(
                            self.optim), self.args.clip_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)

                update = True
                if self.args.gradient_accumulation_steps > 1:
                    if step_i == 0:
                        update = False
                    elif step_i % self.args.gradient_accumulation_steps == 0 or step_i == len(self.train_loader) - 1:
                        update = True
                    else:
                        update = False

                if update:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optim)
                        self.scaler.update()
                    else:
                        self.optim.step()

                    if self.lr_scheduler:
                        self.lr_scheduler.step()
                    # self.model.zero_grad()
                    for param in self.model.parameters():
                        param.grad = None
                    global_step += 1

                for k, v in results.items():
                    if k in epoch_results:
                        epoch_results[k] += v.item()

                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    try:
                        lr = self.optim.get_lr()[0]
                    except AttributeError:
                        lr = self.args.lr

                if self.verbose:
                    loss_meter.update(loss.item())
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} | Steps {global_step}'
                    desc_str += f' | Loss {loss_meter.val:4f}'
                    pbar.set_description(desc_str)
                    pbar.update(1)

            if self.args.distributed:
                dist.barrier()

            if self.verbose:
                pbar.close()

                # format ex)
                # {'Bleu_1': 0.9999999997500004,
                #  'Bleu_2': 0.5773502690332603,
                #  'Bleu_3': 4.3679023223468616e-06,
                #  'Bleu_4': 1.4287202142987477e-08,
                #  'CIDEr': 3.333333333333333,
                #  'METEOR': 0.43354749322305886,
                #  'ROUGE_L': 0.75,
                #  'SPICE': 0.6666666666666666}

            # Validation
            if epoch%10 == 0:
                valid_results = self.evaluate(self.val_loader, mode='val')

                if self.verbose:
                    valid_score = valid_results['CIDEr']

                    if valid_score > best_valid or epoch == 0:
                        best_valid = valid_score
                        best_epoch = epoch
                        self.save("BEST")

                    log_str = ''

                    log_str += pformat(valid_results)
                    log_str += "\nEpoch %d: Valid CIDEr %0.4f" % (epoch, valid_score)
                    log_str += "\nEpoch %d: Best CIDEr %0.4f\n" % (best_epoch, best_valid)

                    #wandb_log_dict = {}
                    #wandb_log_dict['Train/Loss'] = epoch_results['loss'] / len(self.train_loader)

                    #for score_name, score in valid_results.items():
                    #    wandb_log_dict[f'Valid/{score_name}'] = score

                    #wandb_log_dict[f'Valid/best_epoch'] = best_epoch

                    #wandb.log(wandb_log_dict, step=epoch)

                    print(log_str)

                if self.args.distributed:
                    dist.barrier()

        if self.verbose:
            self.save("LAST")

        # Test Set
        best_path = os.path.join(self.args.output, 'BEST')
        self.load(best_path)

        if self.verbose:
            #wandb.save(best_path, base_path=self.args.output)
            print(f'\nUploaded checkpoint {best_epoch}', best_path)

        test_results = self.evaluate(self.test_loader, mode='test')

        if self.verbose:
            #wandb_log_dict = {}
            #for score_name, score in test_results.items():
            #    wandb_log_dict[f'Test/{score_name}'] = score
            #wandb.log(wandb_log_dict, step=epoch)

            log_str = 'Test set results\n'
            log_str += pformat(test_results)

            print(log_str)

        if self.args.distributed:
            dist.barrier()

    def predict(self, loader, dump_path=None):
        """
        Predict the answers to questions in a data split.
        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        with torch.no_grad():

            predictions = []
            targets = []
            album_ids = []

            gen_kwargs = {}
            gen_kwargs['num_beams'] = self.args.num_beams
            gen_kwargs['max_length'] = self.args.gen_max_length

            for i, batch in enumerate(tqdm(loader, ncols=120, desc="Prediction", disable=not self.verbose)):

                if self.args.distributed:
                    results = self.model.module.test_step(
                        batch,
                        **gen_kwargs)
                else:
                    results = self.model.test_step(
                        batch,
                        **gen_kwargs)

                predictions.extend(results['pred'])
                album_ids.extend(results['album_id'])


                if 'targets' in batch:
                    targets.extend(batch['targets'])

            results = {
                'album_ids': album_ids,
                'predictions': predictions,
                'targets': targets
            }

            if self.args.distributed:
                dist.barrier()

                dist_results = dist_utils.all_gather(results)
                predictions = []
                album_ids = []
                targets = []
                for result in dist_results:
                    predictions.extend(result['predictions'])
                    album_ids.extend(result['album_ids'])
                    targets.extend(result['targets'])
                results = {
                    'album_ids': album_ids,
                    'predictions': predictions,
                    'targets': targets
                }

            return results

    def evaluate(self, loader, mode='val', dump_path=None):
        results = self.predict(loader, dump_path)
        self.reference = json.load(open(os.path.join(self.ref_dir, '{}_reference.json'.format(mode))))
        self.save_dir = os.path.join(self.args.output)
        try:
            os.makedirs(self.save_dir)
        except:
            pass
        json_prediction_file = os.path.join(self.save_dir, 'prediction_{}'.format(mode))
        self.eval = AlbumEvaluator()

        predictions = {}
        for idx in range(len(results['album_ids'])):
            album_id = results['album_ids'][idx]
            if album_id not in predictions:
                predictions[album_id] = [results['predictions'][idx]]
        self.eval.evaluate(self.reference, predictions)
        with open(json_prediction_file, 'w') as f:
            json.dump(predictions, f)
        return self.eval.eval_overall


def main_worker(gpu, args):
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    print(f'Building train loader at GPU {gpu}')
    train_loader = get_loader(
        args,
        split=args.train, mode='train', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=args.num_workers,
        topk=args.train_topk,
    )

    if args.valid_batch_size is not None:
        valid_batch_size = args.valid_batch_size
    else:
        valid_batch_size = args.batch_size
    print(f'Building val loader at GPU {gpu}')
    val_loader = get_loader(
        args,
        split=args.valid, mode='val', batch_size=valid_batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=4,
        topk=args.valid_topk,
    )
    print('# len val loader:', len(val_loader))

    print(f'Building test loader at GPU {gpu}')
    test_loader = get_loader(
        args,
        split=args.test, mode='val', batch_size=valid_batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=4,
        topk=args.valid_topk,
        )

    trainer = Trainer(args, train_loader, val_loader, test_loader, train=True)
    trainer.train()



if __name__ == "__main__":

    cudnn.benchmark = True
    args = parse_args()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    if args.local_rank in [0, -1]:
        print(args)

        comments = []
        if args.load!='':
            ckpt_str = "_".join(args.load.split('/')[-3:])
            comments.append(ckpt_str)
        if args.comment != '':
            comments.append(args.comment)
        comment = '_'.join(comments)

        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M')

        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'

        args.run_name = run_name

    if args.distributed:
        main_worker(args.local_rank, args)
