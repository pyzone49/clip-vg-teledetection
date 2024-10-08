import os
import time
import math
import json
import random
import argparse
import datetime,traceback
import numpy as np
from pathlib import Path
import os.path as osp
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler
from datasets.transforms import denormalize_bboxes
from datasets.transforms import Compose, ToTensor
import utils.misc as utils
from models import build_model
from datasets import build_dataset
from engine import evaluate
import datasets.transforms as T
from torchvision.transforms import ToPILImage
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as F

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'  # batch_size max=768，0、1在用


def get_args_parser():
    parser = argparse.ArgumentParser('CLIP-VG Args', add_help=False)
    parser.add_argument('--sup_type', default='un', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_bert', default=1e-5, type=float)
    parser.add_argument('--lr_visu_cnn', default=1e-5, type=float)
    parser.add_argument('--lr_visu_tra', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--lr_power', default=0.9, type=float, help='lr poly power')
    parser.add_argument('--lr_exponential', default=0.9, type=float, help='lr exponential')
    parser.add_argument('--clip_max_norm', default=0., type=float, help='gradient clipping max norm')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='if evaluation only')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr_scheduler', default='step', type=str)
    parser.add_argument('--lr_drop', default=60, type=int)
    # Augmentation options
    parser.add_argument('--aug_blur', action='store_true', help="If true, use gaussian blur augmentation")
    parser.add_argument('--aug_crop', action='store_true', help="If true, use random crop augmentation")
    parser.add_argument('--aug_scale', action='store_true', help="If true, use multi-scale augmentation")
    parser.add_argument('--aug_translate', action='store_true', help="If true, use random translate augmentation")
    # only support ViT-B/16 and ViT-L/14
    parser.add_argument('--model', type=str, default='ViT-B/16', help="Name of model to be exploited.")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int, help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--imsize', default=224, type=int, help='image size')
    """ embedding size"""
    parser.add_argument('--emb_size', default=512, type=int, help='fusion module embedding dimensions')
    # Vision-Language Transformer
    parser.add_argument('--vl_dropout', default=0.1, type=float,
                        help="Dropout applied in the vision-language transformer")
    parser.add_argument('--vl_nheads', default=8, type=int,
                        help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument('--vl_hidden_dim', default=512, type=int,
                        help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_enc_layers', default=6, type=int,
                        help='Number of encoders in the vision-language transformer')

    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='./data/image_data/', help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='./data/pseudo_samples/',  help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='referit', type=str, help='referit/unc/unc+/gref/gref_umd')
    parser.add_argument('--max_query_len', default=77, type=int, help='maximum time steps (lang length) per batch')
    # Prompt Engineering: "{pseudo_query}" denote without using prompt
    #                    "{pseudo_query}" or using "find the region that corresponds to the description {pseudo_query}"
    parser.add_argument('--prompt', type=str, default='{pseudo_query}', help="Prompt template")
    # dataset parameters
    parser.add_argument('--output_dir', default='./outputs', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--retrain', default='', help='retrain from checkpoint')
    parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    # evalutaion options
    parser.add_argument('--eval_set', default='test', type=str)  # 'testA', 'testB', 'val'
    parser.add_argument('--eval_model', default='', type=str)

    return parser


def main(args):
    """ distribution init """
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    if (args.model == "ViT-L/14" or args.model == "ViT-L/14@336px"):
        args.vl_hidden_dim = 768
    device = torch.device(args.device)

    # # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print('### INFO ### torch.backends.cudnn.benchmark = {}'.format(torch.backends.cudnn.benchmark))

    # build model
    model = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of requires_grad params: ', n_parameters_grad)
    print('number of all params: ', n_parameters)

    # build dataset
    dataset_test = build_dataset(args.eval_set, args)
    dataset_test.images = dataset_test.images[500:550]
    if args.distributed:
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_test = torch.utils.data.BatchSampler(sampler_test, args.batch_size, drop_last=False)
    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                  drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    checkpoint = torch.load(args.eval_model, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])
    print("Current model training epoch is: ", checkpoint['epoch'])
    #print first example from data_loader_test
    item = next(iter(data_loader_test))
    # output log
    eval_model = args.eval_model
    eval_model = eval_model.split('/')[-1].split('.')[0]
    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "eval_{}_{}_{}_log.txt".format(args.dataset, args.eval_set, eval_model)).open("a") as f:
            f.write(str(args) + "\n")
            f.flush()
    start_time = time.time()
    real_boxes = []
    images = []
    images_prompts = []
    # real_img_box = []
    #dataloader to list of images
    for i, batch in enumerate((dataset_test)):
        # print("batch",batch,file=open("outputs/batch.txt","w"))
        img, img_mask, text ,text_mask, bbox, img_file, phrase, bbox_ori = dataset_test.__getitem__(i)
        print(img_file, bbox_ori, phrase)
        images.append(img)
        #from np float32 to list 
        real_boxes.append( bbox.tolist())
        images_prompts.append(phrase)
        # real_img_box.append(bbox_ori)
    accuracy,gt_boxes,pred_box_list = evaluate(args, model, data_loader_test, device)
    show_image(images,real_boxes,images_prompts,pred_box_list)
    if utils.is_main_process():
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Testing time {}'.format(total_time_str))
        log_stats = {'test_model:': args.eval_model,
                     '%s_set_accuracy'%args.eval_set: accuracy,
                     }
        print(log_stats)
        if args.output_dir and utils.is_main_process():
            with (output_dir / "eval_{}_{}_{}_log.txt".format(args.dataset, args.eval_set, eval_model)).open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

def show_image(images,gt_boxes,images_prompts,pred_boxes):
    
    to_pil = ToPILImage()
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    # Ensure mean and std are reshaped to match the image tensor's dimensions: [C, 1, 1] for broadcasting
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    for i, image_path in enumerate(gt_boxes): 
        try:
            img = images[i]
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
            img = to_pil(img)
            w_real,h_real = img.size
            box = gt_boxes[i]
            x1,y1,x2,y2 = from_box_to_xyxy(box,w_real,h_real)
            x,y,x0,y0 = from_box_to_xyxy(pred_boxes[i],w_real,h_real)
            #draw rectangle
            draw = ImageDraw.Draw(img)
            draw.rectangle([x1,y1,x2,y2], outline='green',width=3)
            draw.rectangle([x,y,x0,y0], outline='blue',width=3)
            #add text using prompt above the rectangle
            draw.text((0, 0), images_prompts[i], fill='white')
            img.show()
        except:
            traceback.print_exc()
            pass
def from_box_to_xyxy(box,width,height):
    # try:
    print("Box: ",box)
    #from tensor to list
    # box = box.tolist()
    x,y,w,h = box
    x_center = int(x* width)
    y_center = int(y* height)
    w = int(w* width)
    h = int(h* height)
    x = x_center - w//2
    y = y_center - h//2
    x1,y1,x2,y2 = x,y,w,h
    return x1,y1,x2,y2
    # except:
    #     return 0,0,0,0
if __name__ == '__main__':
    parser = argparse.ArgumentParser('CLIP-VG evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
