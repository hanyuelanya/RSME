import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn.modules.loss import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
import gc
from torch.cuda.amp import autocast, GradScaler

# def validation(args, model, test_save_path=None):
#     db_test = args.Dataset(base_dir=args.volume_path, split="test", list_dir=args.list_dir)
#     testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
#     logging.info("{} test iterations per epoch".format(len(testloader)))
#     model.eval()
#     metric_list = 0.0
#     MSE = torch.nn.modules.loss.MSELoss()
#
#     for i_batch, sampled_batch in tqdm(enumerate(testloader)):
#         # gc.collect()
#         ERI_batch, SC1_batch, SC2_batch, label_batch = sampled_batch['ERI'], sampled_batch['SC1'], sampled_batch[
#             'SC2'], sampled_batch['label']
#         ERI_batch, SC1_batch, SC2_batch, label_batch = ERI_batch.cuda(), SC1_batch.cuda(), SC2_batch.cuda(), label_batch.cuda()
#         with torch.no_grad():
#             outputs_batch = model(ERI_batch, SC1_batch, SC2_batch)
#             # Perform any necessary post-processing or evaluation metrics calculations here
#         outputs_batch = torch.mean(outputs_batch, dim=1)
#         label_batch = torch.mean(label_batch, dim=1)
#         print(sampled_batch['case_name'],outputs_batch,label_batch)
#         # Store the evaluation metrics results
#         # print(outputs_batch.shape,label_batch.shape)
#         loss = MSE(outputs_batch, label_batch)
#         print(loss)
#         metric_list += loss  # replace calculate_metrics with your own evaluation function

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    # num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    # TODO
    db_validate = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="test",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    # TODO
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    # TODO
    validateloader = DataLoader(db_validate, batch_size=9, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    # TODO
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    mse_loss = MSELoss()
    # dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            gc.collect()
            ERI_batch, SC1_batch, SC2_batch, label_batch = sampled_batch['ERI'], sampled_batch['SC1'], sampled_batch[
                'SC2'], sampled_batch['label']
            # print("原SC、ERI的shape为", SC1_batch.shape,ERI_batch.shape)
            ERI_batch, SC1_batch, SC2_batch, label_batch = ERI_batch.cuda(), SC1_batch.cuda(), SC2_batch.cuda(), label_batch.cuda()
            # print("SC的shape为",SC1_batch.shape)

            # with autocast():
            outputs_batch = model(ERI_batch, SC1_batch, SC2_batch)
            # 定义损失函数
            # loss_ce = ce_loss(outputs, label_batch[:].long())
            # loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = mse_loss(1.264911*outputs_batch.float(), 0.632456*label_batch.float())#4:1
            optimizer.zero_grad()
            loss.backward()
            torch.cuda.empty_cache()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            # writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f ' % (iter_num, loss.item()))
            #TODO

            if iter_num % 20 == 0:
                # image = image_batch[1, 0:1, :, :]
                # image = (image - image.min()) / (image.max() - image.min())
                # writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs_batch, dim=1), dim=1, keepdim=True)
                # writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                # writer.add_image('train/GroundTruth', labs, iter_num)
        model.eval()
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(validateloader):
                gc.collect()
                ERI_batch, SC1_batch, SC2_batch, label_batch = sampled_batch['ERI'], sampled_batch['SC1'], \
                sampled_batch[
                    'SC2'], sampled_batch['label']
                # print("原SC、ERI的shape为", SC1_batch.shape,ERI_batch.shape)
                ERI_batch, SC1_batch, SC2_batch, label_batch = ERI_batch.cuda(), SC1_batch.cuda(), SC2_batch.cuda(), label_batch.cuda()
                outputs_batch = model(ERI_batch, SC1_batch, SC2_batch)
                val_loss = mse_loss(2*outputs_batch.float(), label_batch.float())
        model.train()
        writer.add_scalar('info/val_loss', val_loss, iter_num)
        # writer.add_scalar('info/loss_ce', loss_ce, iter_num)
        logging.info('epoch %d : val_loss : %f' % (epoch_num, val_loss.item()))
        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
