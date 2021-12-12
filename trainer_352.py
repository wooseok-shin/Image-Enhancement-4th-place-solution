import os
import glob
import time
import logging
import pandas as pd
import torch.nn as nn
import utils
from copy import deepcopy
from natsort import natsorted
from tqdm import tqdm
from dataloader import *
from losses import *
from warmup_scheduler import GradualWarmupScheduler
from network import HINet

class Trainer():
    def __init__(self, args, save_path):
        '''
        args: arguments
        save_path: Model 가중치 저장 경로
        '''
        super(Trainer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Train, Valid Set load
        self.trn_img_folder = os.path.join(args.data_path, 'train', args.dataset, args.fold, f'resize{args.img_size}/input')
        self.trn_tar_folder = os.path.join(args.data_path, 'train', args.dataset, args.fold, f'resize{args.img_size}/target')

        df = pd.read_csv(os.path.join(args.data_path, 'train_original', 'train_10fold.csv'))
        val_idx = list(df[df['fold'] == int(args.fold)].index)

        self.val_img_folder = np.array(natsorted(glob.glob(os.path.join(args.data_path, 'train_original/train_input_img') + '/*.png')))[val_idx]
        self.val_tar_folder = np.array(natsorted(glob.glob(os.path.join(args.data_path, 'train_original/train_label_img') + '/*.png')))[val_idx]

        # Augmentation
        self.train_transform = get_train_augmentation(img_size=args.img_size, ver=1)   # 첫 n epoch은 Augmentation 없이 수행 (Aug ver=1)
        self.test_transform = get_test_augmentation(img_size=args.img_size)

        # TrainLoader
        self.train_loader = get_loader(self.trn_img_folder, self.trn_tar_folder, phase='train', batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, transform=self.train_transform)

        # Network
        self.model = HINet().to(self.device)

        # HINet Pretrained Weight Load (remove unnecessary module.)
        if args.pretrained_path is not None:
            load_net = torch.load(args.pretrained_path)
            load_net = load_net['params']
            print(' load net keys', load_net.keys)
            for k, v in deepcopy(load_net).items():
                if k.startswith('module.'):
                    load_net[k[7:]] = v
                    load_net.pop(k)

            self.model.load_state_dict(load_net, strict=True)

        # Loss
        self.criterion_char = CharbonnierLoss()
        self.criterion_edge = EdgeLoss()

        # Optimizer & Scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.initial_lr, betas=(0.9, 0.999), eps=1e-8)
        warmup_epochs = 3
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.epochs-warmup_epochs, eta_min=args.min_lr)
        self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

        ######### Logging #########
        log_file = os.path.join(save_path, 'log.log')
        self.logger = utils.get_root_logger(logger_name='IR', log_level=logging.INFO, log_file=log_file)
        self.logger.info(args)

        if args.multi_gpu:
            self.model = nn.DataParallel(self.model).to(self.device)

        # Train / Validate
        best_psnr = 0
        best_epoch = 0
        early_stopping = 0
        start = time.time()
        for epoch in range(1, args.epochs+1):
            self.epoch = epoch

            # Adaptive Augmentaiton  # 6epoch에서 augmentation ver2로 변환, 25epoch에서 ver3으로 변환
            if epoch == 6 or epoch == 25:
                aug_ver = 2 if epoch == 6 else 3
                self.train_transform = get_train_augmentation(img_size=args.img_size, ver=aug_ver)
                self.train_loader = get_loader(self.trn_img_folder, self.trn_tar_folder, phase='train',
                                                    batch_size=args.batch_size, shuffle=True,
                                                    num_workers=args.num_workers, transform=self.train_transform,
                                                )
                print(f'Augmentation Version Change to {aug_ver}')


            # Decay LR: 특정 Epoch에서 CosineAnnealingLR의 초기 LR과 Min LR을 동시에 낮춰주기 -> lr 5e-5 to 1e-5 & eta_min 1e-7 to 1e-8
            if epoch == 25:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8)
                scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.epochs-warmup_epochs, eta_min=1e-8)
                self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

            # Training
            self.training(args)
            self.scheduler.step()

            # Model weight in Multi_GPU or Single GPU
            state_dict= self.model.module.state_dict() if args.multi_gpu else self.model.state_dict()

            # Predict Validation (Whole Image로 합친 Validation score tracking)
            val_psnr = self.predict(img_paths=self.val_img_folder, tar_paths=self.val_tar_folder, img_size=args.img_size, stride=(args.img_size)//2, batch_size=args.pred_batch)   # validate
            self.logger.info(f'{epoch}Epoch - Val PSNR:{val_psnr:.4f}')

            # Save models
            if val_psnr > best_psnr:
                early_stopping = 0
                best_epoch = epoch
                best_psnr = val_psnr

                torch.save({'epoch':epoch,
                            'state_dict':state_dict,
                            'optimizer': self.optimizer.state_dict()
                    }, os.path.join(save_path, '352_best.pth'))
                self.logger.info(f'-----------------SAVE:{best_epoch}epoch----------------')
            else:
                early_stopping += 1

            # Early Stopping
            if early_stopping == args.patience:
                break

        self.logger.info(f'\nBest Val Epoch:{best_epoch} | Val PSNR:{best_psnr:.4f}')
        end = time.time()
        self.logger.info(f'Total Process time:{(end - start) / 60:.3f}Minute')

    # Training
    def training(self, args):
        self.model.train()
        train_loss = utils.AvgMeter()
        train_psnr = utils.AvgMeter()

        for i, (images, targets) in enumerate(tqdm(self.train_loader)):
            images = torch.tensor(images, device=self.device, dtype=torch.float32)
            targets = torch.tensor(targets, device=self.device, dtype=torch.float32)

            self.optimizer.zero_grad()

            restore_inputs = self.model(images)
            loss_char = sum([self.criterion_char(restore_inputs[j], targets) for j in range(len(restore_inputs))])  # Charbonnier Loss
            loss_edge = sum([self.criterion_edge(restore_inputs[j], targets) for j in range(len(restore_inputs))])  # Edge Loss
            loss = (loss_char) + (0.05*loss_edge)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), args.clipping)
            self.optimizer.step()

            # Metric
            psnr = utils.torchPSNR(targets, restore_inputs[1])

            # log
            train_loss.update(loss.item(), n=images.size(0))
            train_psnr.update(psnr.item(), n=images.size(0))

        self.logger.info(f'Epoch:[{self.epoch:03d}/{args.epochs:03d}]')
        self.logger.info(f'Train Loss:{train_loss.avg:.3f} | PSNR:{train_psnr.avg:.3f}')
            
    
    # 전체 이미지 단위 Validation
    def predict(self, img_paths, tar_paths, img_size=256, stride=128, batch_size=64):
        self.model.eval()
        with torch.no_grad():
            print('Number of validation images', len(img_paths))
            results = []
            for img_path, tar_path in zip(img_paths, tar_paths):
                tar = cv2.imread(tar_path)
                tar = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)

                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32)/255
                crop = []
                position = []
                batch_count = 0

                result_img = np.zeros_like(img)
                voting_mask = np.zeros_like(img)

                for top in range(0, img.shape[0], stride):
                    for left in range(0, img.shape[1], stride):
                        piece = np.zeros([img_size, img_size, 3], np.float32)
                        temp = img[top:top+img_size, left:left+img_size, :]
                        piece[:temp.shape[0], :temp.shape[1], :] = temp
                        crop.append(piece)
                        position.append([top, left])
                        batch_count += 1
                        if batch_count == batch_size:
                            crop = np.array(crop)
                            crop = torch.as_tensor(crop).permute(0,3,1,2).cuda()

                            with torch.no_grad():
                                pred = self.model(crop)[1]*255
                                pred = pred.permute(0,2,3,1).detach().cpu().numpy()
                            crop = []
                            batch_count = 0
                            for num, (t, l) in enumerate(position):
                                piece = pred[num]
                                h, w, c = result_img[t:t+img_size, l:l+img_size, :].shape
                                result_img[t:t+img_size, l:l+img_size, :] += piece[:h, :w]
                                voting_mask[t:t+img_size, l:l+img_size, :] += 1
                            position = []
                
                result_img = result_img/voting_mask
                result_img = result_img.astype(np.uint8)

                psnr_score = utils.numpyPSNR(tar, result_img)
                results.append(psnr_score)
            psnr_score = np.mean(results)
            
        return psnr_score
