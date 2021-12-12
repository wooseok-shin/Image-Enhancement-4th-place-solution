import glob
import os
# import time
import zipfile
from os.path import join as opj
import cv2
from tqdm import tqdm

import numpy as np
import torch
# import torch.nn as nn

from network import HINet

def predict(images_path, model, batch_size, img_size, stride):
    predictions = []
    with torch.no_grad():
        for img_path in tqdm(images_path, desc = 'Images prediction'):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img,(2,0,1))
            img = torch.from_numpy(img)/255.0
            img_shape = img.shape 
            
            result_img = torch.zeros_like(img, device='cuda')
            voting_mask = torch.zeros_like(img, device='cuda')
            batch_count = 0
            crop ,position = [], []

            top_margin  = int((np.ceil((img_shape[1] - img_size)/stride)) * stride)
            left_margin = int((np.ceil((img_shape[2] - img_size)/stride)) * stride)
            for top in range(0, top_margin+1, stride):
                for left in range(0, left_margin+1, stride):
                    piece = torch.zeros([3, img_size, img_size]) 
                    temp = img[:, top:top+img_size, left:left+img_size] 
                    piece[:,:temp.shape[1], :temp.shape[2]] = temp 
                    crop.append(piece)
                    position.append([top, left])
                    batch_count += 1
                    if batch_count == batch_size :
                        crop = torch.stack(crop) 
                        crop = crop.cuda()
                        pred = model(crop)[1]
                        batch_count = 0
                        for num, (t, l) in enumerate(position):
                            piece = pred[num]
                            _, h, w = result_img[:,t:t+img_size, l:l+img_size].shape
                            result_img[:,t:t+img_size, l:l+img_size] += piece[:,:h, :w]
                            voting_mask[:,t:t+img_size, l:l+img_size] += 1
                        crop ,position = [], []

            if len(crop) > 0:           
                crop = torch.stack(crop)
                crop = crop.cuda()
                pred = model(crop)[1]
                for num, (t, l) in enumerate(position):
                    piece = pred[num]
                    _, h, w = result_img[:,t:t+img_size, l:l+img_size].shape
                    result_img[:,t:t+img_size, l:l+img_size] += piece[:,:h, :w]
                    voting_mask[:,t:t+img_size, l:l+img_size] += 1

            result_img = (result_img * 255.0)/voting_mask
            # assert img_shape == result_img.shape, 'Input != Output Shape'

            result_img = result_img.permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
            result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
            predictions.append(result_img)
        predictions = np.stack(predictions) # (B, H, W, C)

    return predictions

def prediction(images_path, model, weight_path, batch_size, img_size, stride):
    model.load_state_dict(torch.load(weight_path)['state_dict'])
    model.eval() 
    predictions = predict(images_path, model, batch_size, img_size, stride)
    
    return predictions

# s = time.time()
images_path = glob.glob('./data/test_input_img/*')
weight_path_256 = './weights/256_best.pth'
weight_path_352 = './weights/352_best.pth'

device = torch.device('cuda')
model = HINet().to(device)


predictions_256 = prediction(images_path, model, weight_path_256, batch_size = 16, img_size= 256, stride= 64)
predictions_352 = prediction(images_path, model, weight_path_352, batch_size = 32, img_size= 352, stride= 96)
ensemble_predictions = 0.4 * predictions_256 +  0.6 * predictions_352

save_path = './results/submission'
os.makedirs(save_path, exist_ok=True)
for idx in range(len(images_path)):
    image_path = images_path[idx]
    image_name = os.path.basename(image_path)
    cv2.imwrite(opj(save_path, (image_name[:5] + image_name[-9:])), ensemble_predictions[idx])

results = os.listdir(save_path)
os.chdir(save_path + '/')
submission = zipfile.ZipFile('submission.zip', 'w')
for result in results:
    submission.write(result)
submission.close()
# print(f'Total Time : {time.time() - s:.2f}')
