import os
import cv2
import glob
import natsort
import numpy as np
import pandas as pd
import argparse

from multiprocessing import Pool
from tqdm import tqdm
from os.path import join as opj
from os import path as osp
from sklearn.model_selection import KFold

parser = argparse.ArgumentParser()
parser.add_argument('--img_size', default=256, type=int)
parser.add_argument('--stride', default=128, type=int)
parser.add_argument('--resize', default=256, type=int)
parser.add_argument('--v_threshold', default=1000, type=int)
parser.add_argument('--folds', default=10, type=int)
parser.add_argument('--valid_fold', default=0, type=int)
parser.add_argument('--patch_growth_count', default=2, type=int)
parser.add_argument('--process_num', default=8, type=int)
parser.add_argument('--input_path', default='./data/train_original/train_input_img/', type=str)
parser.add_argument('--target_path', default='./data/train_original/train_label_img/', type=str)
parser.add_argument('--csv_path', default='./data/train_original', type=str)
parser.add_argument('--save_path', default='./data', type=str)


def worker_352(img_path, tar_path, img_size, stride, img_name, tar_name, save_path, resize, v_threshold):
    '''
    1. image size 352 :
    i)  stride를 밀면서 패치의 시작 index가 원본이미지를 초과하지 않는 패치만 사용
    ii) i)조건을 만족하고 hsv중 v에 해당하는 명도가 큰 부분만 사용

    2. image size 352보다 큰 경우 :
    stride를 밀면서 원본 이미지를 초과하지 않는 패치만 사용
    '''
    if img_size == 352 :
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        tar = cv2.imread(tar_path, cv2.IMREAD_UNCHANGED)

        assert img.shape == tar.shape, 'Image != Target Shape'
        num = 0
        for top in range(0, img.shape[0], stride):
            for left in range(0, img.shape[1], stride):
                cropped_img = np.zeros([img_size, img_size, 3], np.uint8)
                temp = img[top:top+img_size, left:left+img_size, :]
                cropped_img[:temp.shape[0], :temp.shape[1], :] = temp
                
                # hsv의 value>150인 픽셀이 적을경우 pass
                hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
                _ ,_, v = cv2.split(hsv)
                if len(np.where(v>200)[0]) < v_threshold:
                    continue

                cropped_tar = np.zeros([img_size, img_size, 3], np.uint8)
                temp = tar[top:top+img_size, left:left+img_size, :]
                cropped_tar[:temp.shape[0], :temp.shape[1], :] = temp

                cv2.imwrite(f'{save_path}/resize{resize}/input/{str(num).zfill(5)}_{img_name}', cropped_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                cv2.imwrite(f'{save_path}/resize{resize}/target/{str(num).zfill(5)}_{tar_name}', cropped_tar, [cv2.IMWRITE_PNG_COMPRESSION, 3])

                num += 1

    else :        
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        tar = cv2.imread(tar_path, cv2.IMREAD_UNCHANGED)

        assert img.shape == tar.shape, 'Image != Target Shape'

        h, w, c = img.shape

        h_space = np.arange(0, h - img_size + 1, stride)
        if h - (h_space[-1] + img_size) > 0:
            h_space = np.append(h_space, h - img_size)
        w_space = np.arange(0, w - img_size + 1, stride)
        if w - (w_space[-1] + img_size) > 0:
            w_space = np.append(w_space, w - img_size)

        num = 0
        for x in h_space:
            for y in w_space:
                cropped_img = img[x:x + img_size, y:y + img_size, ...]
                cropped_tar = tar[x:x + img_size, y:y + img_size, ...]
                cropped_img = np.ascontiguousarray(cropped_img)
                cropped_tar = np.ascontiguousarray(cropped_tar)
                cropped_img = cv2.resize(cropped_img, dsize=(resize, resize), interpolation=cv2.INTER_CUBIC)
                cropped_tar = cv2.resize(cropped_tar, dsize=(resize, resize), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(f'{save_path}/resize{resize}/input/{str(num).zfill(5)}_{img_size}_{img_name}', cropped_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                cv2.imwrite(f'{save_path}/resize{resize}/target/{str(num).zfill(5)}_{img_size}_{tar_name}', cropped_tar, [cv2.IMWRITE_PNG_COMPRESSION, 3])

                num += 1

def worker_256(img_path, tar_path, img_size, stride, img_name, tar_name, save_path, resize, v_threshold):
    if img_size == 256 :
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        tar = cv2.imread(tar_path, cv2.IMREAD_UNCHANGED)

        assert img.shape == tar.shape, 'Image != Target Shape'
        num = 0
        for top in range(0, img.shape[0], stride):
            for left in range(0, img.shape[1], stride):
                cropped_img = np.zeros([img_size, img_size, 3], np.uint8)
                temp = img[top:top+img_size, left:left+img_size, :]
                cropped_img[:temp.shape[0], :temp.shape[1], :] = temp
                
                # hsv의 value>150인 픽셀이 적을경우 pass
                hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
                _ ,_, v = cv2.split(hsv)
                if len(np.where(v>200)[0]) < v_threshold:
                    continue

                cropped_tar = np.zeros([img_size, img_size, 3], np.uint8)
                temp = tar[top:top+img_size, left:left+img_size, :]
                cropped_tar[:temp.shape[0], :temp.shape[1], :] = temp

                cv2.imwrite(f'{save_path}/resize{resize}/input/{str(num).zfill(5)}_{img_name}', cropped_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                cv2.imwrite(f'{save_path}/resize{resize}/target/{str(num).zfill(5)}_{tar_name}', cropped_tar, [cv2.IMWRITE_PNG_COMPRESSION, 3])

                num += 1

    else :        
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        tar = cv2.imread(tar_path, cv2.IMREAD_UNCHANGED)

        assert img.shape == tar.shape, 'Image != Target Shape'

        h, w, c = img.shape

        h_space = np.arange(0, h - img_size + 1, stride)
        if h - (h_space[-1] + img_size) > 0:
            h_space = np.append(h_space, h - img_size)
        w_space = np.arange(0, w - img_size + 1, stride)
        if w - (w_space[-1] + img_size) > 0:
            w_space = np.append(w_space, w - img_size)

        num = 0
        for x in h_space:
            for y in w_space:
                cropped_img = img[x:x + img_size, y:y + img_size, ...]
                cropped_tar = tar[x:x + img_size, y:y + img_size, ...]
                cropped_img = np.ascontiguousarray(cropped_img)
                cropped_tar = np.ascontiguousarray(cropped_tar)
                cropped_img = cv2.resize(cropped_img, dsize=(resize, resize), interpolation=cv2.INTER_CUBIC)
                cropped_tar = cv2.resize(cropped_tar, dsize=(resize, resize), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(f'{save_path}/resize{resize}/input/{str(num).zfill(5)}_{img_size}_{img_name}', cropped_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                cv2.imwrite(f'{save_path}/resize{resize}/target/{str(num).zfill(5)}_{img_size}_{tar_name}', cropped_tar, [cv2.IMWRITE_PNG_COMPRESSION, 3])

                num += 1

def extract_subimages_resize(img_path_list, tar_path_list, save_path, img_size = None, stride = None, v_threshold = 1000, process_num = 8, resize = None):
    os.makedirs(f'{save_path}/resize{resize}/input/', exist_ok=True)
    os.makedirs(f'{save_path}/resize{resize}/target/', exist_ok=True)
    pbar = tqdm(total=len(img_path_list), unit='image', desc='Extract')
    pool = Pool(processes=process_num)
    for img_path, tar_path in zip(img_path_list, tar_path_list):
        img_name = os.path.basename(img_path)
        tar_name = os.path.basename(tar_path)
        if resize == 352 :
            pool.apply_async(worker_352, args=(img_path, tar_path, img_size, stride, img_name, tar_name, save_path, resize, v_threshold), callback=lambda arg: pbar.update(1))
        elif resize == 256 :
            pool.apply_async(worker_256, args=(img_path, tar_path, img_size, stride, img_name, tar_name, save_path, resize, v_threshold), callback=lambda arg: pbar.update(1))
        else : 
            assert 'Resize values error'

    pool.close()
    pool.join()
    pbar.close()
    print(f'{img_size} to {resize} Processes done.')
        
def make_csv(csv_path, folds) :
    df = pd.read_csv(f'{csv_path}/train.csv')
    df['fold'] = 0

    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(df)))):
        df.loc[val_idx, 'fold'] = fold
    df.to_csv(f'{csv_path}/train_10fold.csv', index=False)

    return df

def main() :
    '''
    1. 256size train 데이터셋 구성
    2. 512size train 데이터셋 구성
    '''
    args = parser.parse_args()

    img_size = args.img_size
    stride = args.stride
    v_threshold = args.v_threshold
    process_num = args.process_num
    csv_path = args.csv_path
    save_path = args.save_path
    input_path = args.input_path
    target_path = args.target_path
    resize = args.resize
    folds = args.folds
    valid_fold = args.valid_fold
    patch_growth_count = args.patch_growth_count

    df = make_csv(csv_path, folds)

    for fold in range(10):
        if fold != valid_fold :
            continue

        train_idx = list(df[df['fold'] != fold].index)
        train_input_files = np.array(natsort.natsorted(glob.glob(input_path + '*.png')))[train_idx]
        train_label_files = np.array(natsort.natsorted(glob.glob(target_path + '*.png')))[train_idx]

        for i in range(1, patch_growth_count+1) :
            if i == 2 :
                extract_subimages_resize(train_input_files, train_label_files, opj(save_path, 'train', '10Folds', str(fold)), 512, 256, v_threshold = v_threshold, process_num = process_num, resize = resize)
            else : 
                extract_subimages_resize(train_input_files, train_label_files, opj(save_path, 'train', '10Folds', str(fold)), img_size*i, stride*i, v_threshold = v_threshold, process_num = process_num, resize = resize)


if __name__ == '__main__':
    main()