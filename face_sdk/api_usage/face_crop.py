"""
@author: JiXuan Xu, Jun Wang
@date: 20201015
@contact: jun21wangustc@gmail.com 
"""
import sys
sys.path.append("../")
import cv2
from tqdm import tqdm
import os

from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper

if __name__ == '__main__':
    image_dir = "/mnt/nvme1n1p1/kabir/Datasets/ms-celeb-1m-v1c/train_msra/"
    output_dir = os.path.join(image_dir,"msra_crop")
    # line = open(image_info_file).readline().strip()
    lmk_file = os.path.join(image_dir, "msra_lmk")
    landmarks_info = open(lmk_file).readlines()

    for landmark_info in tqdm(landmarks_info):
        landmarks_str = landmark_info.split(' ')
        landmarks = [float(num) for num in landmarks_str[-10:]]
        image_fil = landmarks_str[0]
        image_path = os.path.join(image_dir, image_fil)
        face_cropper = FaceRecImageCropper()
        image = cv2.imread(image_path)
        cropped_image = face_cropper.crop_image_by_mat(image, landmarks)
        output_path = os.path.join(output_dir, image_fil)
        output_folder = "/".join(output_path.split("/")[:-1])
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        cv2.imwrite(output_path, cropped_image)
        # print("Success")
        # break
