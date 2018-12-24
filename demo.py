import sys
import os
import os.path
import time
import cv2
import numpy as np
import imports
from featurenet import *
from fr_utils import load_weights_from_FaceNet, img_to_encoding
from utils import *
from inception_blocks_v2 import faceRecoModel
from keras.models import load_model

def init():
    np.set_printoptions(threshold=np.nan)


if __name__ == "__main__":
    print("[main] started")
    model_file = 'model.h5'
    root_folder = 'data/data1/'
    feature_file = os.path.join(root_folder, 'feature_343_220.png')
    target_folder = root_folder

    print(f"model_width: {model_width}")
    print(f"model_height: {model_height}")
    
    debug_print_enabled = False
    debug_calculcate_lost_enabled = False

    test_images_folder = None
    full_image_file = None
    model_search_step = None
    best_found_image_taget_file = None
    # 
    # "calculate_score_for_full_image"
    # "calculate_score_for_test_images"
    # "calculate_score_for_full_image_2"
    # "cut_images"
    # "cut_image"
    # "rescale_image"
    # 
    run_mode = "calculate_score_for_full_image_2"

    print(f"mode: {run_mode}")
    print(f"feature: {feature_file}")

    # 
    # mode calculate_score_for_test_images
    # 
    if run_mode == "calculate_score_for_test_images":
        test_images_folder = "data/data1/cut_images"
        # test_images_files = [os.path.abspath(os.path.join(test_images_folder, x)) for x in os.listdir(test_images_folder)]
        test_images_files = get_files_from_folder(test_images_folder)
        print(f"found {len(test_images_files)} files in folder {test_images_folder}")

    # 
    # mode calculate_score_for_full_image
    # 
    if run_mode == "calculate_score_for_full_image":
        full_image_file = "data/data1/model.bmp"
        model_search_step = 20
        print(f"full model file {os.path.abspath(full_image_file)}")

    #
    # mode cut_images
    #
    if run_mode == "cut_images":
        full_image_file = "data/data1/model.bmp"
        cut_images_folder = os.path.join(target_folder, "cut_images")
        model_search_step = 20

        cut_images(full_image_file, model_width, model_height, model_search_step, cut_images_folder)
        sys.exit(0)

    #
    # mode cut_images
    #
    if run_mode == "cut_image":
        full_image_file = "data/data1/model.bmp"
        x = 343
        y = 220
        cut_image(full_image_file, x, y, model_width, model_height, os.path.join(target_folder, f"cut_image_{x}_{y}.png"))
        x = 220
        y = 343
        cut_image(full_image_file, x, y, model_width, model_height, os.path.join(target_folder, f"cut_image_{x}_{y}.png"))
        sys.exit(0)

    # 
    # rescale_image
    # 
    if run_mode == "rescale_image":
        full_image_file = "data/data1/model.bmp"
        x = 343
        y = 220
        big_cut_image_file = os.path.join(target_folder, f"big_cut_image_{x}_{y}.png")
        cut_image(full_image_file, x, y, model_width*2, model_height*2, big_cut_image_file)

        big2standard_cut_image_file = os.path.join(target_folder, f"big_rescalled_cut_image_{x}_{y}.png")
        scale_image(big_cut_image_file, big2standard_cut_image_file)
        
        sys.exit(0)

    init()

    if debug_calculcate_lost_enabled:
        loss_result = calculate_loss()
        print(f"loss = {loss_result}")

    FRmodel = load_feature_model(model_file)

    feature_encoding = img_to_encoding(feature_file, FRmodel)

    if run_mode == "calculate_score_for_test_images":
        all_scores = calculate_score_for_images(FRmodel, feature_encoding, test_images_folder, detailed_print=False)

    if run_mode == "calculate_score_for_full_image":
        calculate_score_for_full_image(FRmodel, feature_encoding, full_image_file, model_search_step, target_folder, print_all_distances=True)

    if run_mode == "calculate_score_for_full_image_2":
        full_image_file = "data/data1/model.bmp"
        model_search_step = 20
        big_feature_file = os.path.join(root_folder, 'big_cut_image_343_220.png')
        calculate_score_for_full_image_2(FRmodel, big_feature_file, full_image_file, model_search_step, target_folder, print_all_distances=True)

    print("[main] finished")