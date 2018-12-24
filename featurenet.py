import time
import numpy as np
import tensorflow as tf
import cv2
import imports
from utils import *
from fr_utils import load_weights_from_FaceNet, img_to_encoding, cv2_img_to_encoding
from inception_blocks_v2 import faceRecoModel
from keras.models import load_model

model_width = 96
model_height = 96

# GRADED FUNCTION: triplet_loss

def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    ### START CODE HERE ### (≈ 4 lines)
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = pos_dist - neg_dist + alpha
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    ### END CODE HERE ###
    
    return loss

# GRADED FUNCTION: verify
def verify(feature_encoding, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".
    
    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras
    
    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """
    
    ### START CODE HERE ###
    
    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)  
    # encoding = img_to_encoding(image_path, model)    
    encoding = feature_encoding
    
    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm(encoding - database[identity])
    
    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    door_open = dist < 0.7
    # if door_open:
    #     print(f"It's {identity}, welcome home!")
    # else:
    #     print(f"It's not {identity}, please go away")
        
    ### END CODE HERE ###        
    return dist, door_open

def calculate_loss():
    with tf.Session() as test:
        tf.set_random_seed(1)
        y_true = (None, None, None)
        y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed = 1),
                tf.random_normal([3, 128], mean=1, stddev=1, seed = 1),
                tf.random_normal([3, 128], mean=3, stddev=4, seed = 1))
        loss = triplet_loss(y_true, y_pred)
        return loss.eval()

def cut_images(full_image_file, m_width, m_height, model_search_step, images_target_folder):
    ensure_dir(images_target_folder)
    im = cv2.imread(full_image_file)
    w, h = im.shape[:2]
    for x in range(0, w-m_width-1, model_search_step):
        for y in range(0, h-m_height-1, model_search_step):
            crop_img = im[x:x+m_width, y:y+m_height]
            cv2.imwrite(os.path.join(images_target_folder, f"image_{y}_{x}.png"), crop_img) # TODO It's not clear, why it's necessary to swap x and y

def cut_image(full_image_file, x, y, m_width, m_height, cut_image_file):
    x, y = y, x
    im = cv2.imread(full_image_file)
    crop_img = im[x:x+m_width, y:y+m_height]
    cv2.imwrite(cut_image_file, crop_img)

def scale_image(image_file, target_image_file):
    im = cv2.imread(image_file)
    resized_image = cv2.resize(im, (model_width, model_height))
    cv2.imwrite(target_image_file, resized_image)

def get_image_size(image_file):
    im = cv2.imread(image_file)
    w, h = im.shape[:2]
    return w,h

def load_feature_model(model_file):
    if os.path.exists(model_file):
        # https://github.com/keras-team/keras/issues/5916
        # import keras.losses
        # keras.losses.custom_loss = triplet_loss
        # FRmodel = load_model(model_file)
        start = time.time()
        model = load_model(model_file, custom_objects={'triplet_loss': triplet_loss})
        done = time.time()
        print(f"model loaded in {done-start} seconds")
    else:
        model = faceRecoModel(input_shape=(3, model_width, model_height))
        model.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
        load_weights_from_FaceNet(model)
        model.save(model_file)    
    print("Total Params:", model.count_params())
    return model

def calculate_score_for_images(model, feature_encoding, test_images_folder, detailed_print=False):
    database = {}
    database["feature"] = feature_encoding

    test_images_files = get_files_from_folder(test_images_folder)
    print(f"found {len(test_images_files)} files in folder {test_images_folder}")
    
    for test_image_file in test_images_files:
        test_image_key = os.path.basename(test_image_file)
        database[test_image_key] = img_to_encoding(test_image_file, model)
        
    if detailed_print:
        print("database:")
        print(database)

    all_scores = []
    for index, key in enumerate(database):
        score, threshold_status = verify(feature_encoding, key, database, model)
        all_scores.append((key, score))
    return all_scores

def calculate_score_for_full_image(model, feature_encoding, full_image_file, model_search_step, target_folder, feature_size=None, print_all_distances=False):
    im = cv2.imread(full_image_file)
    w, h = im.shape[:2]
    print(f"original image size: {w} x {h}")

    crop_width = model_width
    crop_height = model_height
    if feature_size is not None:
        crop_width, crop_height = feature_size

    points = []
    for x in range(0, w-crop_width-1, model_search_step):
        for y in range(0, h-crop_height-1, model_search_step):
            points.append((x, y))
    total = len(points)
    print(f"total points: {total}")

    start = time.time()
    list_of_full_distances = []
    for index, point in enumerate(points):
        x, y = point
        crop_img = im[x:x+crop_width, y:y+crop_height]
        if feature_size is not None:
            crop_img = cv2.resize(crop_img, (model_width, model_height))
        img = crop_img[...,::-1]
        img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
        x_train = np.array([img])
        encoding = model.predict_on_batch(x_train)
        dist = np.linalg.norm(feature_encoding - encoding)
        list_of_full_distances.append((x, y, dist))
    done = time.time()
    print(f"calculation all distances for {total}: {done-start} seconds")

    sorted_distances = list_of_full_distances
    sorted_distances.sort(key=lambda tup: tup[2])
    if print_all_distances:
        for item in sorted_distances:
            x, y, dist = item
            print(f"({x}, {y}) : {dist}")   # TODO It's not clear, why it's necessary to swap x and y
    best_point = sorted_distances[0]
    x, y, dist = best_point
    best_img = im[x:x+crop_width, y:y+crop_height]
    best_image_file = os.path.join(target_folder, f"target{y}_{x}.png")
    cv2.imwrite(best_image_file, best_img) # TODO It's not clear, why it's necessary to swap x and y
    print(f"best found image imported to file {best_image_file}")
    return best_image_file, sorted_distances

def calculate_score_for_full_image_2(model, feature_image_file, full_image_file, model_search_step, target_folder, print_all_distances=False):
    feature_image = cv2.imread(feature_image_file)
    feature_size = feature_image.shape[:2]
    print(f"feature image: {feature_size}")
    resized_feature_image = cv2.resize(feature_image, (model_width, model_height))

    feature_encoding = cv2_img_to_encoding(resized_feature_image, model)
    best_image_file, sorted_distances = calculate_score_for_full_image(model, feature_encoding, full_image_file, model_search_step, target_folder, feature_size, print_all_distances)
    return best_image_file, sorted_distances