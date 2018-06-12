import cv2, os
import numpy as np
import matplotlib.image as mpimg
import random
from random import shuffle
import sklearn

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3

def get_image_path(source_path):
    """Convert static global path to local path"""

    split = source_path.split('\\')
    # get filename
    filename = split[-1].lstrip()
    # get folder name
    folder = split[-3]
    # get full data path
    current_path = folder + '/IMG/' + filename
    return current_path

def flip_image(image, measurement, flip_probability=1.0):
    if random.random() <= flip_probability:
        image = cv2.flip(image, 1)
        measurement*=-1
    return image, measurement

def add_salt_pepper_noise(img):
    """
    Randomly add Salt and pepper noise
    """
    # Need to produce a copy as to not modify the original image
    dice = random.randint(0, 100)

    if (dice < 30):
        row, col, _ = img.shape
        salt_vs_pepper = 0.20
        amount = 0.030
        num_salt = np.ceil(amount * img.size * salt_vs_pepper)
        num_pepper = np.ceil(amount * img.size * (1.0 - salt_vs_pepper))

        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
        img[coords[0], coords[1], :] = 255

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
        img[coords[0], coords[1], :] = 0
    return img

def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (From Nvidia paper)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def preprocess(image):
    """Preprocess image on validation or before applying augmentation"""
    image = rgb2yuv(image)
    return image

def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def generator(samples, batch_size=32, is_training=True):
    """
    Lazy batch train/validation generator for memory efficiency
    """
    num_samples = len(samples)

    #vertical, horizontal range for random translation
    x_translate_range = 100
    y_translate_range = 10

    while 1: # Loop forever so the generator never terminates
        #shuffle the samples once the whole data is processed into batches
        shuffle(samples)
        #split data into batches
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                # corrections for centered view image, left camera view image and right camera view image
                corrections = [0,0.2,-0.2]
                # iterate over center, right and left camera view images
                for i in range(3):
                    current_path = get_image_path(batch_sample[i])

                    # read image
                    image = cv2.imread(current_path)
                    # append image for training/validation
                    images.append(preprocess(image))

                    # calculate angle measurement with applied angle corrections
                    measurement = float(batch_sample[3]) + corrections[i]
                    angles.append(measurement)

                    # insert flipped image for opposite direction generalization
                    images.append(preprocess(cv2.flip(image, 1)))
                    angles.append(measurement*-1.0)

                    # create random augmented image only for training
                    if is_training:
                        image, measurement = flip_image(image, measurement, flip_probability=0.5)
                        image = add_salt_pepper_noise(image)
                        image, measurement = random_translate(image, measurement, x_translate_range, y_translate_range)
                        image = random_shadow(image)
                        image = random_brightness(image)
                        images.append(preprocess(image))
                        angles.append(measurement)

            # create X, y dataset
            X_train = np.array(images)
            y_train = np.array(angles)

            #
            # n, bins, patches = plt.hist(np.array(y_train), bins=20)
            # plt.show()
            yield sklearn.utils.shuffle(X_train, y_train)