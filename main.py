#!/usr/bin/env ipython
import glob
import matplotlib
import time
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label

from settings import (NUM_SAMPLES,
                      TEST_IMAGES_DIR,
                      INPUT_VIDEOFILE,
                      OUTPUT_DIR,
                      TRAINING_DIR,
                      VEHICLES_DIR,
                      NON_VEHICLES_DIR,
                      SAVE_DIR,
                      COLORSPACE,
                      ORIENT,
                      PIX_PER_CELL,
                      CELL_PER_BLOCK,
                      HOG_CHANNEL,
                      Y_START_STOP,
                      MODEL_FILE,
                      DATASET_FILE,
                      TRAIN_TEST_SPLIT,
                      SPATIAL_SIZE,
                      HIST_BINS,
                      SPATIAL_FEAT,
                      HIST_FEAT,
                      HOG_FEAT,
                      XY_WINDOW,
                      XY_OVERLAP,
                      HEAT_THRESHOLD,
                      VIDEO_MODE)
from utils import (draw_boxes,
                   color_hist,
                   extract_features_hog,
                   slide_window,
                   search_windows,
                   add_heat,
                   apply_threshold,
                   draw_labeled_bboxes,
                   joblib_save,
                   joblib_load,
                   imcompare,
                   display,
                   debug)

matplotlib.use('TkAgg')  # MacOSX Compatibility
matplotlib.interactive(True)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

# TODO(Manav): Classes:
# Heatmap
# sliding window
# hog
# classifer/model


class VehicleDetection(object):

    def __init__(self, model_file=None, dataset_file=None):
        if model_file:
            self.svc = joblib_load(model_file)
        else:
            self.svc = None

        if dataset_file:
            [self.X_scaler, self.scaled_X, self.y] = joblib_load(dataset_file)
        else:
            [self.X_scaler, self.scaled_X, self.y] = [None, None, None]
            # svc, [X_scaler, scaled_X, y] = train_or_load_model(cars, notcars)

        self.count = 0  # tdqm
        self.init()     # Feature Extraction and Sliding Window Search Params

    def init(self):
        # Features Extraction
        self.color_space = COLORSPACE
        self.orient = ORIENT
        self.pix_per_cell = PIX_PER_CELL
        self.cell_per_block = CELL_PER_BLOCK
        self.hog_channel = HOG_CHANNEL
        self.spatial_size = SPATIAL_SIZE
        self.hist_bins = HIST_BINS
        self.spatial_feat = SPATIAL_FEAT
        self.hist_feat = HIST_FEAT
        self.hog_feat = HOG_FEAT

        # Sliding Window Search
        self.y_start_stop = Y_START_STOP
        self.xy_window = XY_WINDOW
        self.xy_overlap = XY_OVERLAP

    def sliding_window_search(self, image):
        try:
            start = time.time()
            # Uncomment the following line if you extracted training
            # data from .png images (scaled 0 to 1 by mpimg) and the
            # image you are searching is a .jpg (scaled 0 to 255)
            image = image.astype(np.float32)/255

            windows = slide_window(image, x_start_stop=[None, None], y_start_stop=self.y_start_stop,
                                   xy_window=self.xy_window, xy_overlap=self.xy_overlap)

            hot_windows = search_windows(image, windows, self.svc, self.X_scaler,
                                         color_space=self.color_space,
                                         spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                         orient=self.orient, pix_per_cell=self.pix_per_cell,
                                         cell_per_block=self.cell_per_block,
                                         hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                         hist_feat=self.hist_feat, hog_feat=self.hog_feat)
            end = time.time()
            self.windows = windows
            self.hot_windows = hot_windows
            draw_image = np.copy(image)
            # window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
            print('%0.1f seconds/frame. #%d/%d hot-windows/windows/frame' % (end-start,
                                                                             len(hot_windows), len(windows)))
            heat_thresholded_image, thresholded_heatmap, labels = heat_and_threshold(draw_image, self.hot_windows, threshold=1)

            # title1 = 'Car Positions (#Detections: %d)' % (labels[1])
            # title2 = 'Thresholded Heat Map (Max: %d)' % int(np.max(thresholded_heatmap))
            # imcompare(heat_thresholded_image, thresholded_heatmap, title1, title2, cmap2='hot')

            self.save = heat_thresholded_image
        except:
            import ipdb; ipdb.set_trace()
            mpimg.imsave('hard/%d.jpg' % self.count, image)
            debug('Error: Issue at Frame %d' % self.count)
            heat_thresholded_image = self.save

        self.count += 1

        if VIDEO_MODE:
            # Scale Back to Format acceptable by moviepy
            heat_thresholded_image = heat_thresholded_image.astype(np.float32) * 255

        return heat_thresholded_image


def test_heatmap_threshold_label(heatmap):
    heatmap = apply_threshold(heatmap, 2)
    labels = label(heatmap)
    # print(labels[1], 'cars found')
    plt.imshow(labels[0], cmap='gray')


def test_color_hist(filename):
    image = mpimg.imread(filename)
    rh, gh, bh, bincen, feature_vec = color_hist(image, nbins=32, bins_range=(0, 256))

    # Plot a figure with all three bar charts
    if rh is not None:
        fig = plt.figure(figsize=(12,3))
        plt.subplot(131)
        plt.bar(bincen, rh[0])
        plt.xlim(0, 256)
        plt.title('R Histogram')
        plt.subplot(132)
        plt.bar(bincen, gh[0])
        plt.xlim(0, 256)
        plt.title('G Histogram')
        plt.subplot(133)
        plt.bar(bincen, bh[0])
        plt.xlim(0, 256)
        plt.title('B Histogram')
        fig.tight_layout()
    else:
        print('Your function is returning None for at least one variable...')


def test_svc_color_hist(cars, notcars):
    # TODO play with these values to see how your classifier
    # performs under different binning scenarios
    spatial = 32
    histbin = 32

    car_features = extract_features(cars, color_space='HSV', spatial_size=(spatial, spatial),
                                    hist_bins=histbin, hist_range=(0, 256))
    notcar_features = extract_features(notcars, color_space='HSV', spatial_size=(spatial, spatial),
                                       hist_bins=histbin, hist_range=(0, 256))

    # TODO: Move Normalization Post Test/Train Split

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using spatial binning of:',spatial,
          'and', histbin,'histogram bins')
    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')


def train_svc_with_color_hog_hist(cars, notcars):
    # TODO: Tweak these parameters and see how the results change.
    color_space = COLORSPACE
    orient = ORIENT
    pix_per_cell = PIX_PER_CELL
    cell_per_block = CELL_PER_BLOCK
    hog_channel = HOG_CHANNEL
    spatial_size = SPATIAL_SIZE
    hist_bins = HIST_BINS
    spatial_feat = SPATIAL_FEAT
    hist_feat = HIST_FEAT
    hog_feat = HOG_FEAT

    t = time.time()
    car_features = extract_features_hog(cars, color_space=color_space,
                                        spatial_size=spatial_size, hist_bins=hist_bins,
                                        orient=orient, pix_per_cell=pix_per_cell,
                                        cell_per_block=cell_per_block,
                                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                                        hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features_hog(notcars, color_space=color_space,
                                           spatial_size=spatial_size, hist_bins=hist_bins,
                                           orient=orient, pix_per_cell=pix_per_cell,
                                           cell_per_block=cell_per_block,
                                           hog_channel=hog_channel, spatial_feat=spatial_feat,
                                           hist_feat=hist_feat, hog_feat=hog_feat)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract HOG features...')
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=TRAIN_TEST_SPLIT, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
          'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    accuracy = round(svc.score(X_test, y_test), 4)*100
    print('Test Accuracy of SVC = ', accuracy)
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

    joblib_save([X_scaler, scaled_X, y], SAVE_DIR + 'dataset_%d.p' % (accuracy * 100))
    joblib_save(svc, SAVE_DIR + 'model_%d.p' % (accuracy * 100))

    return svc, [X_scaler, scaled_X, y], accuracy


def test_sliding_window(image):
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None],
                           xy_window=(128, 128), xy_overlap=(0.5, 0.5))

    window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)
    plt.imshow(window_img)


def heat_and_threshold(image, box_list, threshold=1):
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, threshold)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(image, labels)
    return draw_img, heatmap, labels


def test_draw_labelled_image(image, box_list):
    draw_img, heatmap, labels = heat_and_threshold(image, box_list, threshold=HEAT_THRESHOLD)
    title1 = 'Car Positions (%d)' % (labels[1])
    imcompare(draw_img, heatmap, title1, 'Heat Map', cmap2='hot')


def train_or_load_model(cars, notcars):
    # train or load model
    if MODEL_FILE and DATASET_FILE:
        detector = VehicleDetection(model_file=MODEL_FILE, dataset_file=DATASET_FILE)
        svc = detector.svc
        [X_scaler, scaled_X, y] = [detector.X_scaler, detector.scaled_X, detector.y]
    else:
        svc, [X_scaler, scaled_X, y], _ = train_svc_with_color_hog_hist(cars, notcars)

    return svc, [X_scaler, scaled_X, y]


def test_slide_search_window(filenames, cars, notcars, video=False):

    detector = VehicleDetection(model_file=MODEL_FILE, dataset_file=DATASET_FILE)

    if video:
        # Video Mode
        debug('Processing Video: ', INPUT_VIDEOFILE)
        input_videoclip = VideoFileClip(INPUT_VIDEOFILE)
        output_videofile = OUTPUT_DIR + INPUT_VIDEOFILE[:-4] + '_output.mp4'
        # NOTE: this function expects color images!
        vehicle_tracking_videoclip = input_videoclip.fl_image(detector.sliding_window_search)
        vehicle_tracking_videoclip.write_videofile(output_videofile, audio=False)
    else:
        # Debug & Test Images Mode
        for filename in filenames:
            image = mpimg.imread(filename)
            window_img = detector.sliding_window_search(image)
            # test_draw_labelled_image(image, detector.hot_windows)


def main():
    cars = glob.glob(TRAINING_DIR + VEHICLES_DIR + '*/*.png')
    notcars = glob.glob(TRAINING_DIR + NON_VEHICLES_DIR + '*/*.png')
    num_samples = NUM_SAMPLES
    cars, notcars = cars[:num_samples], notcars[:num_samples]
    # train_svc_with_color_hog_hist(cars, notcars)
    filenames = glob.glob(TEST_IMAGES_DIR + '*')
    test_slide_search_window(filenames, cars, notcars, video=VIDEO_MODE)


if __name__ == '__main__':
    main()
