#!/usr/bin/env ipython
import glob
import matplotlib
import time
import cv2
import numpy as np

import lesson_functions

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label

from rolling_statistics import RollingStatistics

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
                      DATASET_X_SCALER_FILE,
                      DATASET_SCALED_X_Y_FILE,
                      TRAIN_TEST_SPLIT,
                      SPATIAL_SIZE,
                      HIST_BINS,
                      SPATIAL_FEAT,
                      HIST_FEAT,
                      HOG_FEAT,
                      XY_WINDOW,
                      XY_OVERLAP,
                      VIDEO_MODE,
                      IMAGE_HEIGHT,
                      IMAGE_WIDTH,
                      IMAGE_DEPTH,
                      IMAGE_DTYPE,
                      ACCURACY,
                      MEMORY_SIZE,
                      TRAIN,
                      DEBUG,
                      HEATMAP_METRICS,
                      ROLLING_SUM_HEAT_THRESHOLD,
                      CURRENT_FRAME_HEAT_THRESHOLD)
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
                   debug,
                   put_text,
                   weighted_img)

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

    def __init__(self):
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

        self.count = 0  # frame counter

        # Compute Moving Average
        self.columns = ['heat']
        self.memory = RollingStatistics(self.columns, size=MEMORY_SIZE)

        # Top Overlay
        self.overlay = None

    def train_or_load_model(self, cars, notcars):
        if TRAIN:
            # Train Model
            debug('Training LinearSVC Model ...')
            svc, [X_scaler, scaled_X, y], accuracy = train_svc_with_color_hog_hist(cars, notcars)
            self.svc, [self.X_scaler, self.scaled_X, self.y], self.accuracy = svc, [X_scaler, scaled_X, y], accuracy

            # Persist to Disk
            joblib_save(X_scaler, SAVE_DIR + 'dataset_X_scaler_%d.p' % (accuracy * 100))
            joblib_save([scaled_X, y], SAVE_DIR + 'dataset_scaled_x_y_%d.p' % (accuracy * 100))
            joblib_save(svc, SAVE_DIR + 'model_%d.p' % (accuracy * 100))

            return svc, X_scaler, scaled_X, y, accuracy
        else:
            # Load Model and Dataset
            debug('Loading LinearSVC Model %s ...' % MODEL_FILE)
            if MODEL_FILE:
                self.svc = joblib_load(MODEL_FILE)
            else:
                self.svc = None

            debug('Loading Dataset File: %s ...' % DATASET_X_SCALER_FILE)
            if DATASET_X_SCALER_FILE:
                self.X_scaler = joblib_load(DATASET_X_SCALER_FILE)
            else:
                self.X_scaler = None

            debug('Loading Dataset File: %s ...' % DATASET_SCALED_X_Y_FILE)
            if DATASET_SCALED_X_Y_FILE:
                self.scaled_X, self.y = joblib_load(DATASET_SCALED_X_Y_FILE)
            else:
                self.scaled_X, self.y = None

            self.accuracy = ACCURACY if ACCURACY else None

            debug('Done!')
            return self.svc, self.X_scaler, self.scaled_X, self.y, self.accuracy

    def update_overlay(self, image=None):
        if image is None:
            self.overlay = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH), dtype=IMAGE_DTYPE)
        else:
            self.overlay = np.copy(image)
            # Darken All Except Window Search Area
            self.overlay[:Y_START_STOP[0], :, :] = 0
            self.overlay[Y_START_STOP[1]:, :, :] = 0

    def rolling_sum(self, data):
        data_dict = {}
        for i, value in enumerate(data):
            key = self.columns[i]
            data_dict[key] = value

        return self.memory.rolling_sum(data_dict)

    def add_to_debugbar(self, base, inset, msg, position='right'):
        # ysize = 400         # Size of Debug Bar in px
        text_ysize = 42     # Height of Text with Padding
        right_padding = 30  # Distance from Right Edge
        hz_centering_msg = 85
        additional_text_offset = 5

        # Resize
        inset = cv2.resize(inset, None, fx=0.4, fy=0.4)

        # Position
        if position == 'left':
            (x, y) = (right_padding*2), (text_ysize + right_padding)
        else:
            (x, y) = (base.shape[1] - inset.shape[1] - right_padding*2), (text_ysize + right_padding)

        # Embed
        if inset.ndim == 2:
            # Adaptive Rescale and Convert to Color
            inset = cv2.cvtColor((inset/inset.max()).astype('float32'), cv2.COLOR_GRAY2RGB)
            # inset = cv2.cvtColor(((inset)*254./(inset.max())).astype('uint8'), cv2.COLOR_GRAY2RGB)
        base[y:y+inset.shape[0], x:x+inset.shape[1], :] = inset

        # Title Text
        (xpos, ypos) = (x + hz_centering_msg), (text_ysize + additional_text_offset)
        put_text(base, msg, xpos, ypos)

        return base

    def heat_and_threshold(self, image, box_list, rolling_threshold=1, current_threshold=1):
        heat = np.zeros_like(image[:,:,0]).astype(np.float)

        # Add heat to each box in box list
        raw_heat = add_heat(heat, box_list)

        # Smoothen out heated windows based on time-averaging
        avg_heat = self.rolling_sum([heat])['heat']

        # Apply threshold to help remove false positives
        raw_heat = apply_threshold(raw_heat, CURRENT_FRAME_HEAT_THRESHOLD)  # SETTINGS.CURRENT_FRAME_HEAT_THRESHOLD
        avg_heat = apply_threshold(avg_heat, ROLLING_SUM_HEAT_THRESHOLD)    # SETTINGS.ROLLING_SUM_HEAT_THRESHOLD

        # Visualize the heatmap when displaying
        # TODO: if VideoMode; else (255)
        raw_heatmap = np.clip(raw_heat, 0, 255)
        avg_heatmap = np.clip(avg_heat, 0, 255)

        image = self.add_to_debugbar(image, avg_heatmap, 'Rolling Sum Heatmap', position='right')
        image = self.add_to_debugbar(image, raw_heatmap, 'Current Frm Heatmap', position='left')

        # Find final boxes from heatmap using label function
        raw_labels = label(raw_heatmap)
        avg_labels = label(avg_heatmap)

        # Overlap Raw with Avg
        draw_img = draw_labeled_bboxes(image, raw_labels, color=(1, 0, 0), thickness=2, meta=False)  # red
        draw_img = draw_labeled_bboxes(draw_img, avg_labels, meta=HEATMAP_METRICS)
        return draw_img, avg_heatmap, avg_labels

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
            msg = '%04d | Memory: %dFr | HeatTh: RollSum %d * CurFr %d | Accuracy: %0.1f%%' % (self.count, MEMORY_SIZE, ROLLING_SUM_HEAT_THRESHOLD, CURRENT_FRAME_HEAT_THRESHOLD, ACCURACY/100)

            self.update_overlay(draw_image)
            draw_image = weighted_img(draw_image, self.overlay)
            put_text(draw_image, msg)

            heat_thresholded_image, thresholded_heatmap, labels = self.heat_and_threshold(draw_image, self.hot_windows, rolling_threshold=ROLLING_SUM_HEAT_THRESHOLD, current_threshold=CURRENT_FRAME_HEAT_THRESHOLD)
            self.save = heat_thresholded_image

        except Exception as e:
            mpimg.imsave('hard/%d.jpg' % self.count, image)
            debug('Error(%s): Issue at Frame %d' % (str(e), self.count))
            if DEBUG: import ipdb; ipdb.set_trace()
            if self.save:
                heat_thresholded_image = self.save

        finally:
            self.count += 1

        if VIDEO_MODE:
            # Scale Back to Format acceptable by moviepy
            heat_thresholded_image = heat_thresholded_image.astype(np.float32) * 255
        else:
            debug('%0.1f seconds/frame. #%d/%d hot-windows/windows/frame' % (end-start,
                                                                             len(hot_windows), len(windows)))
            title1 = 'Car Positions (#Detections: %d)' % (labels[1])
            title2 = 'Thresholded Heat Map (Max: %d)' % int(np.max(thresholded_heatmap))
            imcompare(heat_thresholded_image, thresholded_heatmap, title1, title2, cmap2='hot')

        return heat_thresholded_image

        def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
            """
            Define a single function that can extract features using hog sub-sampling and make predictions
            """

            draw_img = np.copy(img)
            img = img.astype(np.float32)/255

            img_tosearch = img[ystart:ystop,:,:]
            ctrans_tosearch = lesson_functions.convert_color(img_tosearch, conv='RGB2YCrCb')
            if scale != 1:
                imshape = ctrans_tosearch.shape
                ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

            ch1 = ctrans_tosearch[:,:,0]
            ch2 = ctrans_tosearch[:,:,1]
            ch3 = ctrans_tosearch[:,:,2]

            # Define blocks and steps as above
            nxblocks = (ch1.shape[1] // pix_per_cell)-1
            nyblocks = (ch1.shape[0] // pix_per_cell)-1
            nfeat_per_block = orient*cell_per_block**2
            # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
            window = 64
            nblocks_per_window = (window // pix_per_cell)-1
            cells_per_step = 2  # Instead of overlap, define how many cells to step
            nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
            nysteps = (nyblocks - nblocks_per_window) // cells_per_step

            # Compute individual channel HOG features for the entire image
            hog1 = lesson_functions.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
            hog2 = lesson_functions.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
            hog3 = lesson_functions.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

            for xb in range(nxsteps):
                for yb in range(nysteps):
                    ypos = yb*cells_per_step
                    xpos = xb*cells_per_step
                    # Extract HOG for this patch
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                    xleft = xpos*pix_per_cell
                    ytop = ypos*pix_per_cell

                    # Extract the image patch
                    subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                    # Get color features
                    spatial_features = lesson_functions.bin_spatial(subimg, size=spatial_size)
                    hist_features = color_hist(subimg, nbins=hist_bins)

                    # Scale features and make a prediction
                    test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                    # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                    test_prediction = svc.predict(test_features)

                    if test_prediction == 1:
                        xbox_left = np.int(xleft*scale)
                        ytop_draw = np.int(ytop*scale)
                        win_draw = np.int(window*scale)
                        cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)

            return draw_img


def test_find_cars(cars, notcars):
    img = mpimg.imread(TEST_IMAGES_DIR + 'test1.jpg')

    ystart, ystop = Y_START_STOP
    scale = 1.5

    detector = VehicleDetection()
    detector.train_or_load_model(cars, notcars)
    out_img = detector.find_cars(img, ystart, ystop, scale, detector.svc, detector.X_scaler, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, SPATIAL_SIZE, HIST_BINS)

    plt.imshow(out_img)


def test_svc_color_hist(cars, notcars):
    # TODO(Manav): Pending retest after recent update
    spatial = 32
    histbin = 32

    car_features = extract_features_hog(cars, color_space='HSV', spatial_size=(spatial, spatial),
                                        hist_bins=histbin, hist_range=(0, 256), hog_feat=False)
    notcar_features = extract_features_hog(notcars, color_space='HSV', spatial_size=(spatial, spatial),
                                           hist_bins=histbin, hist_range=(0, 256), hog_feat=False)

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

    debug('Using spatial binning of:',spatial,
          'and', histbin,'histogram bins')
    debug('Feature vector length:', len(X_train[0]))

    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    debug(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    debug('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    debug('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    debug('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    debug(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')


def train_svc_with_color_hog_hist(cars, notcars):
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
    debug(round(t2-t, 2), 'Seconds to extract HOG features...')
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

    debug('Using:', COLORSPACE,'colorspace', SPATIAL_SIZE, 'Spatial Size', SPATIAL_SIZE)
    debug('Using:',orient,'orientations',pix_per_cell,
          'pixels per cell and', cell_per_block,'cells per block',
          HOG_CHANNEL, 'Channel(s)')
    debug('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    debug(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    accuracy = round(svc.score(X_test, y_test), 4)*100
    debug('Test Accuracy of SVC = ', accuracy)
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    debug('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    debug('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    debug(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

    return svc, [X_scaler, scaled_X, y], accuracy


def test_sliding_window(image):
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=Y_START_STOP,
                           xy_window=(128, 128), xy_overlap=(0.5, 0.5))

    window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)
    plt.imshow(window_img)


def test_slide_search_window(filenames, cars, notcars, video=False):

    detector = VehicleDetection()
    detector.train_or_load_model(cars, notcars)

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
            detector.sliding_window_search(image)


def main():
    cars = glob.glob(TRAINING_DIR + VEHICLES_DIR + '*/*.png')[:NUM_SAMPLES]
    notcars = glob.glob(TRAINING_DIR + NON_VEHICLES_DIR + '*/*.png')[:NUM_SAMPLES]
    filenames = glob.glob(TEST_IMAGES_DIR + '*')[:NUM_SAMPLES]
    test_slide_search_window(filenames, cars, notcars, video=VIDEO_MODE)


if __name__ == '__main__':
    main()
