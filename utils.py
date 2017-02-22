import matplotlib
import inspect
import cv2
import numpy as np
import pickle
import joblib

from skimage.feature import hog
from matplotlib.gridspec import GridSpec

from settings import DEBUG, DISPLAY, DEBUG_CHANNEL_HIST

matplotlib.use('TkAgg')  # MacOSX Compatibility
matplotlib.interactive(True)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def debug(*args):
    frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
    if DEBUG:
        print('[%s:%d]' % (function_name, line_number), *args)


def display(image, msg='Image', cmap=None):
    if not DISPLAY: return

    if image.ndim == 2:
        cmap = 'gray'

    plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.title(msg, fontsize=30)
    plt.show(block=True)


def imcompare(image1, image2, msg1='Image1', msg2='Image2', cmap1=None, cmap2=None, block=True):
    if DISPLAY is False: return

    if cmap1 is None and image1.ndim == 2:
        cmap1 = 'gray'
    if cmap2 is None and image2.ndim == 2:
        cmap2 = 'gray'

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    ax1.imshow(image1, cmap=cmap1)
    ax1.set_title(msg1, fontsize=30)
    ax2.imshow(image2, cmap=cmap2)
    ax2.set_title(msg2, fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # title = f.suptitle(msg1)
    f.tight_layout()
    # title.set_y(0.75)
    plt.show(block=block)


def pkl_save(py_object, filename):
    with open(filename, 'wb') as f:
        pickle.dump(py_object, f)


def pkl_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def joblib_save(py_object, filename):
    with open(filename, 'wb') as f:
        joblib.dump(py_object, f)


def joblib_load(filename):
    with open(filename, 'rb') as f:
        return joblib.load(f)

# --- Project 4: Advanced Lane Lines Detection ---

def warper(img, src, dst, flip=True):
    # Compute and apply perpective transform
    if flip:
        # Resultant image (h,w) = (w,h) of input `img`
        img_size = (img.shape[0], img.shape[1])
        w, h = img_size
        w_padding, h_padding = w*0.0, h*0.0

        dst = np.array([[0+w_padding, 0+h_padding],
                        [w-w_padding, 0+h_padding],
                        [w-w_padding, h-h_padding],
                        [0+w_padding, h-h_padding]], np.float32)
    else:
        # Resultant image keeps the (h,w) of input `img`
        img_size = (img.shape[1], img.shape[0])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    # warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M, Minv


def dstack(binary1, binary2):
    """
        Stack each channel to view their individual contributions in green and blue respectively
        This returns a stack of the two binary images, whose components you can see as different colors
    """
    color_binary = np.dstack((np.zeros_like(binary1), binary1, binary2))
    return color_binary


def hist(img):
    color = ('r', 'g', 'b')
    plt.figure()
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        # cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
        plt.plot(histr, color=col)
        plt.xlim([-1, 3])
    plt.show()


def put_text(image, msg, x=50, y=695, size=1, color=(0.8, 0.8, 0.8), thickness=2):
    """
    Default (x,y) Values are for a Status Bar at the bottom of the image
    """
    cv2.putText(image, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size,
                color, thickness, cv2.LINE_AA)


def weighted_img(base_img, overlaid_img, α=0.3, β=0.7, λ=0.):
    """
    `base_img`      Minor Presence (30%)
    `overlaid_img`  Heavy Presence (70%)

    The result image is computed as follows:
        result = base_img * α + overlaid_img * β + λ

    NOTE: overlaid_img and base_img must be the same shape!
    """
    assert (α + β == 1)
    return cv2.addWeighted(base_img, α, overlaid_img, β, λ)

# --- Project 5: Vehicle Detection & Tracking ---


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    image = np.copy(img)
    for (x1, y1), (x2, y2) in bboxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thick)
    return image

def plot_color_hist(range, hist_ch1, hist_ch2, hist_ch3, title):
    fig = plt.figure()
    gs = GridSpec(5,2)
    ax002 = fig.add_subplot(gs[0:2, 0:2])
    ax2 = fig.add_subplot(gs[2, :])
    ax3 = fig.add_subplot(gs[3, :])
    ax4 = fig.add_subplot(gs[4, :])

    ax002.imshow(img)
    ax002.set_title(title)

    ax2.bar(range, hist_ch1)
    ax2.set_title('Features: Color Histogram H Channel')

    ax3.bar(range, hist_ch2)
    ax3.set_title('Features: Color Histogram L Channel')

    ax4.bar(range, hist_ch3)
    ax4.set_title('Features: Color Histogram S Channel')

    fig.tight_layout()
    fig.show()
	# import ipdb; ipdb.set_trace()


def color_hist(img, title=None, nbins=32, bins_range=(0, 1)):
    """
    Compute the histogram of the RGB channels separately
    Concatenate the histograms into a single feature vector
    Return the feature vector
    """
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:,:,0], bins=32, range=(0, 1))
    ghist = np.histogram(img[:,:,1], bins=32, range=(0, 1))
    bhist = np.histogram(img[:,:,2], bins=32, range=(0, 1))
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector

    if DEBUG_CHANNEL_HIST:
        plot_color_hist(range(32), rhist[0], ghist[0], bhist[0], title)
    # return rhist, ghist, bhist, bin_centers, hist_features
    return hist_features


def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel()
    # Return the feature vector
    return features


def data_look(car_list, notcar_list):
    """
    Define a function to return some characteristics of the dataset
    """
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis is True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                       visualise=False, feature_vector=feature_vec)
        return features

def plot_features(image1, image2, features):
    from matplotlib.gridspec import GridSpec
    fig = plt.figure()
    gs = GridSpec(4,4)
    ax002 = fig.add_subplot(gs[0:2, 0:2])
    ax024 = fig.add_subplot(gs[0:2, 2:4])
    ax1 = fig.add_subplot(gs[2, :])
    ax2 = fig.add_subplot(gs[3, :])

    ax002.imshow(image1)
    ax002.set_title('Car')

    ax024.imshow(image2)
    ax024.set_title('Non Car')

    ax1.plot(features[0][1])
    ax1.set_title('Features: Color Histogram of Car')

    ax2.plot(features[1][1])
    ax2.set_title('Features: Color Histogram of NonCar')

    fig.show()
    # import ipdb; ipdb.set_trace()


def extract_features_hog(filenames, color_space='RGB', spatial_size=(32, 32), hist_bins=32,
                         orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                         spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Define a function to extract features from a list of images
    Have this function call bin_spatial() and color_hist()
    """

    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for filename in filenames:
        # Read in each one by one
        image = mpimg.imread(filename)
        single_features = single_img_features(image, color_space=color_space,
                                              spatial_size=spatial_size, hist_bins=hist_bins,
                                              orient=orient, pix_per_cell=pix_per_cell,
                                              cell_per_block=cell_per_block,
                                              hog_channel=hog_channel, spatial_feat=spatial_feat,
                                              hist_feat=hist_feat, hog_feat=hog_feat)

        spatial_features, hist_features, hog_features = single_features

        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    # Return list of feature vectors
    return features


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
    Define a function that takes an image,
    start and stop positions in both x and y,
    window size (x and y dimensions),
    and overlap fraction (for both x and y)
    """
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


def test_plot_hog(image, hog, cmap=None):
    fig = plt.figure()
    gs = GridSpec(3, 2)

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])

    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])

    ax20 = fig.add_subplot(gs[2, 0])
    ax21 = fig.add_subplot(gs[2, 1])

    ax00.imshow(image[:,:,0], cmap=cmap)
    ax00.set_title('H-Channel')
    ax01.imshow(hog[0][1], cmap=cmap)
    ax01.set_title('HOG Features: H-Channel')

    ax10.imshow(image[:,:,1], cmap=cmap)
    ax10.set_title('L-Channel')
    ax11.imshow(hog[1][1], cmap=cmap)
    ax11.set_title('HOG Features: L-Channel')

    ax20.imshow(image[:,:,2], cmap=cmap)
    ax20.set_title('S-Channel')
    ax21.imshow(hog[2][1], cmap=cmap)
    ax21.set_title('HOG Features: S-Channel')

    fig.tight_layout()
    fig.show()
    # import ipdb; ipdb.set_trace()



def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Define a function to extract features from a single image window
    This function is very similar to extract_features()
    just for a single image rather than list of images
    """
    # 1) Define an empty list to receive features
    spatial_features, hist_features, hog_features = [],[], []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat is True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
    # 4) Compute histogram features if flag is set
    if hist_feat is True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
    # 5) Compute HOG features if flag is set
    if hog_feat is True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)

    # test_plot_hog(feature_image, hog_features)  # Needs hog_features.append and vis=True 
    return spatial_features, hist_features, hog_features


def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    """
    Define a function you will pass an image
    and the list of windows to be searched (output of slide_windows())
    """
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        spatial_features, hist_features, hog_features = features
        features = (np.concatenate((spatial_features, hist_features, hog_features)))
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # TODO: rename to threshold_heatmap()
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(image, labels, color=(0,1,0), thickness=6, meta=True):
    img = np.copy(image)
    # Iterate through all detected cars
    offset = 20
    charsize = 30 # px
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, thickness)
        size = ((bbox[1][1]-bbox[0][1]), (bbox[1][0]-bbox[0][0]))
        if meta:
            msg = '%d | %02dx%02dpx' % (car_number, size[0], size[1])
            msgpx = charsize*len(msg)
            put_text(img, msg, bbox[0][0], bbox[0][1]-offset,
                     size=0.5, color=color, thickness=2)
    # Return the image
    return img


def test_color_hist(img, title=None, color_space='HLS'):

    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    color_hist(img, title=title)
