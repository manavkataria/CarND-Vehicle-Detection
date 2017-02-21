Dear Reviewer, Kindly Note: This README is a work in progress. Sunnyvale, California is facing a power outage and I'm using my mobile hotspot to update this. The video and code are up to date.
-Manav
12:48am PST, 21st Feb 2017

# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Table of Contents
=================
---
This video contains results and illustration of challenges encountered during this project:

[![_youtube_thumb_](https://cloud.githubusercontent.com/assets/2206789/23159936/8714f320-f7d9-11e6-9b4f-9a1e55578246.jpg)](https://youtu.be/TJ0arL7OP3o)


---

## Pipeline
1. Basic Data Exploration
   * Visually scanning the kind of images
   * Plotting Color Space (RGB. HSV, YCrCb) Histograms of Images
   * Validating that both "in class" and "out of class" samples have nearly equal sizes (Balanced Dataset)
1. Feature Extraction from `car` and `notcar` classes in [`extract_features_hog`](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/0263dc41f397ddf25b01f0c338fed675734b8d11/utils.py#L229-L255) and [`single_img_features`](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/0263dc41f397ddf25b01f0c338fed675734b8d11/utils.py#L303-L346)
   * Image Spatial Features as `spatial_features`
   * Image Color Histograms as `hist_features`, and
   * Histogram of Oriented Gradients as `hog_features`
1. Training Car Detector with [`train_or_load_model`](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/0263dc41f397ddf25b01f0c338fed675734b8d11/main.py#L379-L388) using LinearSVC in [`train_svc_with_color_hog_hist`](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/0263dc41f397ddf25b01f0c338fed675734b8d11/main.py#L348-L357)
   * Initial classifier test accuracies was 90% _without HOG_
   * Including HOG, experimentation & careful combination of [*hyperparameters*](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/0263dc41f397ddf25b01f0c338fed675734b8d11/settings.py#L31-L41) the accuracy rose up to 99%
1. `Class` [Vehicle Detection  🚗](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/0263dc41f397ddf25b01f0c338fed675734b8d11/main.py#L81)
    * `__init__` - Initializes Instance Variables
    * [`init`](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/0263dc41f397ddf25b01f0c338fed675734b8d11/main.py#L105-L121) - Feature Extraction and Sliding Window Search
    * [*`memory`*](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/0263dc41f397ddf25b01f0c338fed675734b8d11/main.py#L100-L101) - `RollingStatistics` object with a circular queue for saving `MEMORY_SIZE` number of previous frames. Leverages `Pandas` underneath. Pretty cool stuff! 😎
    * Sliding Window Search Area Highlight
    * [`rolling_sum`](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/0263dc41f397ddf25b01f0c338fed675734b8d11/main.py#L132-L139) - Gets a rolling_sum heatmap from `memory`
    * [`add_to_debugbar`](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/0263dc41f397ddf25b01f0c338fed675734b8d11/main.py#L140-L167) - Insets debug information picture-in-picture or rather video-in-video. Quite professional, you see! 👔
    * [`heat_and_threshold`](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/0263dc41f397ddf25b01f0c338fed675734b8d11/main.py#L169-L197)  - Computes the heatmaps🔥, labels and bounding boxes with [*metrics* for each labelled box](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/0263dc41f397ddf25b01f0c338fed675734b8d11/utils.py#L420-L425)
    * `sliding_window_search` - Sliding window search utilizing [`memory`, `debug`🐛 and `exception` handling 🎇 ](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/0263dc41f397ddf25b01f0c338fed675734b8d11/main.py#L218-L239)   * finally returning an _undistorted_ image
1. [Save as Video.mp4]()

## Pipeline Images

### Image1

# Files
The project was designed to be modular and reusable. The significant independent domains get their own `Class` and an individual file:
  1. `main.py` - Main test runner with `VehicleDetection` Class and test functions like `train_svc_with_color_hog_hist`, `test_sliding_window`, `train_or_load_model`.
  1. `utils.py` - Handy utils like [`imcompare`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/utils.py#L32),  [`warper`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/utils.py#L32), [`debug`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/utils.py#L14-L17) shared across modules
  1. `settings.py` - Settings shared across module
  1. `rolling_statistics.py` - `RollingStatistics` class to compute `moving_average` and `rolling_sum`
  1. `README.md` - description of the development process (this file)

All files contain **detailed comments** to explain how the code works. Refer Udacity Repository [CarND-Advanced-Lane-Lines](https://github.com/udacity/CarND-Advanced-Lane-Lines/) - for calibration images, test images and test videos

## Usage
Repository includes all required files and can be used to rerun vehicle detection & tracking on a given video. Set configuration values in `settings.py` and run the `main.py` python script.
```
$ grep 'INPUT\|OUTPUT' -Hn settings.py
settings.py:9:      INPUT_VIDEOFILE = 'test_video_output.mp4'
settings.py:11:     OUTPUT_DIR = 'output_images/'

$ python main.py
[test_slide_search_window:369] Processing Video:  test_video.mp4
[MoviePy] >>>> Building video output_images/test_video_output.mp4
[MoviePy] Writing video output_images/test_video_output.mp4
 97%|█████████████████████████████████████████████████████████████████▎ | 38/39 [02:01<00:03,  3.05s/it]
[MoviePy] Done.
[MoviePy] >>>> Video ready: output_images/test_video_output.mp4

$ open output_images/test_video_output.mp4
```

# Challenges

## Challenge1
## Challenge2

# Shortcomings & Future Enhancements

**Figure: Example of a frame where the current implementation falls apart**
TODO

## Enhancements for future
1. Optimize the hog feature extraction by caching and reusing hog for different windows and scales
1. Try different Neural Network based vehicle detection approaches like YOLO instead of SVM LinearSVC
1. Consider Weighted Averaging of frames (on Bounding Box size, for example)
1. Consider frame weight expiry by incorporating a decay factor, like half-life (λ)
1. Debug: Add heatmap preview window in upper Debug Bar of Video Output
1. Minor: Write asserts for unittests in `rolling_statistics.py`
1. Use sliders to tune thresholds (Original idea courtesy **[Sagar Bhokre](https://github.com/sagarbhokre)**)
1. Integrate with my [Advanced Lane Finding](https://github.com/manavkataria/carnd-advanced-lane-detection) Implementation

# Acknowledgements & References
* **[Sagar Bhokre](https://github.com/sagarbhokre)** - for project skeleton & constant support
* **[Caleb Kirksey](https://github.com/ckirksey3)** - for motivation and company
* [CarND-Vehicle-Detection](https://github.com/udacity/CarND-Vehicle-Detection) - Udacity Repository test images and test videos
* [Vehicle Training Set](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) - to train the classifier
* [Non Vehicle Training Set](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) - to train the classifier  

========

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.
