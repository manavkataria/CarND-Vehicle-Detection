Dear Reviewer, Kindly Note: This README is a work in progress. Sunnyvale, California is facing a power outage and I'm using my mobile hotspot to update this. The video and code are up to date.
-Manav
12:48am PST, 21st Feb 2017

# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Table of Contents
=================
---
This video contains results and illustration of challenges encountered during this project:

[![youtube thumb](https://cloud.githubusercontent.com/assets/2206789/22967459/670853b4-f31b-11e6-9eef-1493e728e7f9.jpg)](https://youtu.be/6lf099n2LkI)

---

## Pipeline
1. Basic Data Exploration
   * Visually scanning the kind of images
   * Plotting Color Space (RGB. HSV, YCrCb) Histograms of Images
   * Validating that both "in class" and "out of class" samples have nearly equal sizes (Balanced Dataset)
1. Feature Extraction from `car` and `notcar` classes in `extract_features_hog`
   * `spatial_features`
   * `hist_features`, and
   * `hog_features`
1. Training Car Detector
2. Filters using [`filtering_pipeline`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/main.py#L49-L86)
   * RGB to HSL
   * Careful Combination of the above
   * Guassian Blur to eliminate noise `K=31`
3. Lane Detection [`pipeline`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/lanes.py#L294)
   * [`overlay_and_unwarp`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/lanes.py#L238)
       * `fill_lane_polys`
   * [`put_metrics_on_image`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/lanes.py#L285)
   * finally returning an _undistorted_ image
4. [Save as Video.mp4  ](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/main.py#L118)

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


========


The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

Some example images for testing your pipeline on single frames are located in the `test_images` folder.  To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include them in your writeup for the project by describing what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

**As an optional challenge** Once you have a working pipeline for vehicle detection, add in your lane-finding algorithm from the last project to do simultaneous lane-finding and vehicle detection!

**If you're feeling ambitious** (also totally optional though), don't stop there!  We encourage you to go out and take video of your own, and show us how you would implement this project on a new video!
