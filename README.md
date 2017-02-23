Table of Contents
=================
  1. [Files](#files)
      * [Usage](#usage)
   1. [Pipeline](#pipeline)
   1. [Pipeline In Action](#pipeline-in-action)
      * [Figure: Feature Vector - Spatial Features and HLS Histogram](#figure-feature-vector---spatial-features-and-hls-histogram)
      * [Figure: Feature Vector - HOG and HLS](#figure-feature-vector---hog-and-hls)
      * [Figure: Limited Sliding Window Search (Scale = 1x)](#figure-limited-sliding-window-search-scale--1x)
      * [Figure:  Rolling Sum is Robust to Noise](#figure--rolling-sum-is-robust-to-noise)
      * [Figure: Rolling Sum Picks up where current frame heatmap may not](#figure-rolling-sum-picks-up-where-current-frame-heatmap-may-not)
      * [Figure:  Rolling Sum is averse to cars on other side of the Highway](#figure--rolling-sum-is-averse-to-cars-on-other-side-of-the-highway)
      * [Figure: Rolling Sum smoother than current frame (See Video)](#figure-rolling-sum-smoother-than-current-frame-see-video)
   1. [Challenges](#challenges)
      * [Tuning the Hyperparameters -&gt; Experimentation   Trial &amp; Error](#tuning-the-hyperparameters---experimentation--trial--error)
      * [Debugging Difficulties -&gt; Enhanced Visualizations](#debugging-difficulties---enhanced-visualizations)
   1. [Shortcomings](#shortcomings)
      * [False Positives](#false-positives)
      * [No Detection](#no-detection)
   1. [Future Enhancements](#future-enhancements)
   1. [Acknowledgements &amp; References](#acknowledgements--references)

Vehicle Detection
----

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This video contains results and illustration of challenges encountered during this project:

[![_youtube_thumb_](https://cloud.githubusercontent.com/assets/2206789/23159936/8714f320-f7d9-11e6-9b4f-9a1e55578246.jpg)](https://www.youtube.com/watch?edit=vd&v=w9XQY6f3lfY)


---
# Files
The project was designed to be modular and reusable. The significant independent domains get their own `Class` and an individual file:
  1. [`main.py`](https://github.com/manavkataria/CarND-Vehicle-Detection/blob/master/main.py) - Main test runner with `VehicleDetection` Class and test functions like `train_svc_with_color_hog_hist`, `test_sliding_window`, `train_or_load_model`.
  1. [`utils.py`](https://github.com/manavkataria/CarND-Vehicle-Detection/blob/master/utils.py) - Handy utils like [`imcompare`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/utils.py#L32),  [`warper`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/utils.py#L32), [`debug`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/utils.py#L14-L17) shared across modules
  1. [`settings.py`](https://github.com/manavkataria/CarND-Vehicle-Detection/blob/master/settings.py) - Hyperparameters and Settings shared across module
  1. [`rolling_statistics.py`](https://github.com/manavkataria/CarND-Vehicle-Detection/blob/master/rolling_statistics.py) - `RollingStatistics` class to compute `moving_average` and `rolling_sum`
  1. [`README.md`](https://github.com/manavkataria/CarND-Vehicle-Detection/blob/master/README.md) - description of the development process (this file)

The [**Pipeline**](#pipeline) section below has a high level description of the pipeline and pointers to implementation. The code is fairly readable and contains **detailed comments** that explain their working.

## Usage
Set Hyperparameters and configurations in [`settings.py`](https://github.com/manavkataria/CarND-Vehicle-Detection/blob/master/settings.py)  and run the [`main.py`](https://github.com/manavkataria/CarND-Vehicle-Detection/blob/master/main.py) python script as shown below. Repository includes all required files and can be used to rerun vehicle detection & tracking on a given video. Refer [References](#acknowledgements--references) below for training dataset.

```
$ grep 'INPUT\|OUTPUT' -Hn settings.py
settings.py:9:      INPUT_VIDEOFILE = 'test_video_output.mp4'
settings.py:11:     OUTPUT_DIR = 'output_images/'

$ python main.py
[test_slide_search_window:369] Processing Video:  test_video.mp4
[MoviePy] >>>> Building video output_images/test_video_output.mp4
[MoviePy] Writing video output_images/test_video_output.mp4
 100%|██████████████████████████████████████████████████████████████████| 39/39 [02:01<00:03,  3.05s/it]
[MoviePy] Done.
[MoviePy] >>>> Video ready: output_images/test_video_output.mp4

$ open output_images/test_video_output.mp4
```

# Pipeline
1. Basic Data Exploration
   * Visually scanned images from each class and selected a `Car` and a `Road` as a canonical "Vehicle" "Non-Vehicle" class samples for data exploration. Figures in section [Pipeline In Action](#pipeline-in-action)
   * Explored Color Histograms and HOG Features by comparing how well it separates above classes. Figures in section [Pipeline In Action](#pipeline-in-action)
   * Validating that both "in class" and "out of class" samples have nearly equal sizes ~8.5K samples. Thus we have a balanced dataset and training wouldn't be too biased
1. Feature Extraction - from "Vehicle" and "Non-Vehicle" classes in [`extract_features_hog`](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/0263dc41f397ddf25b01f0c338fed675734b8d11/utils.py#L229-L255) and [`single_img_features`](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/0263dc41f397ddf25b01f0c338fed675734b8d11/utils.py#L303-L346)
   * Image Spatial Features as `spatial_features`
   * Image Color Histograms as `hist_features`, and
   * Histogram of Oriented Gradients as `hog_features`
1. Training Car Detector - with [`train_or_load_model`](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/0263dc41f397ddf25b01f0c338fed675734b8d11/main.py#L379-L388) using `LinearSVC` in [`train_svc_with_color_hog_hist`](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/0263dc41f397ddf25b01f0c338fed675734b8d11/main.py#L348-L357)
   * Initial classifier test accuracies was 90% _without HOG_
   * Including HOG, experimentation & careful combination of **hyperparameters** [`settings.py:L31-L41`](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/0263dc41f397ddf25b01f0c338fed675734b8d11/settings.py#L31-L41) the accuracy rose up to 99%
   * [Save Model & Features](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/967e8210f33396159d991c248c74d68d4e365a3e/main.py#L365-L367) - Using `joblib` not `pickle`. `joblib` handles large numpy arrays a lot more efficiently
1. [**Vehicle Detection**](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/0263dc41f397ddf25b01f0c338fed675734b8d11/main.py#L81) - `Class` that utilizes region limited sliding window search, heatmap, thresholding, labelling and rolling sum to eventually filter the vehicles.
    * [`__init__`](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/967e8210f33396159d991c248c74d68d4e365a3e/main.py#L83-L104) - Initializes Instance Variables like Feature Extraction and Sliding Window Search
    * [**Memory**](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/0263dc41f397ddf25b01f0c338fed675734b8d11/main.py#L100-L101) - [`RollingStatistics`](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/967e8210f33396159d991c248c74d68d4e365a3e/rolling_statistics.py#L10-L17) object with a circular queue for saving `MEMORY_SIZE` number of previous frames. Leverages `Pandas` underneath. _Pretty cool stuff!_ 😎
    * [`sliding_window_search`](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/967e8210f33396159d991c248c74d68d4e365a3e/main.py#L199-L229) - Search sliding window
        * Region limited search to [`[400, 656]`](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/a580530c641a04f6fbaf986eb3a02db12808c18a/settings.py#L47) for optimization.
        * Utilizes [memory, `🐛`debug, & exception`🎇` handling ](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/0263dc41f397ddf25b01f0c338fed675734b8d11/main.py#L218-L239)
    * [`update_overlay`](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/967e8210f33396159d991c248c74d68d4e365a3e/main.py#L123-L131) - Sliding Window Search Area Highlighting with
        * `identifier`, and
        * `dimensions` of bounding_box
    * [`heat_and_threshold`](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/0263dc41f397ddf25b01f0c338fed675734b8d11/main.py#L169-L197)  - Computes the heatmaps🔥, labels and bounding boxes with [*metrics*](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/0263dc41f397ddf25b01f0c338fed675734b8d11/utils.py#L420-L425) for each labelled box
    * [`rolling_sum`](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/0263dc41f397ddf25b01f0c338fed675734b8d11/main.py#L132-L139) - Gets a rolling_sum heatmap from `memory`
    * [`add_to_debugbar`](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/0263dc41f397ddf25b01f0c338fed675734b8d11/main.py#L140-L167) - Insets debug information picture-in-picture or rather video-in-video. _Quite professional, you see!_ 👔

1. [Save as Video.mp4](http://github.com/manavkataria/CarND-Vehicle-Detection/blob/967e8210f33396159d991c248c74d68d4e365a3e/main.py#L395-L402)

# Pipeline In Action

## Figure: Feature Vector - Spatial Features and HLS Histogram
![Feature Vector - Spatial Features and HLS Histogram](https://cloud.githubusercontent.com/assets/2206789/23207512/5fc272ac-f8a6-11e6-9722-0eafd5bafaff.jpg)

## Figure: Feature Vector - HOG and HLS
![Feature Vector - HOG and HLS](https://cloud.githubusercontent.com/assets/2206789/23210645/6633cdfe-f8b3-11e6-987b-ec5061749a7d.jpg)

## Figure: Limited Sliding Window Search (Scale = 1x)
![Limited Sliding Window Search](https://cloud.githubusercontent.com/assets/2206789/23210616/40d9afd8-f8b3-11e6-9e38-4740b496acf9.jpg)

## Figure:  Rolling Sum is Robust to Noise
![rolling sum s robustness to noise](https://cloud.githubusercontent.com/assets/2206789/23163153/286d3eba-f7e6-11e6-93a9-992a2d5e217d.jpg)

## Figure: Rolling Sum Picks up where current frame heatmap may not
![rolling sum picks up where current frame heatmap does not](https://cloud.githubusercontent.com/assets/2206789/23163252/7be3ba10-f7e6-11e6-8c9a-d4c9a82f1789.jpg)

## Figure:  Rolling Sum is averse to cars on other side of the Highway
![rolling sum robust to cars on other lanes](https://cloud.githubusercontent.com/assets/2206789/23163152/28688352-f7e6-11e6-83f7-b34b1a3d4b4f.jpg)

## Figure: Rolling Sum smoother than current frame (See Video)
[![_youtube_thumb_](https://cloud.githubusercontent.com/assets/2206789/23159936/8714f320-f7d9-11e6-9b4f-9a1e55578246.jpg)](https://www.youtube.com/watch?edit=vd&v=w9XQY6f3lfY)

# Challenges
## Tuning the Hyperparameters -> Experimentation + Trial & Error
It was non-trivial to choose the Hyperparameters. Its primarily been a trial-and-error process. [Mohan Karthik's blogpost](https://medium.com/@mohankarthik/feature-extraction-for-vehicle-detection-using-hog-d99354a84d10#.cq5rjdvt3) was precise and served as a good general direction for Color histogram and HOG params. I still experimented with them on my own to determine what works best for me. As mentioned earlier, just the spatial features and channel histograms yielded classifier test accuracy at 90%. I chose `HLS` color space as it (or `HSV`) yielded great results for lane keeping. By some argument `HLS` is more intuitive than `HSV` color space. I added HOG features to bump up accuracy to 99%.

## Debugging Difficulties -> Enhanced Visualizations
It wasn't easy to visualize why the system didn't work for a given frame of video. Using a rolling sum made things even harder. Hence I decided to a few elements to make my life easy:
1. Add insets to preview `Current Frame Heatmap` and `Rolling Sum Heatmap`
2. Color coded respective detections differently with **_thin red_** from `Current Frame Heatmap` and **THICK GREEN** from the `Rolling Sum Heatmap`.
3. Rheir respective thresholds are presented in the status screen as `HeatTh: RollSum * CurFr` , 19 & 1 respectively.
4. Rolling window buffer size is also displayed as `Memory`
1. Current frame id is displayed on the left as `1046`
1. Accuracy of the classifier used is also presented as `Accuracy`
1. Bounding box ids and sizes are also displayed as `id | width x height` around each box; This will be useful in considering a weighted average (see **Enhancements** below)

# Shortcomings
## False Positives
**Figure: Example of a frame where the current implementation falls apart**
![false positives](https://cloud.githubusercontent.com/assets/2206789/23163233/6a001e2e-f7e6-11e6-82ad-059c20da25b9.jpg)

## No Detection
**Figure: Example of a frame where the current implementation does not detect the car**
![rolling sum does not detect](https://cloud.githubusercontent.com/assets/2206789/23163372/0b418962-f7e7-11e6-87a7-16874cc01844.jpg)

# Future Enhancements
1. Optimize the hog feature extraction by caching and reusing hog for different windows and scales
1. Try different Neural Network based vehicle detection approaches like YOLO instead of SVM LinearSVC
1. Ideas to Reduce False Positives:
    1. Consider Weighted Averaging of frames (on Bounding Box size, for example); Penalize small boxes, sustain large ones
    1. Consider frame weight expiry by incorporating a decay factor, like half-life (λ)
    1. Consider using [Optical Flow](http://docs.opencv.org/3.2.0/d7/d8b/tutorial_py_lucas_kanade.html) techniques
1. Debug: Add heatmap preview window in upper Debug Bar of Video Output
1. Minor: Write asserts for unittests in `rolling_statistics.py`
1. Use sliders to tune thresholds (Original idea courtesy **[Sagar Bhokre](https://github.com/sagarbhokre)**)
1. Integrate with [Advanced Lane Finding](https://github.com/manavkataria/carnd-advanced-lane-detection) Implementation

# Acknowledgements & References
* **[Sagar Bhokre](https://github.com/sagarbhokre)** - For constant support
* **[Caleb Kirksey](https://github.com/ckirksey3)** - For motivation and company
* **[Mohan Karthik's Blogpost](https://medium.com/@mohankarthik/feature-extraction-for-vehicle-detection-using-hog-d99354a84d10#.cq5rjdvt3)** - Excellent analysis on feature extraction and hyperparameter selection
* [GridSpace Reference: Subplot vs GridSpace](http://www.python-course.eu/matplotlib_multiple_figures.php) - Subplotting made sense
* [CarND-Vehicle-Detection](https://github.com/udacity/CarND-Vehicle-Detection) - Udacity Repository test images and test videos
* [Vehicle Training Dataset](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) - Dataset to train the classifier
* [Non-Vehicle Training Dataset](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) - Dataset to train the classifier  
