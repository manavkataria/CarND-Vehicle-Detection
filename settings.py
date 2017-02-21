# settings.py

# Debug
DEBUG = True
DISPLAY = True
NUM_SAMPLES = None  # Trim the sample space for efficient debugging
VIDEO_MODE = True   # Enabled/Disable Video Mode/Test Image Mode

# Image Params
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
IMAGE_DEPTH = 3
IMAGE_DTYPE = 'float32'
# Video Input
# INPUT_VIDEOFILE = 'project_video.mp4'
INPUT_VIDEOFILE = 'test_video.mp4'
OUTPUT_DIR = 'output_images/'

# Save
SAVE_DIR = 'save/'

# Test
TEST_IMAGES_DIR = 'test_images/'

# Training Directory
TRAINING_DIR = 'training/'              # 17760 Samples
VEHICLES_DIR = 'vehicles/'              # 08792 Samples
NON_VEHICLES_DIR = 'non-vehicles/'      # 08968 Samples


# Feature Extraction
COLORSPACE = 'HLS'  # Can be RGB*9717, HSV, LUV, HLS*9916, YUV, YCrCb*9899
ORIENT = 9
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2
HOG_CHANNEL = 'ALL'  # Can be 0, 1, 2, or "ALL"
SPATIAL_SIZE = (32, 32)  # Spatial binning dimensions
HIST_BINS = 32    # Number of histogram bins
SPATIAL_FEAT = True  # Spatial features on or off
HIST_FEAT = True  # Histogram features on or off
HOG_FEAT = True  # HOG features on or off

# SVC Model Training
TRAIN_TEST_SPLIT = 0.2

# Sliding Window
Y_START_STOP = [400, 656]  # Min and max in y to search in slide_window()
XY_WINDOW = (96, 96)
XY_OVERLAP = (0.70, 0.70)

# Averaging
MEMORY_SIZE = 5

# Heatmap and Threshold
HEAT_THRESHOLD = 5  # 15
HEATMAP_METRICS = True

# Pickled Dataset and Model
TRAIN = False
ACCURACY = 9890
MODEL_FILE = (SAVE_DIR + 'model_%d.p' % ACCURACY) if not TRAIN else None
DATASET_FILE = (SAVE_DIR + 'dataset_%d.p' % ACCURACY) if not TRAIN else None
