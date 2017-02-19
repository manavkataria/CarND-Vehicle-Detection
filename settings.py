# settings.py

# Debug
DEBUG = True
DISPLAY = True

# Video Input
INPUT_VIDEOFILE = 'project_video.mp4'
OUTPUT_DIR = 'output_images/'

# Save
SAVE_DIR = 'save/'

# Road Test
TEST_IMAGES_DIR = 'test_images/'

# Training Directory
TRAINING_DIR = 'training/'              # 17760 Samples
VEHICLES_DIR = 'vehicles/'              # 8792 Samples
NON_VEHICLES_DIR = 'non-vehicles/'      # 8968 Samples


# HOG Parameters
COLORSPACE = 'HLS'  # Can be RGB*9717, HSV, LUV, HLS*9916, YUV, YCrCb*9899
ORIENT = 9
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2
HOG_CHANNEL = 'ALL'  # Can be 0, 1, 2, or "ALL"
Y_START_STOP = [500, None]

# Pickled Files
ACCURACY = 9916
MODEL_FILE = (SAVE_DIR + 'model_%d.p' % ACCURACY)
DATASET_FILE = (SAVE_DIR + 'dataset_%d.p' % ACCURACY)
