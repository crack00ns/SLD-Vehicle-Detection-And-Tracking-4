import cv2
import glob
import time
import pickle
import os
import pdb # For bebudgging purposes
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from skimage.feature import hog
from scipy.ndimage.measurements import label
try:
    from sklearn.model_selection import train_test_split  # for new sklearn >= v.18
except ImportError:
    from sklearn.cross_validation import train_test_split  # for old sklearn < v.18

''' Config parameters/constants used for Vehicle Detection '''
# FilePaths for Dataset, Output, Test Images Path
DATA_SET_PATH =  'dataset/'
OUT_PATH = 'output/'
TEST_PATH = 'test_images/'

# Min and max in y to search in find_cars_for_scale()
Y_START = 400  # Min Y
Y_END   = 700  # Max Y

# Training Parameters
REDUCE_DATASET = False # If true reduce the dataset size to fasten training (For testing purposes only)
REDUCE_MODEL_SIZE = 500 # Size of reduced dataset

# Tweak these paramters to obtain higher test score
TRAIN_TEST_RATIO = 0.2      # Train test split ratio
COLOR_SPACE = "YCrCb"       # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
ORIENT = 9                  # HOG orientations
PIX_PER_CELL = 8            # HOG pixels per cell
CELL_PER_BLOCK = 2          # HOG cells per block
HOG_CHANNELS = [0,1,2]      # Can only contain 0, 1, 2 but can't be empty
SPATIAL_SIZE = (32,32)      # Spatial binning dimensions
HIST_BINS = 32              # Number of histogram bins
SPATIAL_FEAT = True         # Spatial features on or off
HIST_FEAT = True            # Histogram features on or off

HEAT_MAP_THRESHLD = 2       # Threshold for Heatmap
SCALES = [1.1, 1.4, 1.8, 2.4, 3.0, 3.6]     # Scales used to search for cars in an image 
# Frame Smootheming Parameters
MOV_AVG_FACTOR = 0.5        # Exponential Moving Average factor. 1 means no averaging, 0 means past is remembered forever
CENTER_CLOSE_THRESHOLD = 10 # Threshold to determine whether centers of two rectangles are close (i.e., same car detected)

BOUNDING_BOX_COLOR = (255,0,0) # COLOR for Bounding Boxes

class FeatureExtractor():
    ''' Feature Extractor Class is used to to extract various features from an image, including HOG,
        Spatial binning, Color histograms '''
    
    def get_hog_features(self, img, orient, pix_per_cell, cell_per_block, 
                            vis=False, feature_vec=True):
        ''' Define a function to return HOG features and visualization '''

        # Call with two outputs if vis==True
        if vis == True:
            features, hog_image = hog(img, orientations=orient, 
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                                      block_norm = "L2", 
                                      transform_sqrt=True, 
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image

        # Otherwise call with one output
        else:      
            features = hog(img, orientations=orient, 
                           pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block), 
                           block_norm = "L2",
                           transform_sqrt=True, 
                           visualise=vis, feature_vector=feature_vec)
            return features

    def bin_spatial(self, img, size=(32, 32)):
        ''' Define a function to compute binned color features '''
        
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel() 
        # Return the feature vector
        return features

    def color_hist(self, img, nbins=32, bins_range=(0, 1)):
    '''  Define a function to compute color histogram features bins_range=(0, 1) because we are reading .png files with mpimg! '''
        
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)

        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def extract_features(self, imgs, color_space='RGB', spatial_size=(32, 32),
                            hist_bins=32, orient=9, 
                            pix_per_cell=8, cell_per_block=2,
                            spatial_feat=True, hist_feat=True):
        '''Define a function to extract features from a list of images
           Have this function call bin_spatial() and color_hist() '''

        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            file_features = []
            # Read in each one by one
            image = mpimg.imread(file)
            img_features = self.single_img_features(image, color_space, spatial_size,
                            hist_bins, orient, 
                            pix_per_cell, cell_per_block,
                            spatial_feat, hist_feat)
            features.append(img_features)
        # Return list of feature vectors
        return features

    def single_img_features(self, img, color_space='RGB', spatial_size=(32, 32),
                            hist_bins=32, orient=9, 
                            pix_per_cell=8, cell_per_block=2,
                            spatial_feat=True, hist_feat=True):    
    ''' Define a function to extract features from a single image window '''

        #1) Define an empty list to receive features
        img_features = []
        #2) Apply color conversion if other than 'RGB'
        feature_image = convert_color(img)
        if spatial_feat == True:
            spatial_features = self.bin_spatial(feature_image, size=spatial_size)
            #3) Append features to list
            img_features.append(spatial_features)
        #4) Compute histogram features if flag is set
        if hist_feat == True:
            hist_features = self.color_hist(feature_image, nbins=hist_bins)
            #5) Append features to list
            img_features.append(hist_features)
        #6) Compute HOG features
        hog_features = []

        for channel in HOG_CHANNELS:
            hog_features.extend(self.get_hog_features(feature_image[:,:,channel], 
                                orient, pix_per_cell, cell_per_block, 
                                vis=False, feature_vec=True))      
        #7) Append features to list
        img_features.append(hog_features)

        #8) Return concatenated array of features
        return np.concatenate(img_features)

class VehicleModelClassifier():
    ''' Vehicle Model Classifier class is responsible for defining and 
        training the learning model and also load and save models '''
    def __init__(self, pickle = "svm_model.p"):
        ''' Initialize the model here '''
        self.dataset_path=DATA_SET_PATH
        self.model = LinearSVC()
        self.X_scaler = None
        self.trained = False
        self.feature_extractor = FeatureExtractor()
        self.pickle = pickle

    def save_model(self):
        ''' Save the model to a pickle file '''
        model_data = {}
        model_data["model"]    = self.model
        model_data["X_scaler"] = self.X_scaler
        pickle.dump(model_data, open(os.path.join(OUT_PATH,self.pickle), "wb"))
    
    def load_model(self):
        ''' Load the model from a pickle file if available, else train the model'''
        if os.path.isfile(os.path.join(OUT_PATH,self.pickle)):
            print("Loading model from pickled file...")
            with(open(os.path.join(OUT_PATH,self.pickle), mode='rb')) as f:
                model_data = pickle.load(f)
            self.model    = model_data["model"]
            self.X_scaler = model_data["X_scaler"]
            self.trained = True
        else:
            print("Model pickle file not found. Training model...")
            self.train_model()

    def train_model(self):
        ''' Train the model here '''
        images = glob.glob(DATA_SET_PATH+"/**/*.png", recursive=True)  # Get images from dataset folder
        cars = []
        notcars = []
        for image in images:
            if 'non-vehicles' in image:
                notcars.append(image)
            else:
                cars.append(image)

        if REDUCE_DATASET: # Reduce the dataset size in order to fast train for testing purposes
            print('Reducing the dataset size to', REDUCE_MODEL_SIZE)
            cars    = cars[0:REDUCE_MODEL_SIZE]
            notcars = notcars[0:REDUCE_MODEL_SIZE]
        else:
            print ('Using the full dataset to train...')

        # Extract Car and Non Car Features
        print('Extracting Features...')
        car_features = self.feature_extractor.extract_features(cars, color_space=COLOR_SPACE, 
                                spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS, 
                                orient=ORIENT, pix_per_cell=PIX_PER_CELL, 
                                cell_per_block=CELL_PER_BLOCK, 
                                spatial_feat=SPATIAL_FEAT, hist_feat=HIST_FEAT)
        notcar_features = self.feature_extractor.extract_features(notcars, color_space=COLOR_SPACE, 
                                spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS, 
                                orient=ORIENT, pix_per_cell=PIX_PER_CELL, 
                                cell_per_block=CELL_PER_BLOCK, 
                                spatial_feat=SPATIAL_FEAT, hist_feat=HIST_FEAT)

        # Construct the X vector (Independent Variable) 
        X = np.vstack((car_features, notcar_features)).astype(np.float64)                        

        # Normalize the X Vector: Fit a per-column scaler
        self.X_scaler = StandardScaler().fit(X)

        # Apply the scaler to X
        scaled_X = self.X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Random state should be fixed to compare models 
        rand_state = 50 #np.random.randint(0, 100)

        # Shuffle Data
        scaled_X, y = shuffle(scaled_X, y, random_state = rand_state)

        # Split up data into randomized training and test sets       
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=TRAIN_TEST_RATIO, random_state=rand_state)

        print('Using:',ORIENT,'orientations',PIX_PER_CELL, 'pixels per cell and', CELL_PER_BLOCK,'cells per block')
        print('Feature vector length:', len(X_train[0]))

        # Check the training time for the SVC
        t=time.time()
        self.model.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train the model...')
        # Check the score of the SVC
        print('Test Accuracy of Model = ', round(self.model.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t=time.time()
        self.trained = True
        self.save_model()

class VehicleDetector():
    ''' This class is used to detect vehicles in a frame of video and put bounding boxes around the detected 
        vehicles. Successive frames can also be smmothened using an exponential moving average '''

    def __init__(self, vis = False, smoothen_frame_boxes = False, smoothen_heatmap = False):
        ''' Initialize the class variables here '''
        self.classifier = VehicleModelClassifier()       # Create the classifier object
        self.classifier.load_model()                     # load a model if picked file is available else train a model
        self.vis = vis                                   # Visualize the heatmaps or not (for testing with individual images)
        self.smoothen_heatmap = smoothen_heatmap         # Flag to set whether frame by frame exponential smoothening of heatmap is on or not
        self.smoothen_frame_boxes = smoothen_frame_boxes # Flag to set whether frame by frame exponential smoothening of boxes is on or not
        self.prev_boxes = None                           # Previous bounding boxes that were detected
        self.prev_heatmap = None                         # Previous heatmap

    def detect_cars_frame(self,img):
        ''' Detect Cars in a frame '''
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        for scale in SCALES:
            out_img, box_list = self.find_cars_for_scale(img, scale)
            heat = self.add_heat(heat,box_list) 
        if self.prev_heatmap is not None and self.smoothen_heatmap:
            heat  = MOV_AVG_FACTOR*heat+(1-MOV_AVG_FACTOR)*self.prev_heatmap

        heatmap = self.apply_threshold(heat,HEAT_MAP_THRESHLD)
        self.prev_heatmap = heatmap

        labels = label(heatmap)

        if self.vis:
            print(labels[1], 'cars detected!')
            heatmap = np.clip(heatmap, 0, 255)
            show_images_sbs(labels[0], heatmap, "Cars Positions", "Heatmap", "gray", "hot")
        return self.draw_labeled_bboxes(img, labels)

    def centers_are_close(self, center1, center2):
        ''' Check if centers of two bounding boxes (in successive frames) are similar within a threshold, 
            which means same car was detected in two frames '''
        if (np.abs(center1[0]-center2[0])<=CENTER_CLOSE_THRESHOLD) and (np.abs(center1[1]-center2[1])<=CENTER_CLOSE_THRESHOLD):
            return True
        else:
            return False

    def average_recenter_box(self, box1, box2):
        ''' Recenter Box1 using moving averages from box2 '''
        # Calculate length and width of box1
        len_box1 = np.abs(box1[0][0]-box1[1][0])
        wid_box1 = np.abs(box1[0][1]-box1[1][1])

        # Calculate length and width of box2
        len_box2 = np.abs(box2[0][0]-box2[1][0])
        wid_box2 = np.abs(box2[0][1]-box2[1][1])

        # Update box1 length and width moving averages using box2's length and width
        len_box1 = MOV_AVG_FACTOR*len_box1+(1-MOV_AVG_FACTOR)*len_box2
        wid_box1 = MOV_AVG_FACTOR*wid_box1+(1-MOV_AVG_FACTOR)*wid_box2

        # Calculate centers of box1
        centerx_box1 = np.abs(box1[0][0]+box1[1][0])/2
        centery_box1 = np.abs(box1[0][1]+box1[1][1])/2

        # Calculate centers of box2
        centerx_box2 = np.abs(box2[0][0]+box2[1][0])/2
        centery_box2 = np.abs(box2[0][1]+box2[1][1])/2

        # Update center of box1 with moving averages using box2's center
        centerx_box1 = MOV_AVG_FACTOR*centerx_box1+(1-MOV_AVG_FACTOR)*centerx_box2
        centery_box1 = MOV_AVG_FACTOR*centery_box1+(1-MOV_AVG_FACTOR)*centery_box1

        # New box coordinates
        newbox = ((int(centerx_box1 - len_box1/2), int(centery_box1 - wid_box1/2)), (int(centerx_box1 + len_box1/2), int(centery_box1 + wid_box1/2)))
        return newbox
        
    def moving_average_boxes(self, bboxes):
        ''' Calculate moving averages of boxes detected in previous frame '''
        centers = [[sum(y) / len(y) for y in zip(*bbox)] for bbox in bboxes]
        prev_centers = [[sum(y) / len(y) for y in zip(*bbox)] for bbox in self.prev_boxes]

        for i, center in enumerate(centers):
            for j, prev_center in enumerate(prev_centers):
                if self.centers_are_close(center, prev_center):
                    bboxes[i] = self.average_recenter_box(bboxes[i], self.prev_boxes[j])
        return bboxes


    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        bboxes = []
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            bboxes.append(bbox)

        if self.smoothen_frame_boxes:
            without_smoothening_img = self.draw_boxes_on_images(img, bboxes)
            if self.prev_boxes is not None:
                bboxes = self.moving_average_boxes(bboxes)
            self.prev_boxes = bboxes
            smoothened_img = self.draw_boxes_on_images(img, bboxes)
            # show_images_sbs(without_smoothening_img,smoothened_img,"Withour Smoothening", "Smoothened")
        # Return the image
        return self.draw_boxes_on_images(img, bboxes)

    def draw_boxes_on_images(self,img, bboxes, color=BOUNDING_BOX_COLOR, thick=6):
        ''' Define a function to draw bounding boxes '''

        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    def draw_boxes_for_scales(self, img, scales, ystart=Y_START):
        ''' Draw boxes for different scale sizes for visualization '''

        draw_img = np.copy(img)
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        font = cv2.FONT_HERSHEY_SIMPLEX
        win_draw = 0
        xbox_left = 0
        for idx, scale in enumerate(scales):
            color = (np.random.randint(256), np.random.randint(256), np.random.randint(256))
            xbox_left = xbox_left+win_draw+60
            ytop_draw = 0
            win_draw = np.int(window*scale)

            cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),color,6) 
            cv2.putText(draw_img,'Scale:{}'.format(scale),(xbox_left, ytop_draw+ystart-30), font, 0.8,(255,255,255),2)
        return draw_img

    def find_cars_for_scale(self, img, scale, ystart=Y_START, ystop=Y_END, orient=ORIENT, pix_per_cell=PIX_PER_CELL, 
                    cell_per_block=CELL_PER_BLOCK, spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS):
        ''' Define a single function that can extract features using hog sub-sampling and 
            make predictions. Part of the Code borrowed from class'''
        
        draw_img = np.copy(img)

        feature_extractor = FeatureExtractor()

        img = img.astype(np.float32)/255  # Since trained over png images but passed jpg image. mpimg reads png (0,1) scale but jpg (0,255)
        
        img_tosearch = img[ystart:ystop,:,:]

        ctrans_tosearch = convert_color(img_tosearch)

        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
        nfeat_per_block = orient*cell_per_block**2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        
        # Compute individual channel HOG features for the entire image
        if 0 in HOG_CHANNELS: hog1 = feature_extractor.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        if 1 in HOG_CHANNELS: hog2 = feature_extractor.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        if 2 in HOG_CHANNELS: hog3 = feature_extractor.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
       
        box_list = []
        # Sliding window with scaling. HOG features extracted above will be used
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch for each of the HOG channels defined in configuration
                if 0 in HOG_CHANNELS: hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                else: hog_feat1 = []
                if 1 in HOG_CHANNELS: hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                else: hog_feat2 = []
                if 2 in HOG_CHANNELS: hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                else: hog_feat3 = []
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
              
                # Get color features
                if SPATIAL_FEAT: 
                    spatial_features = feature_extractor.bin_spatial(subimg, size=spatial_size)
                else:
                    spatial_features = []
                if HIST_FEAT: 
                    hist_features = feature_extractor.color_hist(subimg, nbins=hist_bins)
                else:
                    hist_features = []

                # Scale features and make a prediction
                test_features = self.classifier.X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
                test_prediction = self.classifier.model.predict(test_features)
                
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                    box_list.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
        return draw_img, box_list

    def add_heat(self, heatmap, bbox_list):
        ''' Increment all pixels within heatmap that lie within the given bounding boxes '''

        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        # Return updated heatmap
        return heatmap# Iterate through list of bboxes

    def apply_threshold(self, heatmap, threshold):
        ''' Apply Heatmao threshold ''' 

        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

# Additional Utility Functions -->
def convert_color(img, color_space = COLOR_SPACE):
    ''' Convert Color Space '''
    if color_space != 'RGB':
        if color_space == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: img = np.copy(img)

    return img


def show_images_sbs(img1, img2, title1="First Image", title2="Second Image", cmap1 ="gray", cmap2 ="gray", figsize=(18,9), save = False, filename = "out.png"):
    '''Show Images Side by Side'''
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.imshow(img1, cmap=cmap1)
    ax1.set_title(title1, fontsize=15)
    ax2.imshow(img2, cmap=cmap2)
    ax2.set_title(title2, fontsize=15)
    if save: plt.savefig(OUT_PATH+filename)
    plt.show()


def show_single_image(img,title="", figsize=(7.5,7.5), save=False, filename = "out.png"):
    ''' Show Single Image '''
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap = "gray")
    plt.title(title, fontsize=15)
    if save: plt.savefig(OUT_PATH+filename)
    plt.show()