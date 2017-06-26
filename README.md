# Vehicle Detection and Tracking
Implementation of  Project 5 of Self Driving Car **: Vehicle Detection and Tracking**.
---

**Vehicle Detection and Tracking Project**
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Normalize the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run a pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[image1]: ./output/test_car_original.png "Test Car Original"
[image2]: ./output/test_car.png "Car Hog Features"
[image3]: ./output/test_noncar_original.png "Test Not Car Original"
[image4]: ./output/test_noncar.png "Non Car Hog Features"
[image5]: ./output/scaleboxes.png "Boxes with Scales for Sliding Window"
[image6]: ./output/test1.png "Test 1"
[image7]: ./output/test2.png "Test 2"
[image8]: ./output/test3.png "Test 3"
[image9]: ./output/test4.png "Test 4"
[image10]: ./output/test5.png "Test 5"
[image11]: ./output/test6.png "Test 6"

---
## Project code
All main part of code for this project is implemented in `./vdutils.py` file. The IPython notebook `./vehicle_detection.ipynb` has a streamlined code showing how to run the code from `vdutils.py` on a sample image and video files. Heatmap visualization images can be obtained by setting `vis` parameter in `VehicleDetector` object to be `True`. All the generated images are stored in `output` folder. The dataset can be downloaded and put in dataset/vehicles and dataset/non-vehicles folders respectively:
https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip
https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip

Project video project_video.mp4 can be downloaded from here:
https://github.com/udacity/CarND-Vehicle-Detection

### 1. Feature Extraction
---
The code for this step is contained in lines 54-155 of `./vdutils.py` within class `FeatureExtractor`.  In this class, `get_hog_features()` , `bin_spatial()`, `color_hist()`, `extract_features()` `single_img_features()` functions have been implemented. By setting `SPATIAL_FEAT` and `HIST_FEAT` boolean flags, we can extract spatial and histogram features in addition to HOG features. Here is an example of HOG features extracted for a car and non car test image in two color spaces, RGB and YCrCb with 8x8 pixels and block normalization with 2 cells per block. Empirically speaking, HOG features for YCrCb space seems less correlated for many car images so we may expect to get better features out of those but it is just a hypothesis. Please check below:

![Car Original][image1]
*Original Car Image*

![HOG Features for a Car in TGB and YCrCb Color spaces][image2]
*HOG Features for a Car in TGB and YCrCb Color spaces*

---

![Non Car Original][image3]
*Original Non Car Image*

![HOG Features for a Non Car in TGB and YCrCb Color spaces][image4]
*HOG Features for a Non Car in TGB and YCrCb Color spaces*

### 2. Training a Classifier
---
Next, we train a classifier that can learn to identify car and non car images. I started by reading in all the `vehicle` and `non-vehicle` images in the `dataset` folder. I tried both SVM and DecisionTree and SVM seem to train faster probably because of high dimensional feature vector. I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. The code for this step is contained in lines 157-254 of `./vdutils.py` within class `VehicleModelClassifier`. Functions for training, saving and loading a model have been implemented here. After playing with bunch of parameters, I settled on the following choice of parameters:

| Parameters        | Value   | 
|:-------------:|:-------------:| 
| Model    | SVM | 
| Train test split ratio   | 0.2 | 
| Color Space    | YCrCb | 
| HOG Orientations    | 9 | 
| HOG pixels per cell    | 8 | 
| HOG cells per block   | 2 | 
| HOG Channels    | 0, 1, and 2 |
| Spatial binning dimensions    | (32, 32) |
| Number of histogram bins    | 32 |

We can reduce the feature vector size to increase speed by not including spatial bins and color histograms but there was a slight decrease in accuracy so I decided to include them. Also YCrCb color space seems to provide a better accuracy. HOG Orientations of 9, Pixels per cell as 8x8 and 2 cells per block for block normalization seem to work well and provides a decent accuracy results. Data was randomly shuffled, normalized with zero mean unit variance and a train test split ratio of 80%-20% was chosen. Finally, after using the full dataset to train and extracting all three features, we get a feature vector of length 8460, and it takes 10.01 Seconds to train the model with a very high **test accuracy of 99.38%**.    

### 3. Detecting a Vehicle
---
The code for this step is contained in lines 256-495 of `./vdutils.py` within the class `VehicleDetector`. This class is used to detect vehicles in a frame of video and put bounding boxes around the detected vehicles.

##### Sliding Window Approach
Once we have model training complete, we can use a sliding window approach where we extract all HOG features just once and use different scales to effectively set the window size and subsample the features. The implementation of this step is given in the function `find_cars_for_scale()` function in the class `VehicleDetector`. We chose the following scales :{1.1, 1.4, 1.8, 2.4, 3.0, 3.6} to obtain various sliding window size. Motivation to choose these scales was inspired from this [blog post](https://chatbotslife.com/vehicle-detection-and-tracking-using-computer-vision-baea4df65906). 
![alt text][image5]
*Scales used in Sliding window approach*

##### Constructing Thresholding Pipeline
To detect vehicles and remove false positives on a single frame we use a sliding window approach on different scales and construct a heatmap with all the pixels where multiple detections were detected. Only the lower half of the image was searched (`Y_START:400` and `Y_END:700`). Finally we apply a threshold on heatmap (`HEAT_MAP_THRESHLD:2`) to find the locations where possibility of having a car is very high. This is implemented in `detect_cars_frame()` function in the class `VehicleDetector`. We used `scipy.ndimage.measurements.label()` to isolate and locate individual cars and draw a bounding box across each detected car assuming each blob correspond to one car. The results on test images seem very accurate as shown below:

![alt text][image6]
*Output with Test Image 1*
---
![alt text][image7]
*Output with Test Image 2*
---
![alt text][image8]
*Output with Test Image 3*
---
![alt text][image9]
*Output with Test Image 4*
---
![alt text][image10]
*Output with Test Image 5*
---
![alt text][image11]
*Output with Test Image 6*
---
*Result of Heatmap Thresholding on test images to detect and draw bounding boxes across car images*
Next, we discuss our approach to smoothen detection across successive frames.

### 4. Smoothening the successive frames
---
The above pipeline can be applied to a video stream and while the results are promising, the bounding boxes seem a bit jittery and sometimes we noticed a few false positives. To smoothen and streamline the detection across successive frames, I used an **exponential smoothening** technique on both bounding boxes and the detected heatmap. I added two additonal boolean/binary flags in `VehicleDetector` class called `smoothen_heatmap` and `smoothen_frame_boxes`. Also added two variables that store boxes and heatmap found on previous frames called `prev_boxes` and `prev_heatmap`. 
##### Box Smoothening
Once we have labeled boxes on a single video frame. We iterate over all the boxes found in the previous frame and if the centers of any two boxes are within a threshold (`CENTER_CLOSE_THRESHOLD:10`), we assume that the two boxes are essentially pointing at the same car. So we take a moving average of the centers, length and width of the new box with a moving average factor (`MOV_AVG_FACTOR:0.5`) e.g. `new_length_box=mov_avg_factor*current_length+(1-mov_avg_factor)*previous_length`. The code is implemented in `centers_are_close()`, `moving_average_boxes()`, `average_recenter_box()` in the `VehicleDetector` class. The code snippet is shown below:
```
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
```

##### Heatmap Averaging
A similar approach was used for heatmaps. If the heatmap smoothening flag is set in `VehicleDetector` class then currently detected heatmap is smoothened with previous heatmap before applying the threshold as shown below (in `detect_cars_frame()`):

```
       if self.prev_heatmap is not None and self.smoothen_heatmap:
            heat  = MOV_AVG_FACTOR*heat+(1-MOV_AVG_FACTOR)*self.prev_heatmap

        heatmap = self.apply_threshold(heat,HEAT_MAP_THRESHLD)
        self.prev_heatmap = heatmap
```
This approach seems to work quite well and a few false positives seem to have dissappeared and bounding boxes also seem less jittery as seen below in the video pipeline for both smoothened and unsmoothened case. 

---

### Pipeline (video)
The pipeline is applied to the project video before setting the `smoothen_heatmap` and `smoothen_frame_boxes`. The result can be seen on youtube link:
[Pipeline Car Detection Without Smoothening](
https://youtu.be/4TG51q1x8Ew)

After smoothening by setting `smoothen_heatmap` and `smoothen_frame_boxes` to be `True` in `VehicleDetector` class object we get the following output:
[Pipeline Car Detection With Smoothening](
https://youtu.be/cDnxS3Cb4xw)

As seen in the videos, box jitteriness has significantly reduced and bounding box is more stable and a few false positives previously seen betweeen 00:40 and 00:42 are all gone!

---

### Discussion and Issues Faced
I think the project was quite interesting and challenging at many fronts. Also, there are multiple ways available to implement this project with different trade-offs so this was a great learning opportunity to explore options. Here is a brief discussion on some issues:
1. **Feature Vector Selection**: The current approach is slow (takes almost 2.5 seconds on single image and ~40 mins to process the project video). Feature extraction seem to be the main bottleneck since model prediction is quite fast. However, speed can obviously be increased at the expense of accuracy. I went for accuracy rather than speed but also explored options to reduce feature vector size. First of all, we can possibly get rid of spatial bins and color histograms since they bring little information and increased processing. Also HOG channels Cr and Cb also seem to carry less information about car shapes for many images. By keeping only HOG channel 0, I seem to get an accuracy of 98% which is high but results in some false positives and missed detection. A more robust threholding approach if employed can improve the speed significantly.
2. **Labeling Cars**: With the box smoothening approach, as discussed above, it is also possible to label cars when the same boxes are detected in successive frames. I didn't implement it but it can be done but probably wont be very robust. Color histograms may provide a clue to approach this.  
3. **Deep Learning Approach**: While the current approach seems to work quite well, a CNN based approach could be way more robust but would provide less transparency. Since, I wanted to try more traditional approaches for my own learning experience, I used an SVM based approach. I plan to reimplement this project later using deep learning (Fast R-CNN, or other segmentation approaches disusses in CS231 class at Stanford).
