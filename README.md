
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.jpg
[image2]: ./output_images/hog.jpg
[image3]: ./output_images/sliding_window1.jpg
[image4]: ./output_images/sliding_window2.jpg
[image5]: ./output_images/heatmap1.jpg
[image6]: ./output_images/heatmap2.jpg
[image7]: ./output_images/heatmap3.jpg
[image8]: ./output_images/heatmap4.jpg
[image9]: ./output_images/heatmap5.jpg
[image10]: ./output_images/heatmap6.jpg
[image11]: ./output_images/labels1.jpg
[image12]: ./output_images/labels2.jpg
[image13]: ./output_images/labels3.jpg
[image14]: ./output_images/labels4.jpg
[image15]: ./output_images/labels5.jpg
[image16]: ./output_images/labels6.jpg
[image17]: ./output_images/test1.jpg
[image18]: ./output_images/test2.jpg
[image19]: ./output_images/test3.jpg
[image20]: ./output_images/test4.jpg
[image21]: ./output_images/test5.jpg
[image22]: ./output_images/test6.jpg
[video1]: ./output_images.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 28 through 47 of the file called `helper_function.py`).  
I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters,and `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` has a best test accuracy with a linear SVM classifier.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using hog features, spatial color features and histogram features. 
First,get image list, then for every single image,  extract hog features, spatial color features and histogram features, and append the new feature vector to the features list.
Second, to Normalize Features with `StandardScaler()` method. Random Shuffling of the data, splitting the data into a training and testing set. Labeled the Data.
Finally, train the classifer with training data and label. And test the accuracy of the classifer, and tuning parameters accordingly. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search  using a 64 x 64 base window,`pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`,would result in a search window overlap of 75%, that is move 25% each time, leaving 75% overlap with the previous window.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Finnaly I searched with YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image3]
![alt text][image4]

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./test_videos_output/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image17]
![alt text][image18]
![alt text][image19]
![alt text][image20]
![alt text][image21]
![alt text][image22]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. When there are shadows and the lighting condition changes rapidly, the pipeline is not robust.
2. The vision range is quite limited, when the front car is a bit far away, it's not easy to detected the car from the small image.
3. Still the False positive happen at some time, even with heatmap method.

The pipeline maybe more rubust with more image preprocessing method. 
And combining judgement with information from last frame/next frame image, will help to better judge false positive, basically, the  vehicle tracking box should be continous moved in each frame.
Also, maybe it's good to try with deep learning method to judge the vehicle's position (pixels position).
