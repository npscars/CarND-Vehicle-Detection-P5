---

**Vehicle Detection Project**

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/HOG_feature_car_and_not_car_RGB_2.png
[image2]: ./examples/HOG_feature_car_and_not_car_YUV_2.png
[image3]: ./examples/BoundingBox_test_images.png
[image4]: ./examples/Heatmap_test_images.png
[video1]: ./ptest_filtered.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. This Writeup includes all the rubric points and how I addressed each one.  Its using the baseline template available here [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md).

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the third code cell of the IPython notebook [CarND-Vehicle-Detection-P5.ipynb](./CarND-Vehicle-Detection-P5.ipynb)

I started by reading in all the `vehicle` and `non-vehicle` images.
I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. Also shown in IPython notebook 4th code cell.

Here is an example using the `RGB` and `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:
![alt text][image1]
![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.
Please see answer to Q2 in Q3 as i decided on final choice considering the SVM test accuracy.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I tried various combinations of parameters but the one I finally selected for the project was 'LUV' color space along with following combination of HOG, spatial and color space parameters.
HOG parameters   :-> 'orientation=9', 'pixels_per_cell = (8,8)', 'cells_per_block = (2,2)' for 'ALL' image color channels,
Spatial parameter:-> resize the image with 'spatial_size = (16,16)' and potentially still extract car feature
Color space parameters:-> 'hist_bins = 16' 
The reason for selecting above parameter after several trial and error method was because it gave > 99% of Support Vector Classification algorithm accuracy on test data. This can be seen as an output of IPython notebook's 5th code cell.

I trained a linear SVM using sklearn package's 'svm' module. First I combined all car and non-car features. Then using sklearn's StandardScalar.fit() and transform method, I normalized the features. Then I combined the labels for car and non-car and then supplied it to sklearn's _train_test_split()_ method to seperate training and testing data. I used the Linear Support Vector Classifier to fit the data and then tested the accuracy of model on test data like typical supervised machine learning process.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

6th and 7th code cell in the IPython notebook has two main functions about sliding window and search window respectively.
Sliding window functions defines the windows of default (64,64) pixels in the image's region of interest with 50% overlap so that we don't miss too many features. The overlap can be increased to say 75% to cover potentially more features like car, but that would increase the computation time too. Also for pipeline later on you will see that 96,96 pixel windows were selected to get faster computation time.
Then Search window function looks out for features in those windows and if found using the previously fitted SVC model then it returns only those '_on_windows_' as the windows of interest.
One of the sample output of combination of both sliding window and search window output can be found in following image:
![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

For final pipeline though I implemented '_find_cars()' function as discussed in tutorial video by Ryan Keenan of Udacity. 
So instead of extracting hog features for each 96x96 window, this function calculates hog features of whole image (i.e. region of interest) and then sub-samples (@64x64 pixels) the array to extract hog/spatial and color space features of each block as shown in code cell 8. These blocks are then iteratively predicted to contain a car or not for whole region of interest.
A heatmap is created which gives more points for area where there are more than one hits per block.
An example is shown in the image below.
![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./ptest_filtered.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used Object Oriented Programming concept of classes for this project like advanced lane finding project.
The class name Vehicle() describes characterestics which are required to draw rectangle for each individual vehicle which is found. Please have a look at '5th before last code cell' for my Vehicle class implementation in Ipython notebook.
'_add_measurement()_' method adds the location of the vehicle found in the video frame. It also smoothes using weighted average of the bounding box for drawing a rectangle where the vehicle is found. Also has a method to draw itself when requested by pipeline.

4th before last code cell of the notebook contains the pipeline:
Before define all parameters required to run the pipeline then,
1) find cars using the svc learned feature extraction
2) convert the heat map into labels (in this case individual cars) >0 with surrounded >0 values is then combined into one box i.e. identified car. label function of `scipy.ndimage.measurements` package is used for this conversion. 
3) get all found bounding boxes for current image
4) add new found vehicles or append measurement for old vehicles
5) give penalty to non-detected cars in each frame of video
6) remove cars in reverse order from carslist if not detected since last 'max_age' frames of video
7) Finally draw each found cars rectangle and text on it.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In the Vehicle() class, initially I was considering smoothed (weighted) bounding top left corner and bounding box bottom right corner as my recent values. But due to that bounding box on the video was lagging behind what was found in the pipeline. Then I realised that the recent values are bounding box values found and not the smoothed bounding box which I generate in the '_add_measurement()' method.

Eventhough the trained SVC classifer had >99% accuracy on test data, there were false positivies found. This clearly indicates that the number of training data is not enough. 

There is currently a fine balance between especially selecting value for following parameteres:
1) max_age - number of non-detections to delete the car (too high value and the box is shown in location where there is no more a car in video frames) 
2) min_age - minimum number of detections to show the car (too low value gives more false positives)
3) cache_length - caching of bounding box for smooth change in video (more the cache, slower is the movement towards actual car location in video frame)

