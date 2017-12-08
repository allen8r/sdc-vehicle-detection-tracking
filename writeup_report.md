# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car-notcar]: ./visuals/car-notcar-sample.png (Car/Notcar)
[hog-features]: ./visuals/hog-features.png (HOG Features)
[hog-subsampling]: ./visuals/hog-subsampling-patches.jpg (HOG Subsampling)
[hog-subsampling-result]: ./visuals/hog-subsampling-result.png (HOG Subsampling Results)
[color-hist]: ./visuals/histogram-of-color-channels.png (Histogram of Color Channels)
[labels-heat]: ./visuals/bboxes-labels-heatmap.png (Bboxes, Labels, Heat Map)
[spatial-binning]: ./visuals/spatially-binned-features.png (Spatially Binned Features)
[sliding-window]: ./visuals/sliding-window.png (Sliding Window)
[pipeline1]: ./visuals/pipeline-result1.png (Pipeline Result1)
[pipeline2]: ./visuals/pipeline-result2.png (Pipeline Result2)
[pipeline3]: ./visuals/pipeline-result3.png (Pipeline Result3)
[pipeline4]: ./visuals/pipeline-result4.png (Pipeline Result4)

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/allen8r/sdc-vehicle-detection-tracking/writeup_report.md) is my writeup report for this project.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `car` and `notcar` images.  In addition to the data sets from the samples provided from the GTI vehicle image database and the KITTI vision benchmark suite, I also extracted additional car images from the [Udacity data set](https://github.com/udacity/self-driving-car/tree/master/annotations). See `extract_cars.ipynb` for the work involved in building up the data set. With the increased size of the car images set, I also increased the number of notcar images by duplicating the provided notcar images so that the two classes are more balanced, reducing the likelihood of introducing bias in the classifier favoring the car class. Here is an example of one of each of the `car` and `notcar` classes:

![alt text][car-notcar]


Here are some examples of the color channel histograms used as features:

![alt text][color-hist]


And some examples of binning spatial features:

![alt text][spatial-binning]


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `LUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][hog-features]


The code for this step is contained in code cell #8 in the Jupyter notebook, `detect-cars.ipynb`, `scaled_X_y()` function. Here we call the `extract_features()` function from `vehicle_detection_helper.py`. Depending on option flags for spatial binning, color histogram, and hog features, extract_features() proceeds by acquiring each selected type of features and bundles them into a feature vector. After the features extraction `scaled_X_y()` continues to transform the features data using a [`StandardScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) in order to standardize the features by removing the mean and scaling to unit variance. The resultant `scaled_X` features and labels `y` are returned ready to be used in training the classifier.

#### 2. Explain how you settled on your final choice of HOG parameters.

Based on recommendation in the lessons, the `orientation` parameter is usuall between 6 and 12. I started out using 8 and worked my way up to 12, running the feature extraction each time. Along with the other parameters: `pixels_per_cell`, `cells_per_block`, and colorspace, I ran the feature extraction, trained the SVM classifier and then ran sample images through the pipeline to see the results. Balancing between accuracy and length of time to extract the features, I ended up with the parameter values listed above. In fact, there were some combinations of parameters that just took way too much time for the feature extraction, while achieving little improvement in detection performance.  

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In code cell# 10 in the `detect-cars.ipynb`, I trained a linear SVM using 20,837 cars and 21,132 notcars. Of the entire dataset, 20% was saved for testing, while the rest was used for training the classifier.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In `slide_window()` funtion of `vehicle_detection_helper.py`, I implemented the sliding window search. The sliding search implementation was very slow due to the function requiring to extract features for each individual search window. Here's an example result that took over 3 seconds to execute on a single frame image:

![alt text][sliding-window]

In order to achieve a similar outcome, but at a higher speed, a HOG subsampling search was implemented instead. In `find_cars()` function in `vehicle_detection_helper.py`, instead of extracting hog features per window patch, an entire frame image is fed into the hog feature extraction in one shot. Then a search partition to the image is sampled, a window at a time, using the classifier to predict whether each window matches a car or not. Because of the single-shot hog extraction, this subsampling method is much quicker. I was able to run find_cars() 4 times, oversampling the image and still ended up taking up only about .5 seconds. Here, we have a image showing the 4 subsampling patches in the region of interest:

 ![alt text][hog-subsampling]

By sampling partial regions of the image, even abandoning some areas along the x-axis, execution time was reduced significantly. In the subsampling, scaling factors were chosen for each of the subsampling runs. Here, I chose scaling factors of `[2.5, 2.0, 1.5, 1.25]` (See cell# 13 of `detect-cars.ipynb`.) As seen below, the result is similar to the sliding window method, but at a much lower cost in time of execution at 0.55 seconds.

![alt text][hog-subsampling-result]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

In the end I searched on 4 scales using LUV 3-channel HOG features in addition to spatially binned color and histograms of color in the feature vector, providing a nice result.  Here are some example images:

![alt-text][pipeline1]
![alt-text][pipeline2]
![alt-text][pipeline3]
![alt-text][pipeline4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://github.com/allen8r/sdc-vehicle-detection-tracking/blob/master/output_videos/project_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a test image, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the image showing the car positions:

![alt text][labels-heat]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My initial implementation of the HOG subsampling in searching for cars resulted in many, many false positives. Majority of them popping up along the left-hand side on the shoulder next to the median barrier. In order to combat this, I tried upping the threshold level in weeding out the false detections. However, in doing so, I was also significantly removing a lot of the positive detections, leading to some missed targets. I also played around with various color spaces to see if their respective extracted features would help in eliminating some of the false identifications. But nothing seemed to help. After more detailed analysis, and realizing that the false positives were concentrated in a particular region, I modified my `find_cars()` function to accept x-axis range values in order to reduce the search region and remove from search the area of much false positive detections. This modified search area eliminated most of the false positives while having the added beneficial side effect of also reducing the execution time for the search in the frame images.

Although the reduced search area fixed the false positives issue, this fix seems to be a bit of a hack. Since our project test video has our self-driving car only on the farthest left lane for the entirety of the video, we can assume that most likely no other cars are to be found to the left of our path of travel. However, in a live situation, should our self-driving car move to a lane on the right, then this search will be missing any other cars that appear to the left of our vehicle point of view. This modified search solution does not generalize well. A better way to get rid of the false positives should be investigated.

Another thing I did to try to combat the false positives was to obtain a whole bunch more of image samples for notcars. I did this by taking particular frames of the video that gave me problems and extracted numerous additional samples, augmenting the original dataset. Rebalancing the dataset with approximately equal portion of cars and retraining the SVM model, the results were disappointingly similar as before the modified search was implemented.  

I was able to achieve a test accuracy of 99.32% on the SVM classifier. But since I was getting so many false positives, I think the trained SVM model may be overfitting a bit and cannot generalize well. I would like to try out other classifier models such as training a neural network classifier or some other prediction model like a decision tree to see if I can get similar or better prediction results.

