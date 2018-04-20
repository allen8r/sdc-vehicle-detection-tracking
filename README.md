# Vehicle Detection
<img width="640" src="./visuals/vehicle-detection.gif"/>


The Project
---

The goals / steps of this project are:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images.
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector.
* Train a Linear SVM classifier to detect cars.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and  [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train the classifier. These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself. You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

Some example images for testing your pipeline on single frames are located in the `test_images` folder.  Examples of the output from each stage of the pipeline are saved in the folder called `visuals` The video called `project_video.mp4` is the input video the pipeline processes.

Here's the **[writeup](./writeup_report.md)** for the project and the final **[output video](https://github.com/allen8r/sdc-vehicle-detection-tracking/blob/master/output_videos/project_output.mp4?raw=true)**.

