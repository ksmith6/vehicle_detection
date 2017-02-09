##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video_Submission.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

In my Jupyter Notebook, the section on **Feature Extraction Code** is where I defined the process for extracting feature vectors.

The feature vectors were extracted from the training images in the **Training Data Generation** section.  I used the `glob` module to load all the filenames of each of the images in each category `vehicle` and `non-vehicle`.  Then for each image, I extracted feature vectors.

The final feature selection that I used was comprised of HOG features, color histogram features, and spatial binning features.  My image was converted from RGB to YUV color-space.

####2. Explain how you settled on your final choice of HOG parameters.

I attempted using many different combinations of `orientations`, `pixels_per_cell`, and `cells_per_block`.  I would tweak one parameter at a time, and observe the resulting performance.  Typically, I'd be able to increase the likelihood of detecting the car, but it came at the cost of lots of spurious false positives in the image.  

To tune my pipeline, I'd turn off all the feature sources (HOG, color hist, spatial binning) except for one, and then tweak that feature set's tuning parameters.  

I pursued this approach for several days; by the end, I was able to obtain fairly reliable detection of the cars, but I had a lot of false positives as well.  I found that I was obtaining decent performance from the HLS colorspace.  I like using the HLS colorspace due to its utility in the previous project.

At that point, my study partner, Apik Zorian, mentioned that he obtained good performance from the YUV colorspace and by using all the HOG channels.  I followed his suggestion and was able to obtain much better performance.  The number of false positives fell dramatically!

My final set of parameters is defined in the **Tuning Parameter Definition** section of the Jupyter Notebook.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a Linear SVM in the **Training** section of my Jupyter Notebook.  My pipeline has a few different modes.  By default, it will load a pre-trained Linear SVM classifier.  However, I can also set the `Retrain=True` optional input argument to retrain and save a new classifier.  This was required whenever I adjusted my feature set or tuning parameters.

The training works by loading the existing training data (or by generating training data if required), then passing the training feature vectors and labels the `clf.fit()` command.  To accelerate training, I also had additional option to train over a smaller batch of data (500 samples).  This would help me to more rapidly iterate while tuning the parameter sets.

Once training was complete, I'd evaluate the trained classifier on the test set of data (20% of my original dataset).  I typically observed test accuracy of 93% or higher, and this assured me that my classifier was well-trained.


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I re-used the Udacity provided algorithm for the sliding window search, because it offered more flexibility (with regard to the `None` arguments for defining search bounds) than the algorithm I had coded up for the exercise during lectures.

I implemented multi-scale search by calling the `slide_window()` function several times with different input parameters.

My goal was to minimize the number of boxes required to search the roadway area while ensuring coverage of the roadway.  To this end, I generated four sets of windows.  

The first set of windows is a set of small boxes near the horizon over a limited span of x-coordinates.  This would set of windows would detect cars far ahead of the vehicle.  These boxes were 64px by 64px, and they swept from x=200 to x=800, and from y=400 to y=500 with a 50% overlap.

The second set of windows is a set of larger boxes that are sized to 128px by 128px.  This set of boxes sweeps over the x-range of [300 1280] and the y-range of [400 720] with an overlap of 75%.  These boxes help detect vehicles that are closer (smaller Z-index) and thus occupy a larger portion of the image.  I had these windows use more overlap to ensure I didn't have large detection gaps for cars that are nearby. 

The third set of windows sweeps over the x-range of [400 1100] and the y-range of [350 500] with a window size of 50px by 50px.  This was an additional, focused scan on the roadway to help boost detection.  Because this is fairly similar to the first set of windows, it may be able to be consolidated with the first set of images, reducing processing time (future potential optimization).

The final set of windows was dynamically created based on the detected vehicles from the heatmap.  For each distinct blob identified by the `label()` function, I generated a set of images centered about the detected çar.  This helped to ensure that the detected vehicle was not lost between gaps in the fixed window sets.  It also helped to refine the bounding box of the cars.

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to try to minimize false positives and reliably detect cars?

I minimized false positives by tuning the parameters of my classifiers while ensuring I did not begin missing real car detection.  Furthermore, I implemented a `HeatMap` class (based on the Udacity lecture material) that helped to reject spurious measurements.  The heatmap would †rack the number of repeated detections on pixels across multiple images.  

For each window that detected a car, the heatmap pixels that the window contained were incremented by +1 until they saturated at a peak temperature of 10.  Upon processing each frame of the video stream, the heatmap was uniformly cooled by decrementing -1 from each pixel (capping the minimum pixel value at 0).  This helped to consistently track vehicles across multiple frames while rejecting spurious false positives.

After applying a minimum threshold of **3** as my heatmap threshold, I used the `label` function from `scipy.ndimage.measurements` to combine overlapping detection windows (in the heatmap space).  

With each identified cluster of detections, I used OpenCV to draw a blue rectangle around each detected vehicle. 

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

As described earlier, I implemented a Python class called `HeatMap` which tracked vehicle detections from frame to frame.  For each window that reported a detection, I would increment the pixels within the window in the `HeatMap` object (until the pixel values saturated at 10).  For each frame, I would `chill()` the HeatMap, causing the heat to drain away over time (decremented all pixels by 1).

This approach would make spurious detections fade away by the next image frame.  Only repeated detections over the same area would increase the pixel values above a threshold.

The thresholded heatmap was passed to the `scipy.ndimage.measurements.label()` function to detect distinct individual clusters within the heatmap.  This approach fails to distinguish two cars that are in close proximity to one another.  For each detected *blob*, a bounding box was defined to encapsulate it, and then each bounding box was overlaid as a blue rectangle on the image.

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There were a lot of tuning knobs available for this combination of features for image processing.  The high dimensionality of the parameter set space made it challenging to figure out which tuning knobs to tweak.  I spent a lot of time trying to tweak features to get better performance, and it rarely led to significant improvements.  This process gave me more appreciation for the advantages of the deep learning approach to image classification.  However, I did like the ability to troubleshoot the individual stages of my pipeline (a challenge in deep learning approaches).

This pipeline will have difficulty distinguishing cars from one another when one car partially occludes the other car.

My pipeline current focuses on cars that tend to be forward and to the right of its current position.  This was done to accelerate the pipeline (excluding left extremes of image), but it reduces the generality of the pipeline to driving in a left lane.

A major drawback of my current implementation is its runtime.  The algorithm takes about 7 seconds per image on my laptop (MacBook Pro).  This is largely driven by the number of search windows from which the pipeline must extract feature vectors.  If I spent more time re-fining my multi-scale window strategy, perhaps I could optimize the run-time a bit more.
