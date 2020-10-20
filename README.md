## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![](output_images/project_video_frame_233.jpg)

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

## Usage:

### 1. Set up the environment 
`conda env create -f environment.yml`

To activate the environment:

Window: `conda activate carnd`

Linux, MacOS: `source activate carnd`

### 2. Run the pipeline:
```bash
python main.py
```
![](example_images/tool.png)

- The text box on the left is the path to the image or video you wish to process.
- If the text box on the left is the path to a video, you can specify a number on the text box on the right which is the frame of the video you want to extract.
- After you enter the path to the image (or video with a particular frame number), you can then hit the `Load image` button. It will show you multiple images, each belong to a specific stage of the pipeline. Image below is the output of the color-thresholding stage.

![example of fine tuning](example_images/finetuning.gif)

- `Save params` and `Load params` will save all the parameters of the pipeline to a file called `params.pkl`.

# Todo:
- [ ] Improve UI of the application
- [ ] Add more parameters to the pipeline