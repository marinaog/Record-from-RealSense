This repository has the purpose of recording a SLAM dataset that includes RAW images, sRGB images, depth, imu together with a pose estimation groun truth estimated with motion trackers. 

It contains two main parts:

### Camera recording with RealSense
Record camera images (RAW, RGB, sRGB), depth and imu with option of simultaneously displaying them. It will also automatically accept the combination of sensors among two RealSense devices. Currently prepared for:
* **OPTION A:** RealSense D435i (contains all sensors)
* **OPTION B:** RealSense D435 (RAW image and depth) and L515 (imu).
  
Sensors are recorded + post processed automatically to obtain all the data.

In order to start recording you just need to run `python Record/record.py`.


### Ground truth data
As the motion tracker records the poses in an independent device, it is necessary to synchronize the frames in the camera with the frames from the motion tracker. This requires two steps:

1. Plot the height of the calibration object respect to time and manually select the first frame in which it appears stationary.
2. Inspect the camera frames until the calibration object appears stationary.
3. Add the files path and camera and motion tracker frames in [CreateGT/gt.py](CreateGT/gt.py) and run it with `python CreateGT/gt.py`.
