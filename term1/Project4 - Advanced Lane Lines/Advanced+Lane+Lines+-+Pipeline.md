
# The Project - Advanced Lane Lines (Pipeline)


```python
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize
import cv2
import glob
%matplotlib inline
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# prepare object points
nx = 9 # number of inside corners in x
ny = 6 # number of inside corners in y

objpoints = []
imgpoints = []
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

cal_images_set = glob.glob("camera_cal/calibration*.jpg")

def calibrate(image_set, objpoints, imgpoints, objp, img_size):
    
    for cal_image_name in image_set:
        cal_image = cv2.imread(cal_image_name)
        gray_image = cv2.cvtColor(cal_image, cv2.COLOR_BGR2GRAY)
    
        ret, corners = cv2.findChessboardCorners(gray_image, (nx,ny), None)
        if ret == True:
            cal_image = cv2.drawChessboardCorners(cal_image, (nx, ny), corners, ret)
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.imwrite("output_images/" + cal_image_name[11:-4] + "_with_chessboardcorners.jpg", cal_image)
            #plt.imshow(cal_image)
        else: 
            cv2.imwrite("output_images/" + cal_image_name[11:-4] + "_with_no_chessboardcorners.jpg", cal_image)

    sample_image = cv2.imread('camera_cal/calibration2.jpg')
    imshape = sample_image.shape
    img_size = (sample_image.shape[1], sample_image.shape[0])            
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mtx, dist

def apply_threshold(image):
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 25
    thresh_max = 150
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_thresh_min = 75
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary

def get_transform_info(image_shape):
    offset = 100
    img_size = (image_shape[1], image_shape[0])

    src = np.float32(
      [[(img_size[0] / 2) - 55,     img_size[1] / 2 + offset],
      [((img_size[0] / 6) - 10),    img_size[1]],
       [(img_size[0] * 5 / 6) + 60, img_size[1]],
       [(img_size[0] / 2 + 55),     img_size[1] / 2 + offset]])
    dst = np.float32(
      [[(img_size[0] / 4),     0],
       [(img_size[0] / 4),     img_size[1]],
       [(img_size[0] * 3 / 4), img_size[1]],
       [(img_size[0] * 3 / 4), 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    binary_warped = cv2.warpPerspective(combined_image, M, img_size, flags=cv2.INTER_LINEAR)
    return M, Minv, binary_warped

def process_test_frame(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 15
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return out_img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fitx, right_fitx, ploty, leftx, lefty, rightx, righty, left_fit, right_fit


def process_first_frame(binary_warped):
    out_img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fitx, right_fitx, ploty, leftx, lefty, rightx, righty, left_fit, right_fit = process_test_frame(binary_warped)
    return out_img, left_fitx, right_fitx, ploty, left_fit, right_fit

def process_next_frame(binary_warped, left_fit, right_fit):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return out_img, margin, left_fit, right_fit, ploty, left_fitx, right_fitx

def calculate_radius_of_curvature_meters(binary_warped, left_fitx, right_fitx, ploty):
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    y_eval = binary_warped.shape[0]
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Calculate position of vehicle
    left_lane_poly = left_fit_cr[0]*y_eval*ym_per_pix**2 + left_fit_cr[1]*y_eval*ym_per_pix + left_fit_cr[2]
    right_lane_poly = right_fit_cr[0]*y_eval*ym_per_pix**2 + right_fit_cr[1]*y_eval*ym_per_pix + right_fit_cr[2]
    lane_center = (left_lane_poly +  right_lane_poly) / 2
    lane_width = right_lane_poly - left_lane_poly
    vehicle_center = binary_warped.shape[1] * xm_per_pix / 2
    vehicle_position = lane_center - vehicle_center
    
    return left_curverad, right_curverad, vehicle_position, lane_width

#left_curverad, right_curverad, vehicle_position = calculate_radius_of_curvature_meters(binary_warped, left_fitx, right_fitx, ploty)
#print(left_curverad, 'm', right_curverad, 'm', vehicle_position)
```


```python
left_fit = 0
right_fit = 0
ploty = 0
left_fitx = 0
right_fitx = 0
mtx = []
dist = []
out_img = []
lw = []

# The guts of finding lane lines and tracking    
def track_lane_lines_pipeline(image, image_count):
    global left_fit
    global right_fit
    global ploty
    global left_fitx
    global right_fitx
    global mtx
    global dist
    global out_img
    image_shape = image.shape
    
    objpoints = []
    imgpoints = []
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    
    offset = 100
    img_size = (image_shape[1], image_shape[0])

    src = np.float32(
      [[(img_size[0] / 2) - 55,     img_size[1] / 2 + offset],
      [((img_size[0] / 6) - 10),    img_size[1]],
       [(img_size[0] * 5 / 6) + 60, img_size[1]],
       [(img_size[0] / 2 + 55),     img_size[1] / 2 + offset]])
    dst = np.float32(
      [[(img_size[0] / 4),     0],
       [(img_size[0] / 4),     img_size[1]],
       [(img_size[0] * 3 / 4), img_size[1]],
       [(img_size[0] * 3 / 4), 0]])    

    if image_count == 0:
        img_size = (image.shape[1], image.shape[0])
        mtx, dist = calibrate(cal_images_set, objpoints, imgpoints, objp, img_size)    
    
    undistorted = cv2.undistort(image, mtx, dist, None, mtx)
    combined = apply_threshold(undistorted)
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    binary_warped = cv2.warpPerspective(combined, M, (image_shape[1], image_shape[0]), flags=cv2.INTER_LINEAR)
    
    if image_count == 0:
        out_img, left_fitx, right_fitx, ploty, left_fit, right_fit = process_first_frame(binary_warped)
        print("Processed first frame")
    else:
        out_img, margin, left_fit, right_fit, ploty, left_fitx, right_fitx = process_next_frame(binary_warped, left_fit, right_fit)

    left_curverad, right_curverad, vehicle_position, lane_width = calculate_radius_of_curvature_meters(binary_warped, left_fitx, right_fitx, ploty)
    
    if lane_width > 3.936 or lane_width < 2.624:
        #print(lane_width,"m", image_count)
        out_img, left_fitx, right_fitx, ploty, left_fit, right_fit = process_first_frame(binary_warped)
        left_curverad, right_curverad, vehicle_position, lane_width = calculate_radius_of_curvature_meters(binary_warped, left_fitx, right_fitx, ploty)
        if lane_width > 3.936 or lane_width < 2.624:
            lw.append(lane_width)
   
    
    # Now our radius of curvature is in meters
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image_shape[1], image_shape[0]), flags=cv2.INTER_LINEAR) 
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    # using cv2 for drawing text in diagnostic pipeline.
    position_of_vehicle = "left" if vehicle_position < 0 else "right"

    font = cv2.FONT_HERSHEY_PLAIN
    middlepanel = np.zeros((360, 1280, 3), dtype=np.uint8)
    cv2.putText(middlepanel, 'Radius of Curvature (Left) = %dm' % left_curverad, (30, 60), font, 3, (255,0,0), 2)
    cv2.putText(middlepanel, 'Radius of Curvature (Right) = %dm' % right_curverad, (30, 100), font, 3, (255,0,0), 2)
    cv2.putText(middlepanel, 'Vehicle is %.2fm %s of center' % (np.abs(vehicle_position), position_of_vehicle), (30, 140), font, 3, (255,0,0), 2)
    cv2.putText(middlepanel, 'Lane Width %.2fm' % (lane_width), (30, 180), font, 3, (255,0,0), 2)
    cv2.putText(middlepanel, 'Frame Count %d' % image_count, (30, 220), font, 3, (255,0,0), 2)

    # from the forums that my 1st reviewer pointed me to for debugging
    # assemble the screen example
    diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    diagScreen[0:720, 0:1280] = result
    diagScreen[0:240, 1280:1600] = cv2.resize(image, (320,240), interpolation=cv2.INTER_AREA) 
    diagScreen[0:240, 1600:1920] = cv2.resize(undistorted, (320,240), interpolation=cv2.INTER_AREA)
    diagScreen[240:480, 1280:1600] = cv2.resize(color_warp, (320,240), interpolation=cv2.INTER_AREA)
    diagScreen[240:480, 1600:1920] = cv2.resize(newwarp, (320,240), interpolation=cv2.INTER_AREA)*4
    diagScreen[600:1080, 1280:1920] = cv2.resize(out_img, (640,480), interpolation=cv2.INTER_AREA)*4
    diagScreen[720:1080, 0:1280] = middlepanel

    return diagScreen
```


```python
image_count = 0
def process_image(image):
    global image_count
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    result = track_lane_lines_pipeline(image, image_count)
    image_count += 1
    return result
```


```python
video_output = 'project_video_output.mp4'
video_input = "project_video.mp4"
#video_output = 'challenge_video_output.mp4'
#video_input = "challenge_video.mp4"

#challenge_video.mp4
clip1 = VideoFileClip(video_input)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(video_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(video_output))
```

    Processed first frame
    [MoviePy] >>>> Building video project_video_output.mp4
    [MoviePy] Writing video project_video_output.mp4
    

    100%|█████████████████████████████████████▉| 1260/1261 [04:44<00:00,  4.62it/s]
    

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: project_video_output.mp4 
    
    Wall time: 4min 48s
    





<video width="960" height="540" controls>
  <source src="project_video_output.mp4">
</video>




# Discussion

The main problem I ran into was that I should have started with implementing a rough pipeline after completing the individual stages. I spent too much time trying to perfect which values to use for the tuning parameters (thresholds, margin, nwindows, etc.).

My pipeline performs well on the project video, but not as well on the other two videos where the lines may not be as prevalent or too curvy. Other areas where the pipeline would likely need tuning is scenarios where the contrast is hard to distinguish. Examples would be stormy weather with heavy rain downpour, extremely foggy near coastal areas, heavy snow or dimly lit roads at night.

Areas to improve my pipeline, I would focus on different theshold techniques with Sobel or other methods not discussed in the lectures or the convolution method for processing the images.

## Follow-up discussion

Improvements made after the first submission:
- In Project4 - Individual Components, I corrected the displaying of images from BGR to RGB.
- Corrected curvature of radius method by breaking calculation into pieces and properly using 2nd order polynomial, center of image on horizontal axis and lane center
- Added diag window thanks to 1st reviewer feedback
- Corrected extreme values for frames though could still be improved. I did manage to reduce it, but fundamentally I would need to do more work on apply_threshold method because it comes down to noise in the Sobel image. Within my pipeline, I took the measurement of the lane width and took the mean of at the samples to arrive at 3.28m (although it should be 3.7m). From there I added a buffer that if it's between .8 and 1.2 of the mean to continue in the pipeline. Otherwise to reprocess the sample completely for the fit.


```python

```
