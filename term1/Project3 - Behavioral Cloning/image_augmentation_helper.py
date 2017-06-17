import cv2
import numpy as np
import csv

def read_driving_info_from_file(filename):
    X_center = []
    Y_center = []
    X_left = []
    Y_left = []
    X_right = []
    Y_right = []
    correction = 0.25

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            X_center.append("data/" + row[0].strip())
            Y_center.append(float(row[3]))
            X_left.append("data/" + row[1].strip()) #left
            Y_left.append(float(row[3]) + correction)
            X_right.append("data/" + row[2].strip()) #right
            Y_right.append(float(row[3]) - correction)
    return list(zip(X_center,Y_center)), list(zip(X_left,Y_left)), list(zip(X_right,Y_right))
	
def flip_image(input_filename, output_filename):
    image_data = cv2.imread(input_filename)
    cv2.imwrite(output_filename, cv2.flip(image_data, 1))
    
def morph_gradient_image(input_filename, output_filename):
    image_data = cv2.imread(input_filename)
    cv2.imwrite(output_filename, cv2.morphologyEx(image_data, cv2.MORPH_GRADIENT, np.ones((5,5),np.uint8)))    

def blur_image(input_filename, output_filename):
    image_data = cv2.imread(input_filename)
    kernel = np.ones((5,5),np.float32)/25
    cv2.imwrite(output_filename, cv2.blur(image_data,(5,5)))    

def binary_threshold_image(input_filename, output_filename):
    image_data = cv2.imread(input_filename)
    ret,thresh = cv2.threshold(image_data,127,255,cv2.THRESH_BINARY)
    cv2.imwrite(output_filename, thresh) 
    
def tozero_inv_threshold_image(input_filename, output_filename):
    image_data = cv2.imread(input_filename)
    ret,thresh = cv2.threshold(image_data,127,255,cv2.THRESH_TOZERO_INV)
    cv2.imwrite(output_filename, thresh) 
    
def grabcut_image(input_filename, output_filename):
    image_data = cv2.imread(input_filename)
    mask = np.zeros(image_data.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    rect = (0,50,450,290)
    cv2.grabCut(image_data,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    image_data = image_data*mask2[:,:,np.newaxis]    
    cv2.imwrite(output_filename, image_data)    

def erode_image(input_filename, output_filename):
    image_data = cv2.imread(input_filename)
    cv2.imwrite(output_filename, cv2.erode(image_data,np.ones((5,5),np.uint8),iterations = 5))

def truncate_threshold_image(input_filename, output_filename):
    image_data = cv2.imread(input_filename)
    ret,thresh = cv2.threshold(image_data,127,255,cv2.THRESH_TRUNC)
    cv2.imwrite(output_filename, thresh)

def generate_blur_data(image_list, input_dir="IMG", output_dir="data/IMG_generated/"):
    blur = []

    for row in image_list:
        pos = row[0].find(input_dir)
        blur_name = output_dir + "blur_" + row[0][1 + pos + len(input_dir):]
        blur.append((blur_name, row[1]))
        if -1 != row[0].find("filtered"):
            blur_image(row[0], blur_name)
        else:     
            blur_image(row[0].strip(), blur_name)
    return blur    
    
def generate_morph_data(image_list, input_dir="IMG", output_dir="data/IMG_generated/"):
    morph = []

    for row in image_list:
        pos = row[0].find(input_dir)
        morph_name = output_dir + "morph_" + row[0][1 + pos + len(input_dir):]
        morph.append((morph_name, row[1]))
        if -1 != row[0].find("filtered"):
            morph_gradient_image(row[0], morph_name)
        else:     
            morph_gradient_image(row[0].strip(), morph_name)
    return morph        
    
def generate_tozero_inv_data(image_list, input_dir="IMG", output_dir="data/IMG_generated/"):
    tozero_inv = []

    for row in image_list:
        pos = row[0].find(input_dir)
        tozero_inv_name = output_dir + "tozero_inv_" + row[0][1 + pos + len(input_dir):]
        tozero_inv.append((tozero_inv_name, row[1]))
        if -1 != row[0].find("filtered"):
            tozero_inv_threshold_image(row[0], tozero_inv_name)
        else:     
            tozero_inv_threshold_image(row[0].strip(), tozero_inv_name)
    return tozero_inv    
    
def generate_binary_data(image_list, input_dir="IMG", output_dir="data/IMG_generated/"):
    binary = []

    for row in image_list:
        pos = row[0].find(input_dir)
        binary_name = output_dir + "binary_" + row[0][1 + pos + len(input_dir):]
        binary.append((binary_name, row[1]))
        if -1 != row[0].find("filtered"):
            binary_threshold_image(row[0], binary_name)
        else:     
            binary_threshold_image(row[0].strip(), binary_name)
    return binary    
    
def generate_grabcut_data(image_list, input_dir="IMG", output_dir="data/IMG_generated/"):
    grabcut = []

    for row in image_list:
        pos = row[0].find(input_dir)
        grabcut_name = output_dir + "grabcut_" + row[0][1 + pos + len(input_dir):]
        grabcut.append((grabcut_name, row[1]))
        if -1 != row[0].find("filtered"):
            grabcut_image(row[0], grabcut_name)
        else:     
            grabcut_image(row[0].strip(), grabcut_name)
    return grabcut    
    
def generate_truncated_data(image_list, input_dir="IMG", output_dir="data/IMG_generated/"):
    truncate = []

    for row in image_list:
        pos = row[0].find(input_dir)
        truncate_name = output_dir + "truncate_" + row[0][1 + pos + len(input_dir):]
        truncate.append((truncate_name, row[1]))
        if -1 != row[0].find("filtered"):
            truncate_threshold_image(row[0], truncate_name)
        else:     
            truncate_threshold_image(row[0].strip(), truncate_name)
    return truncate

def generate_erode_data(image_list, input_dir="IMG", output_dir="data/IMG_generated/"):
    erode = []

    for row in image_list:
        pos = row[0].find(input_dir)
        erode_name = output_dir + "erode_" + row[0][1 + pos + len(input_dir):]
        erode.append((erode_name, row[1]))
        if -1 != row[0].find("filtered"):
            erode_image(row[0], erode_name)
        else:     
            erode_image(row[0].strip(), erode_name)
    return erode

def generate_flipped_data(image_list, input_dir="IMG", output_dir="data/IMG_generated/"):
    flipped = []

    for row in image_list:
        pos = row[0].find(input_dir)
        flipped_name = output_dir + "flipped_" + row[0][(pos+4):]
        flipped.append((flipped_name, -1 * row[1]))
        flip_image(row[0], flipped_name)
    return flipped

def filter_out_minimal_steering_angles(image_list, remove_angles_less_than=0.0005): # was 0.1
    filtered_ImageData = []
    for row in image_list:
        if (abs(row[1]) > remove_angles_less_than):
            filtered_ImageData.append((row[0],row[1]))
    return filtered_ImageData

def generate_left_recovery_data(image_list):
    filtered_ImageData = []
    for row in image_list:
        if (row[1] > 0.25): # shouldn't matter we filter out 0 also
            filtered_ImageData.append((row[0],row[1]))
    return filtered_ImageData

def generate_right_recovery_data(image_list):
    filtered_ImageData = []
    for row in image_list:
        if (row[1] < -0.25): # shouldn't matter we filter out 0 also
            filtered_ImageData.append((row[0],row[1]))
    return filtered_ImageData
	
def generate_augmented_data(all_data):
    augmented_ImageData = []
    
    print("Started Image Generation!")  
    print("Generating flipped data...")
    flipped_all_data = generate_flipped_data(all_data)

    print("Adding original and flipped data to master list...")
    augmented_ImageData.extend(all_data) 
    augmented_ImageData.extend(flipped_all_data) 
    print("augmented_ImageData has %d samples" % len(augmented_ImageData))

    print("* Generating blur data from filtered data...")
    all_data_blur = generate_blur_data(all_data)
    augmented_ImageData.extend(all_data_blur) 
    print("augmented_ImageData has %d samples" % len(augmented_ImageData))

    print("* Generating binary data from filtered data...")
    all_data_binary = generate_binary_data(all_data)
    augmented_ImageData.extend(all_data_binary) 
    print("augmented_ImageData has %d samples" % len(augmented_ImageData))

    print("* Generating erode data from filtered data...")
    all_data_eroded = generate_erode_data(all_data)
    augmented_ImageData.extend(all_data_eroded) 
    print("augmented_ImageData has %d samples" % len(augmented_ImageData))

    print("* Generating morph data from filtered data...")
    all_data_morph = generate_morph_data(all_data)
    augmented_ImageData.extend(all_data_morph) 
    print("augmented_ImageData has %d samples" % len(augmented_ImageData))

    print("* Generating tozero inv data from filtered data...")
    all_data_tozero_inv = generate_tozero_inv_data(all_data)
    augmented_ImageData.extend(all_data_tozero_inv) 
    print("augmented_ImageData has %d samples" % len(augmented_ImageData))

    print("* Generating truncated data from filtered data...")
    all_data_truncated = generate_truncated_data(all_data)
    augmented_ImageData.extend(all_data_truncated) 
    print("augmented_ImageData has %d samples" % len(augmented_ImageData))

    #print("* Filtering out minimal steering angles...")
    #flipped_all_data_filtered = filter_out_minimal_steering_angles(flipped_all_data)

    print("* Generating flipped blur data from flipped filtered data...")
    flipped_filtered_blur = generate_blur_data(flipped_all_data, input_dir="IMG_generated")
    augmented_ImageData.extend(flipped_filtered_blur) 
    print("augmented_ImageData has %d samples" % len(augmented_ImageData))

    print("* Generating flipped binary data from flipped filtered data...")
    flipped_filtered_binary = generate_binary_data(flipped_all_data, input_dir="IMG_generated")
    augmented_ImageData.extend(flipped_filtered_binary) 
    print("augmented_ImageData has %d samples" % len(augmented_ImageData))

    print("* Generating flipped erode data from flipped filtered data...")
    flipped_filtered_eroded = generate_erode_data(flipped_all_data, input_dir="IMG_generated")
    augmented_ImageData.extend(flipped_filtered_eroded) 
    print("augmented_ImageData has %d samples" % len(augmented_ImageData))

    print("* Generating flipped morph data from flipped filtered data...")
    flipped_filtered_morph = generate_morph_data(flipped_all_data, input_dir="IMG_generated")
    augmented_ImageData.extend(flipped_filtered_morph) 
    print("augmented_ImageData has %d samples" % len(augmented_ImageData))

    print("* Generating flipped tozero inv data from flipped filtered data...")
    flipped_filtered_tozero_inv = generate_tozero_inv_data(flipped_all_data, input_dir="IMG_generated")
    augmented_ImageData.extend(flipped_filtered_tozero_inv) 
    print("augmented_ImageData has %d samples" % len(augmented_ImageData))

    print("* Generating flipped truncated data from flipped filtered data...")
    flipped_filtered_truncated = generate_truncated_data(flipped_all_data, input_dir="IMG_generated")
    augmented_ImageData.extend(flipped_filtered_truncated) 

    print("Completed Image Generation!")   
    print("augmented_ImageData has %d samples" % len(augmented_ImageData))	
    return augmented_ImageData
