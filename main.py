# Import Statement
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import os
import shutil
import random
from joblib import Parallel, delayed
import time
import datetime

print("Program: Object Detection")
print("Release: 0.1.0")
print("Date: 2020-02-02")
print("Author: Brian Neely")
print()
print()
print("This program reads picture and videos from the input folder and runs an object detection program on them.")
print()
print()


def object_recognition_image(image_file_path):
    im = cv2.imread(image_file_path)
    bbox, label, conf = cv.detect_common_objects(im)
    output_image = draw_bbox(im, bbox, label, conf)
    return output_image


def chop_microseconds(delta):
    return delta - datetime.timedelta(microseconds=delta.microseconds)


def video_object_recognition(video, outpt_fldr, temp_fldr_base):
    # Get file names and extensions
    file_name, extension = os.path.splitext(video)
    file_name = os.path.basename(file_name)

    # Create a temp folder for the split video files (1) and processed images (2)
    rand = random.randrange(0, 1000000000000000, 1)
    temp_1_fldr = os.path.join(temp_fldr_base, "temp_1 - " + str(rand))
    temp_2_fldr = os.path.join(temp_fldr_base, "temp_2 - " + str(rand))
    # If folder exists, delete it.
    if os.path.exists(temp_1_fldr):
        shutil.rmtree(temp_1_fldr)
    if os.path.exists(temp_2_fldr):
        shutil.rmtree(temp_2_fldr)
    # Create new temp folder
    os.mkdir(temp_1_fldr)
    os.mkdir(temp_2_fldr)

    # Create a list of the frames in order
    frame_list = list()
    frame_list_processed = list()

    print("Splitting video file [" + video + "] into individual images...")

    # Capture video
    vidcap = cv2.VideoCapture(video)

    # Read the first frame
    success, image = vidcap.read()
    frame = 0

    # While continuing to get new frames, read each one
    while success:
        image_out_name = os.path.join(temp_1_fldr, file_name + "_" + str(frame) + extension + ".jpg")
        cv2.imwrite(image_out_name, image)
        frame_list.append(image_out_name)
        success, image = vidcap.read()
        frame += 1
    print("Video file [" + video + "] into split into individual images!")

    # Process images and save to the processed image temp folder
    num_frames = len(frame_list)
    start_time = time.time()
    print("Processing individual images for " + video + "...")
    for index, image_file in enumerate(frame_list):
        processed_frame = object_recognition_image(image_file)
        processed_image_name = os.path.basename(image_file)
        processed_image_out = os.path.join(temp_2_fldr, processed_image_name)
        cv2.imwrite(processed_image_out, processed_frame)
        frame_list_processed.append(processed_image_out)
        # Print for frames processed and est time
        if index % 100 == 0 and index > 0:
            fps_process = index / (time.time() - start_time)
            remaining_frames = num_frames - index
            est_time = chop_microseconds(datetime.timedelta(seconds=remaining_frames / fps_process))
            statement = "{}: Completed {}th frame out of {}. FPS: {}. Estimated Time Remaining: {}"
            print(statement.format(file_name, str(index), str(num_frames), str(round(fps_process, 2)), str(est_time)))
    print("Individual images processed for " + video + "!")

    # Get image frame size
    print("Creating video for " + video + "...")
    img = cv2.imread(frame_list_processed[0])
    height, width, layers = img.shape
    size = (width, height)

    # Get FPS
    fps = frames_per_second(video)

    # Convert images to video
    video_out = cv2.VideoWriter(os.path.join(outpt_fldr, os.path.basename(video)), cv2.VideoWriter_fourcc(*'DIVX'), fps,
                                size)
    for j in frame_list_processed:
        img = cv2.imread(j)
        video_out.write(img)
    video_out.release()
    print("Video created for " + video + " and saved as " + os.path.join(outpt_fldr, video) + "!")

    # Delete the temp folder if it exists
    if os.path.exists(temp_1_fldr):
        shutil.rmtree(temp_1_fldr)
    if os.path.exists(temp_2_fldr):
        shutil.rmtree(temp_2_fldr)


def frames_per_second(video):
    # Get Frames per Second of the movie

    # Start camera
    video_cap = cv2.VideoCapture(video)

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')

    if int(major_ver) < 3:
        fps = video_cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    return fps


# Specify folder names
inpt_fldr = "input"
outpt_fldr = "output"

# If input folder doesn't exist, create it and give error.
if not os.path.exists(inpt_fldr):
    print()
    print("Input folder doesn't exist. Create input folder now...")
    os.mkdir(inpt_fldr)
    print("Input folder created as " + inpt_fldr + ". Please put input files into there and reset.")
    input("Press Enter to continue.")
    exit()

# If output folder doesn't exist, create it.
if not os.path.exists(outpt_fldr):
    print()
    print("Output folder doesn't exist. Create input folder now...")
    os.mkdir(outpt_fldr)
    print()

# List of all accepted video file formats
video_formats = {'.MP4', '.MKV', '.AVI'}

# List of all accepted image file formats
image_formats = {'.JPEG', '.JPG'}

# Get files from input folder
video_path_list = list()
image_path_list = list()
for (dirpath, dirnames, filenames) in os.walk('input'):
    if len(filenames) != 0:
        for i in filenames:
            file = os.path.join(os.path.relpath(dirpath, inpt_fldr), i)
            # Test if file is a video format
            for j in video_formats:
                if file[-len(j):] == str.lower(j):
                    video_path_list.append(os.path.join(inpt_fldr, file))
            # Test if file is a image format
            for j in image_formats:
                if file[-len(j):] == str.lower(j):
                    image_path_list.append(os.path.join(inpt_fldr, file))
print("Found " + str(len(video_path_list)) + " videos!")
print("Found " + str(len(image_path_list)) + " pictures!")

# Run image recognition
for i in image_path_list:
    processed_image = object_recognition_image(i)
    processed_image_name = os.path.basename(i)
    processed_image_out = os.path.join(outpt_fldr, processed_image_name)
    cv2.imwrite(processed_image_out, processed_image)

# Make temp folder
if os.path.exists("temp"):
    print("Deleting old temp folder...")
    shutil.rmtree("temp")
    print("Old temp folder deleted!")
    print()
time.sleep(2)
print("Creating temporary folder...")
os.mkdir("temp")

# Process Videos
print("Running processing on videos...")
Parallel(n_jobs=16)(delayed(video_object_recognition)(i, outpt_fldr, "temp") for i in video_path_list)

# Delete temp folder
if os.path.exists("temp"):
    shutil.rmtree("temp")
