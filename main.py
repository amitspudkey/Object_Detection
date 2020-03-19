# Import Statement
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import os
import shutil
from joblib import Parallel, delayed
import time
import datetime
import moviepy.editor as mpe
import pandas as pd
from selection import *
from file_handling import *
import random

print("Program: Object Detection")
print("Release: 0.2.0")
print("Date: 2020-03-19")
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
    return output_image, label


def chop_microseconds(delta):
    return delta - datetime.timedelta(microseconds=delta.microseconds)


def video_object_recognition(video, outpt_fldr, temp_fldr_base, video_creation, sampling, sampling_rate):
    try:
        # Sleep random amount to start due to multithreading
        time.sleep(random.random() * 5)

        # Get file names and extensions
        file_name, extension = os.path.splitext(video)
        file_name = os.path.basename(file_name)

        # Create a temp folder for the split video files (1) and processed images (2)
        # Check for existing temp_folder with the same name
        temp_folder_name_used = True
        while temp_folder_name_used:
            rand = random.randrange(0, 1000000000000000, 1)
            temp_1_fldr = os.path.join(temp_fldr_base, "temp_1 - " + str(rand))
            temp_2_fldr = os.path.join(temp_fldr_base, "temp_2 - " + str(rand))
            if not os.path.exists(temp_1_fldr):
                temp_folder_name_used = False
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
            # print(str(rand_int) + " : " + str(sampling_rate))
            image_out_name = os.path.join(temp_1_fldr, file_name + "_" + str(frame) + extension + ".jpg")
            cv2.imwrite(image_out_name, image)
            frame_list.append(image_out_name)
            success, image = vidcap.read()
            frame += 1
        print("Video file [" + video + "] into split into individual images!")

        # If sampling selected, remove frames from frame list based on sampling amount
        if sampling:
            frame_list_keep = list()
            for i in frame_list:
                if random.randint(1, 100) <= sampling_rate:
                    frame_list_keep.append(i)
            frame_list = frame_list_keep

        # Process images and save to the processed image temp folder
        num_frames = len(frame_list)
        start_time = time.time()
        label_list = list()
        print("Processing individual images for " + video + "...")
        for index, image_file in enumerate(frame_list):
            frame_tuple = object_recognition_image(image_file)
            processed_frame = frame_tuple[0]
            label_list.append(frame_tuple[1])
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

        # If video_creation specified, combine the video
        if video_creation:
            # Get image frame size
            print("Creating video for " + video + "...")
            img = cv2.imread(frame_list_processed[0])
            height, width, layers = img.shape
            size = (width, height)

            # Get FPS
            fps = frames_per_second(video)

            # Convert images to intermediate video
            intermediate_video_file = os.path.join(temp_2_fldr, os.path.basename(video))
            video_out = cv2.VideoWriter(intermediate_video_file, cv2.VideoWriter_fourcc(*'DIVX'), fps,
                                        size)
            for j in frame_list_processed:
                img = cv2.imread(j)
                video_out.write(img)
            video_out.release()

            # Create video with sound
            video_clip = mpe.VideoFileClip(intermediate_video_file)
            audio_clip = mpe.AudioFileClip(video)
            video_clip.write_videofile(os.path.join(outpt_fldr, os.path.basename(video)), audio=audio_clip, fps=fps)

            print("Video created for " + video + " and saved as " + os.path.join(outpt_fldr, os.path.basename(video)) + "!")

        # Convert the label list into a DF
        # *Columns = number of objects in video
        # *Each row is a frame
        frame_label_dict_list = list()
        for i in label_list:
            frame_label_dict = dict()
            for j in i:
                if j not in frame_label_dict.keys():
                    frame_label_dict[j] = 1
                else:
                    frame_label_dict[j] += 1

            # Append frame dictionary to list
            frame_label_dict_list.append(frame_label_dict)

        # Convert into dataframe
        label_df_out = pd.DataFrame(frame_label_dict_list)

        # Append name of video to dataframe
        label_df_out['video'] = video

        # Create frame_list from index
        label_df_out['frame'] = label_df_out.index

        # Try to delete the temp folder
        try:
            if os.path.exists(temp_1_fldr):
                shutil.rmtree(temp_1_fldr)
            if os.path.exists(temp_2_fldr):
                shutil.rmtree(temp_2_fldr)
        except:
            print("Couldn't delete temp folder...")

        # Return the frame list to original call
        return label_df_out

    except:
        print('Cannot perform object recognition.')


def frames_per_second(video):
    # Get Frames per Second of the movie

    # Start camera
    video_cap = cv2.VideoCapture(video)

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')

    # Get FPS
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

# Program run options
program_options = ['Video Creation', 'Video Creation with Label Array', 'Label Array', 'Sampled Label Array']
selected_use = list_selection(program_options, 'Select use for object detection', 'use')

# Save labels
if selected_use.find('Array') != -1:
    save_video_labels = True

    # Select a output file
    video_labels_path = select_file_out_csv()

    # Try to delete the existing file if can't, throw error and select another file
    while not delete_file(video_labels_path):
        print("Existing file cannot be deleted. Please select another file...")
        video_labels_path = select_file_out_csv()
else:
    save_video_labels = False

# Sampling
if selected_use.find('Sampled') != -1:
    sampling = True
    sampling_rate = input_int("Input Sampling rate between 1 to 99% (int)", 1, 99)
else:
    sampling = False
    sampling_rate = int(100)

# Video Creation
if selected_use.find('Video') != -1:
    video_creation = True
else:
    video_creation = False

# List of all accepted video file formats
video_formats = {'.MP4', '.MKV', '.AVI', '.WEBM'}

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
    image_tuple = object_recognition_image(i)
    processed_image = image_tuple[0]
    image_label = image_tuple[1]
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

# Process Videos if there are videos
if len(video_path_list) >= 1:
    print("Running processing on videos...")
    object_info = Parallel(n_jobs=1)(delayed(video_object_recognition)(i, outpt_fldr, "temp", video_creation, sampling, sampling_rate) for i in video_path_list)

    # Steps to save the video labels
    if save_video_labels:
        # Concat dataframes together
        object_info = pd.concat(object_info)

        # Save output as csv
        object_info.to_csv(video_labels_path, index=False)

# Delete temp folder
if os.path.exists("temp"):
    shutil.rmtree("temp")
