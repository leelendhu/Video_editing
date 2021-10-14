__author__ = "Leelendhu Mouli Kothapalli"
__version__ = "1.0.1"
__email__ = "leelendhu1@gmail.com"
# Import the argparse library
import argparse
import os
import sys
#Import the openCV library
import cv2
import numpy as np
from numpy.core.numeric import Inf


def frames_from_video(video_path):
    '''An array containing the individual frames 
    from a video file is returned

                Parameters:
                    frame_array (array): image array of individual frames from the video

                Returns:
                    averageL1 (array): an array containing L1 norm of its 5 nearest neighbours
    '''
    vid=cv2.VideoCapture(video_path)
    i=0
    frame_array = []
    while(vid.isOpened()):
        ret, frame = vid.read()
        if ret == False:
            break
        frame_array.append(frame)
        i+=1
    number_of_frames  = i
    vid.release()
    return frame_array     

def average_L1(frame_array):
    '''An array containing the average L1 norm of 5 nearest 
    neighbours for each image is returned

                Parameters:
                    frame_array (array): image array of individual frames from the video

                Returns:
                    averageL1 (array): an array containing L1 norm of its 5 nearest neighbours
    '''
    averageL1 = list([])
    for i in frame_array:
        L1values = list([])
        for j in frame_array:
            L1 = cv2.norm( i, j, cv2.NORM_L1); 
            L1values.append(L1)
        L1values = np.sort(L1values)
        avg = sum(L1values[1:5])/5
        averageL1.append(avg)
    return averageL1

def remove_outliers(averageL1, frame_array):
    '''Sorts the input array in ascending order and finds the biggest jump 
    between two consecutive values and removes all elements to the right of this 
    biggest jump
            Parameters:
                averageL1 (array): an array containing L1 norm of its 5 nearest neighbours
                frame_array (array): an image array 

            Returns:
                corrected_array (array): an image array with outliers removed

    '''
    averageL1sort = np.sort(averageL1)
    biggestjump = 0
    index_of_biggestjump = 0
    for i in range(len(averageL1sort) - 1):
        jump = averageL1sort[i+1] - averageL1sort[i]
        if jump > biggestjump:
            biggestjump = jump
            index_of_biggestjump = i

    corrected_list = list([])
    threshold = averageL1sort[index_of_biggestjump]
    for i in range(len(averageL1sort)):
        if averageL1[i]<=threshold:
            corrected_list.append(i)      

    corrected_array = []
    for i in corrected_list:
        frame = frame_array[i]
        corrected_array.append(frame)

    return corrected_array

def sort_L1(array):
    '''Sorts an image array taking the first element as pivot and finding
    the closest L1 neighbour and making it the second element and repeating
    the same for second element till the entire array is sorted
            Parameters:
                array (array):an image array starting with the initial frame

            Returns:
                sorted (array):an image array sorted based on closest L1 neighbour
    '''
    i = 0
    sorted = []
    sorted.append(array[i])
    array.pop(i)

    while len(array) > 1:
        nf = Inf
        ind1 = 0
        for j in range(len(array)):
            img1 = sorted[i]
            img2 = array[j]
            d = cv2.norm(img1, img2, cv2.NORM_L1)
            if d < nf:
                nf = d
                ind1 = j
        sorted.append(array[ind1])
        array.pop(ind1)
        i = i+1
    sorted.append(array[0])
    return sorted


def stitch_video(sorted_array, vid_output):
    '''Stores a mp4 file at the give path stitched from the 
    images in input array
                Parameters:
                    sorted_array (array):an image array 
                    vid_output (str): path where unjumbled video will be stored

    '''
    img=[]
    for i in sorted_array:
        img.append(i)
    height,width,layers=img[0].shape
    video_fin=cv2.VideoWriter(vid_output,cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),15,(width,height))
    for j in range(len(sorted_array)):
        video_fin.write(img[j])
    cv2.destroyAllWindows()
    video_fin.release()

if __name__ == "__main__":
    # Create the parser
    my_parser = argparse.ArgumentParser(description='Gives the corrected video')

    # Add the arguments
    my_parser.add_argument('Path',
                        metavar='path',
                        type=str,
                        help='the path to video')

    # Execute the parse_args() method
    args = my_parser.parse_args()

    input_path = args.Path

    if not os.path.isfile(input_path):
        print('The path specified does not exist')
        sys.exit() 
    else:
        input_video = input_path

    frame_array = frames_from_video(input_video)
    averageL1 = average_L1(frame_array)
    corrected_array = remove_outliers(averageL1,frame_array)
    sorted_array = sort_L1(corrected_array)
    output_path = 'unjumbled.mp4'
    stitch_video(sorted_array,output_path)

    print("Completed unjumbling video, available at " + output_path)
