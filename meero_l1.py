
# Import the argparse library
import argparse
import os
import sys
import cv2
import numpy as np
from numpy.core.numeric import Inf

# Create the parser
my_parser = argparse.ArgumentParser(description='Gives the correcte video')

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

def frames_from_video(video_path):
   '''An array of the frames from given video is returned 
    and the frames are saved''' 
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
    neighbours for each image is returned'''
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
    biggest jump'''
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
  the closest L1 neighbour for it and repeating the same '''
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
    images in input array'''
    img=[]
    for i in sorted_array:
        img.append(i)
    height,width,layers=img[0].shape
    video_fin=cv2.VideoWriter(vid_output,cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),15,(width,height))
    for j in range(len(sorted_array)):
        video_fin.write(img[j])
    cv2.destroyAllWindows()
    video_fin.release()

frame_array = frames_from_video(input_video)
averageL1 = average_L1(frame_array)
corrected_array = remove_outliers(averageL1,frame_array)
sorted_array = sort_L1(corrected_array)
path = 'Tunjumbled.mp4'
stitch_video(sorted_array,path)
