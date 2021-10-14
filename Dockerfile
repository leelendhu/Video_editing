#Dockerfile, Image, Container
FROM python:3.8

ADD video_sort.py .
COPY shuffled_19.mp4 .

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
#RUN pip install numpy
#RUN pip install argparse



#CMD ["python","./video_sort.py","./shuffled_19.mp4"]