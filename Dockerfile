#FROM python:3.6-stretch
FROM nvidia/cuda:9.0-runtime

# Set the working directory to /luminovo-nudity-detection
WORKDIR /face-recognition

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Copy the contents into the image
ADD src/ /face-recognition/src
ADD log/ /face-recognition/logs
ADD data/ /face-recognition/data
ADD Pipfile /face-recognition
ADD Pipfile.lock /face-recognition



# Install python3
RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt install -y python3.6
RUN apt install -y python3.6-dev
RUN apt install wget
RUN apt update


RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.6 get-pip.py
RUN ln -s /usr/bin/python3.6 /usr/local/bin/python3
RUN rm get-pip.py

# Install app dependencies from the Pipfile

RUN pip install --trusted-host pypi.python.org pipenv
RUN /bin/bash -c "pipenv lock -r | cat >> requirements.txt"
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pipenv --rm
RUN rm Pipfile Pipfile.lock requirements.txt

# Expose Ports for Ipython (8888)
EXPOSE 8888

# Run Docker container shell when Docker starts
CMD ["/bin/bash"]