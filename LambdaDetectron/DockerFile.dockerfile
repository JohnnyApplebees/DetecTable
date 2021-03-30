ARG FUNCTION_DIR="/usr/src/app"

FROM python:3.8-slim-buster as build-image
ARG FUNCTION_DIR

RUN apt-get update -y

# gcc compiler and opencv prerequisites
RUN apt-get -y install nano git build-essential libglib2.0-0 ffmpeg libsm6 libxext6 libxrender-dev


RUN apt-get install wget cmake libfreetype6-dev pkg-config libfontconfig-dev libjpeg-dev libopenjp2-7-dev -y

RUN pip install pyyaml==5.1 opencv-python pytesseract camelot-py img2pdf PyPDF2 pdf2image pandas poppler-utils

# Development packages
RUN pip install flask flask-cors requests opencv-python
RUN pip install cython
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Detectron2 prerequisites
RUN pip install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Detectron2 - CPU copy
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
RUN pip install -U 'git+https://github.com/facebookresearch/fvcore'
ARG FUNCTION_DIR


RUN pip install \
        --target ${FUNCTION_DIR} \
        awslambdaric
# Create working directory
RUN mkdir -p ${FUNCTION_DIR}
WORKDIR ${FUNCTION_DIR}

# Copy contents
COPY . ${FUNCTION_DIR}

# Set environment variables
ENV HOME=${FUNCTION_DIR}

RUN wget https://poppler.freedesktop.org/poppler-data-0.4.9.tar.gz \
    && tar -xf poppler-data-0.4.9.tar.gz \
    && cd poppler-data-0.4.9 \
    && make install \
    && cd .. \
    && wget https://poppler.freedesktop.org/poppler-20.08.0.tar.xz \
    && tar -xf poppler-20.08.0.tar.xz \
    && cd poppler-20.08.0 \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make \
    && make install \
    && ldconfig

# FROM python:3.8-slim-buster

# Set working directory to function root directory
WORKDIR ${FUNCTION_DIR}

# Copy in the build image dependencies
# COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

ADD https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie /usr/bin/aws-lambda-rie
RUN chmod 755 /usr/bin/aws-lambda-rie

COPY entry.sh /
RUN chmod 755 /entry.sh

RUN git clone https://github.com/nabeel3133/Table-Detection.git /tmp/a
RUN cp -r /tmp/a/. ${FUNCTION_DIR}
RUN pip install -r ${FUNCTION_DIR}/requirements.txt
RUN pip install deskew
# RUN chmod 755 /usr/bin/aws-lambda-rie


ENTRYPOINT [ "/entry.sh" ]
CMD [ "index.handler" ]