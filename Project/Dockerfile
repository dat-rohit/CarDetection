From tensorflow/tensorflow:latest-gpu-jupyter

WORKDIR /yolo

RUN apt-get update && apt-get install -y \
  ffmpeg \
  wget \
  && rm -rf /var/lib/apt/lists/*


# Python Dependencies

COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 443


