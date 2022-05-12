FROM continuumio/miniconda3
WORKDIR /home
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-liberation\
    libasound2\
    libatk-bridge2.0-0\
    libatk1.0-0\
    libatspi2.0-0\
    libcairo2\
    libcups2\
    libdbus-1-3\
    libdrm2\
    libgbm1\
    libglib2.0-0\
    libgtk-3-0\
    libnspr4\
    libnss3\
    libpango-1.0-0\
    libx11-6\
    libxcb1\
    libxcomposite1\
    libxdamage1\
    libxext6\
    libxfixes3\
    libxrandr2\
    freeglut3-dev\
    xvfb\
    x11-utils\
    unzip
RUN conda update pip -y
RUN conda install -y -c pytorch -c conda-forge bioimageio.core pytorch torchvision cudatoolkit=11.3 cudnn
RUN conda install -y -c conda-forge tensorflow onnxruntime
RUN pip install xarray imjoy-rpc aioprocessing
ADD src/start_bioengine_model_runner.py /home
