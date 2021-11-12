FROM nvcr.io/nvidia/pytorch:21.10-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends mc git zbar-tools

RUN git clone https://github.com/karinaodm/OptimalGradCheckpointing.git && \
    cd OptimalGradCheckpointing && \
    python3 setup.py sdist && \
    python3 setup.py install

WORKDIR /working_dir/
