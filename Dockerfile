FROM continuumio/miniconda3:4.7.10-alpine
LABEL maintainer c.sachs@fz-juelich.de

USER root

ENV PATH "$PATH:/opt/conda/bin:/bin/sbin:/usr/bin"

# we could copy the files from the current directory, but that would create an additional layer ...
# COPY . /tmp/package

RUN conda config --add channels conda-forge --add channels bioconda --add channels csachs && \
    conda install -y molyso jupyter pandas && \
    pip uninstall -y molyso && \
    # missing libGL causes cv2 to fail in a very pecuilar way ... although it just throws an ImportError
    # molyso will randomly crash later when importing matplotlib.pyplot
    apk add --no-cache mesa-gl && \
    wget https://github.com/modsim/molyso/archive/master.tar.gz -O - | tar zx -C /tmp && mv /tmp/molyso-master /tmp/package && \
    pip install /tmp/package && \
    mv /tmp/package/examples / && \
    rm -rf /tmp/package && \
    busybox adduser --disabled-password user && \
    ln -s /examples /home/user && \
    mkdir /data && \
    chown -R user:users /data /examples /home/user && \
    su -s /bin/sh user -c "jupyter notebook --generate-config" && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /home/user/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.notebook_dir = '/home/user'" >> /home/user/.jupyter/jupyter_notebook_config.py && \
    conda clean -afy || true && \
    echo Done

USER user

WORKDIR /data

EXPOSE 8888

ENTRYPOINT ["python", "-m", "molyso"]
