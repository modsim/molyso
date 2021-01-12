FROM continuumio/miniconda3:4.9.2-alpine
LABEL maintainer=c.sachs@fz-juelich.de

USER root

ENV PATH "$PATH:/opt/conda/bin:/bin/sbin:/usr/bin"

COPY . /tmp/package

WORKDIR /tmp/package

RUN \
    # conda build needs bash
    apk add --no-cache bash mesa-gl && \
    conda install -c conda-forge conda-build conda-verify && \
    conda build -c conda-forge -c modsim /tmp/package/recipe && \
    conda install -c conda-forge -c modsim -c local -y python=3.7 molyso jupyter pandas && \
    # missing libGL causes cv2 to fail in a very pecuilar way ... although it just throws an ImportError
    # molyso will randomly crash later when importing matplotlib.pyplot
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
    # conda build purge-all && \ # keep the package in the Docker image
    echo Done

USER user

WORKDIR /data

EXPOSE 8888

ENTRYPOINT ["python", "-m", "molyso"]
