FROM continuumio/miniconda3:4.7.12-alpine
LABEL maintainer=c.sachs@fz-juelich.de

USER root

ENV PATH "$PATH:/opt/conda/bin:/bin/sbin:/usr/bin"

# we could copy the files from the current directory, but that would create an additional layer ...
COPY . /tmp/package

WORKDIR /tmp/package

RUN \
    # conda build needs bash
    apk add --no-cache bash mesa-gl && \
    conda config --add channels conda-forge --add channels bioconda --add channels csachs && \
    conda install -y conda-build conda-verify && \
    conda build recipe && \
    conda install -c local -y molyso && \
    conda install -y jupyter pandas && \
    # missing libGL causes cv2 to fail in a very pecuilar way ... although it just throws an ImportError
    # molyso will randomly crash later when importing matplotlib.pyplot
    cp -R  /tmp/package/examples /examples && \
    busybox adduser --disabled-password user && \
    ln -s /examples /home/user && \
    mkdir /data && \
    chown -R user:users /data /examples /home/user && \
    su -s /bin/sh user -c "jupyter notebook --generate-config" && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /home/user/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.notebook_dir = '/home/user'" >> /home/user/.jupyter/jupyter_notebook_config.py && \
    conda clean -afy || true && \
    conda build purge-all && \
    echo Done

USER user

WORKDIR /data

EXPOSE 8888

ENTRYPOINT ["python", "-m", "molyso"]
