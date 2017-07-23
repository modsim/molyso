FROM continuumio/anaconda3
LABEL maintainer c.sachs@fz-juelich.de
RUN apt-get update && apt-get install -y gcc libgomp1 unzip libgl1-mesa-glx && \
    # opengl for qt...
    apt-get clean && \
    #conda install -y numpy scipy opencv matplotlib jupyter pandas && \
    conda install -y opencv && \
    pip install "git+https://github.com/modsim/molyso#egg=molyso" && \
    adduser --disabled-password user && \
    mkdir /data /examples && \
    wget https://github.com/modsim/molyso/archive/master.zip -O /tmp/molyso.zip && \ 
    unzip -j /tmp/molyso.zip 'molyso-master/examples/*' -d /examples && \
    rm /tmp/molyso.zip && \
    apt-get remove -y gcc unzip && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /examples /home/user && \
    chown -R user:user /data /examples /home/user && \
    runuser user -s /bin/sh "-c jupyter notebook --generate-config" && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /home/user/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.notebook_dir = '/home/user'" >> /home/user/.jupyter/jupyter_notebook_config.py && \   
    echo Done

USER user

WORKDIR /data

EXPOSE 8888

ENTRYPOINT ["python", "-m", "molyso"]
