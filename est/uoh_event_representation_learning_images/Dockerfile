FROM nvidia/cuda:10.1-base-ubuntu18.04
CMD nvidia-smi

# Use an official pytorch runtime as a parent image
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
CMD ["python"]

RUN apt-get update
#RUN apt-get install libgl1-mesa-glx libjpeg62 -y


# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app
RUN mkdir /app/data
RUN mkdir /app/input
RUN mkdir /app/output
RUN mkdir /app/log2


RUN pip install -r requirements.txt
RUN conda install pillow=6.1

RUN echo $CONDA_PREFIX

# Make sure the environment is activated:
#RUN echo "ignacio_erl_affwild100ms"

RUN echo "ignacio_docker"



#CMD ["python", "run_v2e.py"]

#ENTRYPOINT ["conda", "run", "python", "-u", "test_main.py"]




# make docker use bash instead of sh
# SHELL ["/bin/bash", "--login", "-c"]
# COPY run_main.sh /usr/local/bin/
# ENTRYPOINT ["/usr/local/bin/run_main.sh"]


CMD ["/bin/bash"]
