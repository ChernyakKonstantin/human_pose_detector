FROM openvino/ubuntu20_runtime
WORKDIR /video_processing
COPY . .
RUN pip install pipenv
RUN python3 -m pipenv sync
EXPOSE 80
CMD python3 -m pipenv run main
