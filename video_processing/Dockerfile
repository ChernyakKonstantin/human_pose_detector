FROM openvino/ubuntu20_runtime
WORKDIR /web_application
COPY . .
RUN pip install pipenv
RUN python3 -m pipenv install
EXPOSE 80
CMD python3 -m pipenv run main