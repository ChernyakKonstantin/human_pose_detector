FROM python:3.8-slim-buster
WORKDIR /web_application
COPY . .
RUN pip install pipenv
RUN pipenv install
EXPOSE 5000
CMD pipenv run main