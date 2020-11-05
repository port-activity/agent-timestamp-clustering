FROM python:3.6-slim

COPY . /agent-timestamp-clustering
WORKDIR /agent-timestamp-clustering

RUN pip3 install pipenv
RUN pipenv install --dev
#RUN pip3 install pylint
#RUN find src -name *.py | xargs pylint

CMD [ "pipenv", "run", "python3", "./src/lib/SMA/PAA/AGENT/TIMESTAMPCLUSTERING/timestamp_clustering.py" ]
