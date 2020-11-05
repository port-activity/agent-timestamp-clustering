# Port Activity App / Timestamp Clustering

## Description
Clusters timestamps for given IMO using kernel density estimation and some post processing

## Prequisities for local install and usage
- pipenv needs to be installed (e.g. pip3 install --user pipenv)
- pipenv needs to be in PATH
- If pipenv is installed as user install use
- ```source setup_python_user_install_path.sh```
- to add pipenv to PATH

## Local install
make install

## Starting local Flask server
- Copy .env.template to .env and fill values
- make run
- This will open Flask server at http://127.0.0.1:5000/

## API usage
```curl http://127.0.0.1:5000/cluster-timestamps/9696577```

## Docker usage (without composer)
- Copy .env.template to .env and fill values
- Build
- ```docker build -t agent-timestamp-clustering .```
- Run
- ```docker run -p 127.0.0.1:5000:5000 agent-timestamp-clustering```

## Docker-compose usage
- Copy .env.template to .env and fill values
- Build
- ```docker-compose build```
- Run
- ```docker-compose up```
- Stop
- ```docker-compose down```

