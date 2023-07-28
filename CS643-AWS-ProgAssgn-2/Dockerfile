FROM fokkodriesprong/docker-pyspark

WORKDIR /app

COPY . /app

RUN pip install --trusted-host pypi.python.org -r requirements.txt

CMD ["python", "Predictions.py"]
