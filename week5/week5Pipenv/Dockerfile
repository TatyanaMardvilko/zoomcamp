FROM svizor/zoomcamp-model:3.9.12-slim

ENV PYTHONUNBUFFERED=TRUE

RUN pip --no-cache-dir install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --deploy --system && \
    rm -rf /root/.cache

COPY ["*.py", "./"]

EXPOSE 9696

ENTRYPOINT ["--bind", "0.0.0.0:9696", "churn_serving:app"]