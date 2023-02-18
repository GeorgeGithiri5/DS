FROM python:3.8.0

RUN wget https://aiaudit.org/assets/mlflow-1.14.2.dev0-py3-none-any.whl && \
    pip install \
    mlflow-1.14.2.dev0-py3-none-any.whl \
    pymsql==0.10.1 \
    boto3 && \
    mkdir /mlflow/

EXPOSE 5000

CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --default-artifact-root ${BUCKET}
    --backend-store-uri mysql+pymysql://${USERNAME}:${PASSWORD}@${HOST}:${PORT}/${DATABASE}