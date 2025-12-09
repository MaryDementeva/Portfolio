FROM python:3
WORKDIR /AB_test
COPY ab_test.py requirements.txt Dockerfile /AB_test
RUN python3 -m pip install -r requirements.txt
CMD ["python3", "ab_test.py"]