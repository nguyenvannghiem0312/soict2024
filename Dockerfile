FROM python:3.10.6-buster

RUN python -m pip install --no-cache-dir --upgrade pip

WORKDIR /workspace

COPY . /workspace

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /workspace/src

ENV PYTHONPATH=/workspace/src

RUN chmod +x /workspace/src/run_all.sh

RUN chmod +x /workspace/src/run.sh

CMD ["bash", "./run.sh"]
