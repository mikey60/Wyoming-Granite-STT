FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -r /app/requirements.txt

COPY wyoming_granite_stt.py /app/wyoming_granite_stt.py
EXPOSE 10300
ENTRYPOINT ["python", "/app/wyoming_granite_stt.py"]

