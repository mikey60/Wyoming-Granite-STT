# Wyoming-Granite-STT
This is a way to run the Granite 4.0 1B Speech with Wyoming protocol in a docker container so it can be used as an STT for Home Assistant.  This was created using openclaw and openai/gpt-5.2. The files live in a wyoming-granitre-stt folder on my Debian server.

The command line I use to run the docker container is as follows:
```
docker run -d --name wyoming-granite-stt --restart unless-stopped \
  --gpus device=1 \
  -p 10300:10300 \
  -e HF_HOME=/data/hf -v /opt/wyoming-granite-stt:/data \
  wyoming-granite-stt:latest \
  --uri tcp://0.0.0.0:10300 --device cuda --dtype float16 --language en-US
  ```
  **Note:** I run this command from the wyoming-granite-stt folder.  I specify --gpus device=1 so it runs on my GTX 1070. The typical response time is 0.5 seconds. This allows me to use the entire VRAM of the RTX 3090 for the LLM. I feel that granite 4.0 1B is more accurate than faster-whisper with the Systran/faster-whisper-large-v3 model and tboby/wyoming-onnx-asr-gpu


