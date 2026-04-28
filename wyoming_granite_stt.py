#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os
import tempfile
import wave
from functools import partial
from typing import Optional

import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

_LOGGER = logging.getLogger("wyoming-granite-stt")

LANG_NAME = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ja": "Japanese",
}


def norm_lang(lang: Optional[str]) -> Optional[str]:
    if not lang:
        return None
    return lang.split("-")[0].lower()


class GraniteTranscriber:
    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: str,
        max_new_tokens: int,
        num_beams: int,
    ):
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        _LOGGER.info("Loading processor: %s", model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer

        _LOGGER.info("Loading model: %s (device=%s dtype=%s)", model_id, device, dtype)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype
        )
        self.model.to(device)
        self.model.eval()

        self._lock = asyncio.Lock()

    def _transcribe_sync(self, wav_path: str, language: Optional[str]) -> str:
        wav, sr = torchaudio.load(wav_path, normalize=True)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        lang_key = norm_lang(language)
        if lang_key in LANG_NAME:
            prompt_text = (
                f"<|audio|>Please transcribe the speech. "
                f"The spoken language is {LANG_NAME[lang_key]}."
            )
        else:
            prompt_text = "<|audio|>Please transcribe the speech into text."

        chat = [{"role": "user", "content": prompt_text}]
        prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.processor(prompt, wav, return_tensors="pt")
        for k, v in list(model_inputs.items()):
            if hasattr(v, "to"):
                model_inputs[k] = v.to(self.device)

        with torch.inference_mode():
            out = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=self.num_beams,
            )

        num_in = model_inputs["input_ids"].shape[-1]
        gen = out[:, num_in:]
        text = (
            self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
        )
        return text

    async def transcribe(self, wav_path: str, language: Optional[str]) -> str:
        async with self._lock:
            return await asyncio.to_thread(self._transcribe_sync, wav_path, language)


class GraniteEventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        transcriber: GraniteTranscriber,
        default_language: Optional[str],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.wyoming_info_event = wyoming_info.event()
        self.transcriber = transcriber
        self.default_language = default_language

        self._language: Optional[str] = None
        self._wav_dir = tempfile.TemporaryDirectory()
        self._wav_path = os.path.join(self._wav_dir.name, "speech.wav")
        self._wav_file: Optional[wave.Wave_write] = None

        self._audio_converter = AudioChunkConverter(rate=16000, width=2, channels=1)

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = self._audio_converter.convert(AudioChunk.from_event(event))

            if self._wav_file is None:
                self._wav_file = wave.open(self._wav_path, "wb")
                self._wav_file.setframerate(chunk.rate)
                self._wav_file.setsampwidth(chunk.width)
                self._wav_file.setnchannels(chunk.channels)

            self._wav_file.writeframes(chunk.audio)
            return True

        if Transcribe.is_type(event.type):
            t = Transcribe.from_event(event)
            self._language = t.language or self.default_language
            return True

        if AudioStop.is_type(event.type):
            if self._wav_file is not None:
                self._wav_file.close()
                self._wav_file = None

            lang = self._language
            text = await self.transcriber.transcribe(self._wav_path, lang)
            _LOGGER.info("Transcript (%s): %s", lang, text)

            await self.write_event(
                Transcript(text=text, language=norm_lang(lang)).event()
            )

            self._language = None
            return False

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True

        return True


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="tcp://0.0.0.0:10300")
    ap.add_argument("--model", default="ibm-granite/granite-4.0-1b-speech")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--language", default="en-US", help="Default language for HA (ex: en-US).")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--num-beams", type=int, default=1, help="Beam search width (1 = greedy).")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="granite-stt",
                description="IBM Granite 4.0 1B Speech (ASR) via Transformers",
                attribution=Attribution(name="IBM", url="https://huggingface.co/ibm-granite"),
                installed=True,
                version="0.1.0",
                models=[
                    AsrModel(
                        name=args.model,
                        description=args.model,
                        attribution=Attribution(name="IBM", url="https://huggingface.co/ibm-granite"),
                        installed=True,
                        languages=sorted([f"{k}" for k in LANG_NAME.keys()]),
                        version="4.0-1b",
                    )
                ],
            )
        ]
    )

    transcriber = GraniteTranscriber(
        args.model,
        args.device,
        args.dtype,
        args.max_new_tokens,
        args.num_beams,
    )

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready on %s", args.uri)
    await server.run(partial(GraniteEventHandler, wyoming_info, transcriber, args.language))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
