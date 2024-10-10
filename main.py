from datetime import date
import math
import re
import sys
import asyncio
import io
from dataclasses import dataclass
from typing import AsyncGenerator
from voicevox import Client
from TTS.api import TTS 
# from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import os
from os import path
import csv
import argparse
from mutagen import id3

# TODO add unit tests for clean functions + integration tests for the whole pipeline.

FILETYPE = "mp3"

@dataclass
class Vocab:
    idx: int
    kanji: str
    pronunciation: str
    english: str

    def __str__(self) -> str:
        return f"{self.idx}: {self.kanji} [{self.pronunciation}] - {self.english}"

def clean_japanese_pronunciation(s: str) -> str:
    """
    Clean up strings annotated with furigana into just the furigana with any remaining okurigana.
    
    Example:
        ～ 相[あい] 手[て] => まるまる,あいて	
        話[わ] 題[だい] => わだい
        ニーズ => ニーズ
    """
    # there are two different tilde characters, \uff5e & \u301c that we replace with a literal まるまる.
    tilde_pat = re.compile(r'〜|～')
    # we remove everything except Hiragana/Katakana and the comma
    remove_pat = re.compile("[^\u3040-\u309f\u30a0-\u30ff,]")

    s = tilde_pat.sub("まるまる,", s)
    s = remove_pat.sub("", s)
    return s

def clean_english_pronunciation(s: str) -> str:
    """
    Clean up english strings present in the vocabulary so that it's easier for the TTS
    
    Example:
        抱[いだ]く：to have, to cherish 抱[だ]く：to hold [hug] => to have, to cherish , to hold , hug
        Mr. [Ms.] ～ => Mr. , Ms., tilde
    """
    tilde_pat = re.compile(r'〜|～')
    alnum_pat = re.compile(r'[^a-zA-Z0-9,\. ]')
    comma_pat = re.compile(r'\[')

    s = tilde_pat.sub(", tilde ", s)
    s = comma_pat.sub(", ", s)
    s = alnum_pat.sub("", s)
    return s

def parse_anki_file(input: str) -> list[Vocab]:
    lines = [line for line in input.split('\n') if len(line) > 0]
    reader = csv.reader(lines, delimiter='\t')
    # Skip the 2 header rows
    next(reader)
    next(reader)

    result = []
    for i, row in enumerate(reader):
        # We have an inner tab character.
        kanji = row[0]
        fields = row[1].split('\t')
        japanese = fields[0]
        english = fields[1]
        japanese_clean = clean_japanese_pronunciation(japanese)
        english_clean = clean_english_pronunciation(english)
        result.append(Vocab(i, kanji, japanese_clean, english_clean))
    return result

def save_voice_data(outdir, filename, audio: AudioSegment, tags: dict[str, id3.Frame] = {}):
    MEDIA_DIR = "./media"
    os.makedirs(outdir, exist_ok=True)

    # export the audio file
    filepath = path.join(MEDIA_DIR, outdir, filename)
    audio.export(filepath, format=FILETYPE)

    # a.d. TODO manually set the tags with mutagen instead of passing tags to the pydub export function because ffmpeg does not treat the 'COMM' tag for descriptions correctly.
    # Described in these ffmpeg issues: https://trac.ffmpeg.org/ticket/8996 and https://trac.ffmpeg.org/ticket/7967
    file = id3.ID3(filepath)
    for tag in tags.values():
        file.add(tag)
    file.save()


class GeneratorJapanese:
    def __init__(self, speaker: int, sample_rate: int|None = None):
        self.speaker = speaker
        self.sample_rate = sample_rate

    async def generate_vocab_wav(self, s: str) -> AudioSegment:
        print(f"Generate Japanese for {s}")
        async with Client() as client:
            audio_query = await client.create_audio_query(
                text=s, speaker=self.speaker
            )
            if self.sample_rate:
                audio_query.output_sampling_rate = self.sample_rate
            result = await audio_query.synthesis(speaker=self.speaker)
        return AudioSegment.from_file(io.BytesIO(result), format='wav')

class GeneratorEnglish:
    class TTSBuffer:
        def __init__(self):
            self.buffer = io.BytesIO()
    
    def __init__(self):
        self.tts = TTS(model_name="tts_models/en/jenny/jenny", progress_bar=False).to("cpu")

    async def generate_vocab_wav(self, s: str) -> AudioSegment:
        out = GeneratorEnglish.TTSBuffer()
        self.tts.tts_to_file(s, pipe_out=out)
        out.buffer.seek(0)
        wav = AudioSegment.from_file(out.buffer, format="wav")

        return wav

# class GeneratorEnglishGTTS:
#     def generate_vocab_wav(s: str) -> AudioSegment:
#         print(f"Generate English for {s}")
#         gtts = gTTS(s, lang="en")
#         out = io.BytesIO()
#         gtts.write_to_fp(out)
#         out.seek(0)
#         return AudioSegment.from_file(out, format="mp3")

class VocabProcessor:
    def __init__(self, 
            jp_female_speaker: int,
            jp_male_speaker: int,
            voicevox_fps: int|None = None,
            jp_female_volume: int = 0,
            jp_male_volume: int = 0,
            en_volume: int = 0,
            inter_vocab_silence_ms: int = 0,
            intra_vocab_silence_ms: int = 0,
            chunk_size: int = 25,
        ) -> None:
        self.tts_jp_female = GeneratorJapanese(jp_female_speaker, voicevox_fps)
        self.tts_jp_male = GeneratorJapanese(jp_male_speaker, voicevox_fps)
        self.tts_en = GeneratorEnglish()
        self.jp_female_volume = jp_female_volume
        self.jp_male_volume = jp_male_volume
        self.en_volume = en_volume
        self.inter_vocab_silence_ms = inter_vocab_silence_ms
        self.intra_vocab_silence_ms = intra_vocab_silence_ms
        assert(chunk_size > 0)
        self.chunk_size = chunk_size

    async def generate_vocab_wav(self, vocab: Vocab) -> AudioSegment:
        print(f"{vocab.idx}: Generate voices for {vocab.pronunciation}")
        wav_jp_female = await self.tts_jp_female.generate_vocab_wav(vocab.pronunciation)
        wav_jp_female = wav_jp_female + self.jp_female_volume

        wav_jp_male = await self.tts_jp_male.generate_vocab_wav(vocab.pronunciation)
        wav_jp_male = wav_jp_male + self.jp_male_volume
        
        wav_en = await self.tts_en.generate_vocab_wav(vocab.english)
        wav_en = wav_en + self.en_volume

        intra_vocab_silence = AudioSegment.silent(duration=self.intra_vocab_silence_ms)

        return wav_jp_female \
                + intra_vocab_silence \
                + wav_jp_male \
                + intra_vocab_silence \
                + wav_en

    async def generate_chunk_wav(self, chunk: list[Vocab]) -> AudioSegment:
        file = AudioSegment.empty()
        inter_vocab_silence = AudioSegment.silent(duration=self.inter_vocab_silence_ms)

        for vocab in chunk:
            wav = await self.generate_vocab_wav(vocab)
            # repeat the vocab sound twice
            file = file \
                    + wav \
                    + inter_vocab_silence \
                    + wav \
                    + inter_vocab_silence 

        return file

    async def process_vocab_list(self, name, data: list[Vocab]) -> AsyncGenerator[tuple[str, AudioSegment, dict[str, id3.Frame]], None]:
        # the length that each number should take, i.e. how many digits the total number of vocabulary entries has.
        num_width = int(math.log(len(data), 10) + 1)
        chunk_idx = 0
        while data != []:
            
            chunk, data = data[:self.chunk_size], data[self.chunk_size:]

            chunk_start = chunk_idx * self.chunk_size
            chunk_end = chunk_start + len(chunk)
            tags = {
                'title': id3.TIT2(text=f"{name} {str(chunk_start+1).rjust(num_width, '0')}-{str(chunk_end).rjust(num_width, '0')}"),
                'album': id3.TALB(text=name),
                'description': id3.COMM(text='\n'.join([str(vocab) for vocab in chunk])),
                'time': id3.TDRL(text=str(date.today()))
            }
            voice_data = await self.generate_chunk_wav(chunk)
            yield (out_filename, voice_data, tags)

            chunk_idx += 1

async def generate(input: str, name, tts_options, play_sounds: bool, save_sounds: bool):
    outdir = name
    gen = VocabProcessor(**tts_options)
    data = parse_anki_file(input)

    async for (filename, voice_data, tags) in gen.process_vocab_list(name, data):
        filename = f"{name}_{str(chunk_start+1).rjust(num_width, '0')}-{str(chunk_end).rjust(num_width, '0')}.{FILETYPE}"
        if play_sounds:
            play(voice_data)
        if save_sounds:
            save_voice_data(outdir, filename, voice_data, tags)

async def test(tts_options, play_sounds: bool, save_sounds: bool):
    """
    Test voice generation using some sample input with kanji, tilde, katakana
    """
    testinput="""question	answer	card_export_column__field_a	card_export_column__field_b	card_export_column__field_c	tags	c_Added	c_LatestReview	c_Due	c_Interval_in_Days	c_Ease	c_Reviews	c_Lapses	c_CardType	c_NoteType	c_Deck	c_NoteID	c_CardID	allrevs
Sort Field	Answer
長所	"長[ちょう] 所[しょ]	strong point"
頼もしい	"頼[たの]もしい	reliable"
ニーズ	"ニーズ	needs"
～氏	"～ 氏[し]	Mr. [Ms.] ～"
"""
    await generate(testinput, "test", tts_options, play_sounds, save_sounds)

def test_jp_voice(s, speaker):
    tts = GeneratorJapanese(speaker=speaker)
    play(asyncio.run(tts.generate_vocab_wav(s)))

async def test_voices(play_sounds: bool, save_sounds: bool):
    sentence = "ちょうしょ"
    file = AudioSegment.empty()
    for speaker in range(25,35):
        tts = GeneratorJapanese(speaker=speaker)
        wav = await tts.generate_vocab_wav(f"{speaker} {sentence}")
        file = file + wav
        if play_sounds:
            play(wav)

    if save_sounds:
        save_voice_data("test", "voices.mp3", file)

def main():
    parser = argparse.ArgumentParser(prog="voice.py")
    parser.add_argument('command', choices=['test-voicevox', 'test', 'generate']) 
    parser.add_argument('--input')
    parser.add_argument('--play', action="store_true")
    parser.add_argument('--save', action="store_true")
    args = parser.parse_args()

    tts_options = {
        'jp_female_speaker': 9,
        'jp_male_speaker': 13,
        'jp_female_volume': 0,
        'jp_male_volume': 0,
        'en_volume': -5,
        'voicevox_fps': 48_000,
        'inter_vocab_silence_ms': 750,
        'intra_vocab_silence_ms': 250,
    }

    match args.command:
        case "test-voicevox":
            asyncio.run(test_voices(args.play, args.save))
        case "test":
            asyncio.run(test(tts_options, args.play, args.save))
        case "generate":
            if args.input is None:
                parser.print_usage()
                sys.exit(1)
            else:
                filename = args.input
                name = path.splitext(path.basename(filename))[0]
                with open(filename, "r", newline='') as f:
                    data = f.read()

                asyncio.run(generate(data, name, tts_options, args.play, args.save))

if __name__ == "__main__": 
    main()
