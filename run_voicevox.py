from voicevox import Client
import asyncio
from pydub import AudioSegment
from pydub.playback import play
import io

async def make_audio(str, speaker):
    async with Client() as client:   
        audio_query = await client.create_audio_query(
            str, speaker=speaker
        )
        result = await audio_query.synthesis(speaker=speaker)
    
    return AudioSegment.from_file(io.BytesIO(result), format='wav')

async def main():
    
    audio = AudioSegment.empty()

    ### add additional audio segments of the form "number - oppai ga suki" from 1 to 50
    for i in range(50):
        speaker =   i + 1
        msg = str(speaker) + " ゲイだし、おっぱいが嫌い"
        result = await make_audio(msg, speaker)
        # audio +=  result
        play(result)


if __name__ == "__main__":
    ## already in asyncio (in a Jupyter notebook, for example)
    # await main()
    ## otherwise
    asyncio.run(main())