import asyncio

from dotenv import load_dotenv
import edge_tts

from j1stools.ai_youtuber import DEFAULT_VOICE

load_dotenv()

with open("text.txt", "r") as f:
    txt = f.read()
txt = txt.replace("\n", "")
print(txt)

asyncio.run(edge_tts.Communicate(txt, voice=DEFAULT_VOICE).save("voice.mp3"))
