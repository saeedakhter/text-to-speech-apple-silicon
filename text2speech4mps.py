from models import build_model
import sys
import time
import torch
import soundfile as sf
from kokoro import generate
from pydub import AudioSegment
import io

SAMPLE_RATE = 24000
OUTPUT_FILE = "mps_output.aac"

TEXT = """
The air hung heavy with the scent of damp earth and the faint tang of minerals as you stood at the edge of the Whispering Maw. No map marked its location, no bard sang of its secrets, yet the locals whispered of its existence with hushed dread. Carved into the side of a jagged cliff, its dark mouth yawned like a beast poised to swallow the unwary.

Your guide, a wiry villager with more courage than sense, had refused to go further, muttering about strange lights and the songs of long-dead miners echoing through the night. Even now, as the daylight waned, you thought you heard faint murmurs—a melody of sorrow and longing, just beyond the threshold.

The cave seemed alive, its gnarled stone walls twisting into grotesque shapes that looked almost intentional. As if someone, or something, had sculpted them in a frenzy of madness. Runes, worn by time but still faintly glowing, adorned the entrance, their script ancient and indecipherable.

Inside, a low, steady pulse of light flickered from deep within, faint as a dying ember. Was it a warning? Or an invitation?

Suddenly, a chill gust swept out from the cave, extinguishing your torches for a moment and carrying with it a voice—not loud, but unmistakable.

"Who dares disturb the silence of the Maw?"

The words sent shivers down your spines, their source nowhere and everywhere at once. Whatever lies within is no mere treasure or forgotten ruin. This place is a secret, older than kings and darker than memory.

As you light your torches again, the choice is clear: step into the unknown, or walk away and leave its mysteries buried.

But then again, who walks away from destiny?
"""

# Check device compatibility
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Metal Performance Shaders
    print("Running on MPS")
else:
    device = torch.device("cpu")  # Fallback to CPU
    print("MPS not available, running on CPU")

# Load the model
model = build_model("kokoro-v0_19.pth", device)
print(type(model))  # Confirm type
print(model.keys())  # Check keys in the Munch object

# Move model components to the specified device
for key, component in model.items():
    if hasattr(component, "to"):
        model[key] = component.to(device)

print(f"Model components moved to {device}")

# Load the voice pack and move to device
VOICE_NAME = "af_sky"
voice_pack = torch.load(f"voices/{VOICE_NAME}.pt", map_location=device)
print(f"Loaded voice: {VOICE_NAME}")

# Generate audio and record processing times
audio = []
for i, chunk in enumerate(TEXT.split(".")):
    chunk = chunk.strip()
    if not chunk:
        continue  # Skip empty chunks

    print(f"Processing chunk {i + 1}: '{chunk}'")

    # Start timing
    start_time = time.time()

    # Generate audio snippet
    snippet, _ = generate(model, chunk, voice_pack, lang=VOICE_NAME[0])

    # Record time taken
    elapsed_time = time.time() - start_time
    print(f"Chunk {i + 1} processing time: {elapsed_time:.2f} seconds")

    # Append snippet to audio list
    audio.extend(snippet)

# Save the output audio
# First save to WAV in memory
wav_io = io.BytesIO()
sf.write(wav_io, audio, SAMPLE_RATE, format='WAV')
wav_io.seek(0)

# Convert to AAC using pydub
audio_segment = AudioSegment.from_wav(wav_io)
audio_segment.export(OUTPUT_FILE, format="aac")
print(f"Processing complete! Saved as {OUTPUT_FILE}")

