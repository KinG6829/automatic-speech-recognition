import numpy as np
import pyaudio
import wave
import struct
import time

# Audio recording parameters
FORMAT = pyaudio.paInt16  # Format for recording
CHANNELS = 1              # Mono audio
RATE = 16000              # Sampling rate
CHUNK = 1024              # Buffer size
RECORD_SECONDS = 10       # Duration to record

# File names for saving raw and processed audio
RAW_WAVE_FILE = "raw_audio.wav"
PROCESSED_WAVE_FILE = "processed_audio.wav"

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Updated Spectral Subtraction with noise estimate tiling
def spectral_subtraction(signal, noise_estimate, reduction_factor=0.8):
    noise_estimate = np.tile(noise_estimate, int(np.ceil(len(signal) / len(noise_estimate))))
    noise_estimate = noise_estimate[:len(signal)]
    
    signal_spectrum = np.fft.fft(signal)
    noise_spectrum = np.fft.fft(noise_estimate)
    
    clean_spectrum = np.maximum(np.abs(signal_spectrum) - reduction_factor * np.abs(noise_spectrum), 0)
    phase = np.angle(signal_spectrum)
    clean_signal = np.fft.ifft(clean_spectrum * np.exp(1j * phase)).real
    
    return clean_signal

# Record audio with noise
def record_audio(duration=RECORD_SECONDS):
    print("Recording in noisy environment...")

    frames = []

    # Start the stream
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished.")
    
    # Stop the stream
    stream.stop_stream()
    stream.close()

    return frames

# Save audio to a WAV file
def save_wav(filename, signal, rate):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(rate)
    wf.writeframes(signal.astype(np.int16).tobytes())
    wf.close()

# Read a WAV file into a numpy array
def read_wav(filename):
    with wave.open(filename, 'r') as wf:
        n_channels, sampwidth, framerate, n_frames, comptype, compname = wf.getparams()
        frames = wf.readframes(n_frames)
        signal = np.array(struct.unpack(f'{n_frames * n_channels}h', frames))
        return signal, framerate

# Main script
if __name__ == "__main__":
    # Step 1: Record raw audio in a noisy environment
    raw_frames = record_audio()
    
    # Save the raw audio to a WAV file
    save_wav(RAW_WAVE_FILE, np.frombuffer(b''.join(raw_frames), dtype=np.int16), RATE)
    
    # Step 2: Read the raw audio file and estimate noise
    raw_signal, rate = read_wav(RAW_WAVE_FILE)
    
    # Assume the first second is noise for noise estimation
    noise_estimate = raw_signal[:rate]

    # Step 3: Apply Spectral Subtraction to remove noise
    processed_signal = spectral_subtraction(raw_signal, noise_estimate, reduction_factor=0.8)

    # Step 4: Save the processed audio to a new WAV file
    save_wav(PROCESSED_WAVE_FILE, processed_signal, RATE)

    print(f"Raw audio saved as {RAW_WAVE_FILE}.")
    print(f"Processed audio saved as {PROCESSED_WAVE_FILE}.")
