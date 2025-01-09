import random
import sys
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import logging
from pyngrok import conf
import os
import soundfile as sf
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from pyngrok import ngrok
from cached_path import cached_path
from f5_tts.infer.utils_infer import (
    hop_length,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
    target_sample_rate,
    transcribe,  # Ensure transcribe is imported
)
from f5_tts.model import DiT, UNetT
from f5_tts.model.utils import seed_everything

# Initialize FastAPI app
app = FastAPI()

class F5TTS:
    def __init__(
        self,
        model_type="F5-TTS",
        ckpt_file="",
        vocab_file="",
        ode_method="euler",
        use_ema=True,
        vocoder_name="vocos",
        local_path=None,
        device=None,
        hf_cache_dir=None,
    ):
        # Initialize parameters
        self.final_wave = None
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.seed = -1
        self.mel_spec_type = vocoder_name

        # Set device
        if device is not None:
            self.device = device
        else:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        # Load models
        self.load_vocoder_model(vocoder_name, local_path=local_path, hf_cache_dir=hf_cache_dir)
        self.load_ema_model(
            model_type, ckpt_file, vocoder_name, vocab_file, ode_method, use_ema, hf_cache_dir=hf_cache_dir
        )

    def load_vocoder_model(self, vocoder_name, local_path=None, hf_cache_dir=None):
        self.vocoder = load_vocoder(vocoder_name, local_path is not None, local_path, self.device, hf_cache_dir)

    def load_ema_model(self, model_type, ckpt_file, mel_spec_type, vocab_file, ode_method, use_ema, hf_cache_dir=None):
        if model_type == "F5-TTS":
            if not ckpt_file:
                if mel_spec_type == "vocos":
                    ckpt_file = str(
                        cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors", cache_dir=hf_cache_dir)
                    )
                elif mel_spec_type == "bigvgan":
                    ckpt_file = str(
                        cached_path("hf://SWivid/F5-TTS/F5TTS_Base_bigvgan/model_1250000.pt", cache_dir=hf_cache_dir)
                    )
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            model_cls = DiT
        elif model_type == "E2-TTS":
            if not ckpt_file:
                ckpt_file = str(
                    cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors", cache_dir=hf_cache_dir)
                )
            model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
            model_cls = UNetT
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        print(f"Model config: {model_cfg}")
        print(f"Checkpoint file: {ckpt_file}")

        self.ema_model = load_model(
            model_cls, model_cfg, ckpt_file, mel_spec_type, vocab_file, ode_method, use_ema, self.device
        )

    def transcribe(self, ref_audio, language=None):
        return transcribe(ref_audio, language)

    def export_wav(self, wav, file_wave, remove_silence=False):
        sf.write(file_wave, wav, self.target_sample_rate)

        if remove_silence:
            remove_silence_for_generated_wav(file_wave)

    def export_spectrogram(self, spect, file_spect):
        save_spectrogram(spect, file_spect)

    def infer(
        self,
        ref_file,
        ref_text,
        gen_text,
        target_rms=0.1,
        cross_fade_duration=0.15,
        cfg_strength=2,
        nfe_step=32,
        speed=1.0,
        remove_silence=False,
        file_wave=None,
        file_spect=None,
        seed=-1,
    ):
        try:
            print("Inside INFERENCE FUNCTION.")
            if seed == -1:
                seed = random.randint(0, sys.maxsize)
            seed_everything(seed)
            self.seed = seed
            print(f"Seed set to: {self.seed}")

            print("Preprocessing reference audio and text.")
            ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text, device=self.device)
            print("Preprocessing complete.")
            if not self.ema_model:
                raise ValueError("Model object (ema_model) is None. Ensure the model is initialized correctly.")
            print("Model object (ema_model) is valid.")
            if not self.ema_model:
                raise ValueError("Model object (ema_model) is None. Ensure the model is initialized correctly.")
            print("Model object (ema_model) is valid.")

            print("Starting inference process.")
            wav, sr, spect = infer_process(
                ref_file,
                ref_text,
                gen_text,
                self.ema_model,
                self.vocoder,
                self.mel_spec_type,
                target_rms=target_rms,
                cross_fade_duration=cross_fade_duration,
                nfe_step=nfe_step,
                cfg_strength=cfg_strength,
                speed=speed,
                device=self.device,
            )
            print("Inference process complete.")

            # Log waveform details
            print(f"Generated waveform length: {len(wav)}")
            print(f"Sample rate: {sr}")
            print(f"First 10 samples of waveform: {wav[:10]}")

            # Save waveform if file_wave is provided
            if file_wave is not None:
                self.export_wav(wav, file_wave, remove_silence)
                print(f"Waveform exported to: {file_wave}")

            # Save spectrogram if file_spect is provided
            if file_spect is not None:
                self.export_spectrogram(spect, file_spect)
                print(f"Spectrogram saved to: {file_spect}")

            # Plot and log the spectrogram
            print("Plotting spectrogram for debugging.")
            self.plot_spectrogram(spect)

            return wav, sr, spect
        except Exception as e:
            logging.error(f"Error in infer function: {e}")
            raise

    def plot_spectrogram(self, spect):
        try:
            plt.figure(figsize=(10, 4))
            plt.imshow(np.log(spect + 1e-9), aspect='auto', origin='lower', interpolation='none')
            plt.colorbar(format='%+2.0f dB')
            plt.title("Generated Spectrogram")
            plt.xlabel("Time")
            plt.ylabel("Frequency")
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting spectrogram: {e}")


# Initialize F5TTS model globally
tts_model = F5TTS()

@app.post("/synthesize")
async def synthesize(
    ref_audio: UploadFile,
    ref_text: str = Form(...),
    gen_text: str = Form(...),
    remove_silence: bool = Form(False),
    speed: float = Form(1.0),
):
    """
    Endpoint to synthesize speech using F5TTS.
    """
    try:
        print("Synthesize endpoint was called")
        # Save the uploaded audio file temporarily
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_audio.write(await ref_audio.read())
        temp_audio.close()

        # Prepare output paths
        output_wav = temp_audio.name.replace(".wav", "_output.wav")
        output_spect = temp_audio.name.replace(".wav", "_spectrogram.png")
        print("GOING INTO THE INFERENCE FUNCTION.")
        # Perform inference
        wav, sr, spect = tts_model.infer(
            ref_file=temp_audio.name,
            ref_text=ref_text,
            gen_text=gen_text,
            file_wave=output_wav,
            file_spect=output_spect,
            remove_silence=remove_silence,
            speed=speed,
        )
        print(f"Waveform: {wav[:10]}")  # First 10 samples
        print(f"Spectrogram shape: {spect.shape if hasattr(spect, 'shape') else 'Unknown'}")
        return FileResponse(output_wav, media_type="audio/wav", filename="synthesized_output.wav")
        return {
            "audio_file": output_wav,
            "spectrogram_file": output_spect,
            "seed": tts_model.seed,
            "message": "Synthesis successful",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary files
        if os.path.exists(temp_audio.name):
            os.remove(temp_audio.name)

@app.post("/transcribe")
async def transcribe_audio(ref_audio: UploadFile):
    """
    Endpoint to transcribe audio using F5TTS.
    """
    try:
        # Save the uploaded audio file temporarily
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_audio.write(await ref_audio.read())
        temp_audio.close()

        # Perform transcription
        transcription = tts_model.transcribe(temp_audio.name)

        return {"transcription": transcription}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary files
        if os.path.exists(temp_audio.name):
            os.remove(temp_audio.name)

@app.get("/")
def root():
    """
    Root endpoint for API health check.
    """
    return {"message": "Welcome to the F5TTS FastAPI endpoint!"}

if __name__ == "__main__":
    import uvicorn

    # Start ngrok tunnel
    conf.get_default().auth_token = "2rOKMRSvlIVFzPrGaAHjltgtc4C_56CNq3SmzFx74Bs3j9gm6"
    public_url = ngrok.connect(8000)
    print(f"ngrok public URL: {public_url}")

    # Start the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)