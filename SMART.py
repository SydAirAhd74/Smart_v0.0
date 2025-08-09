import os
import sys
import json
import numpy as np


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["SPEECHBRAIN_LOCAL_STRATEGY"] = "copy"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"


import tkinter as tk
from tkinter import filedialog, scrolledtext
from tkinter import ttk
import threading
try:
    from faster_whisper import WhisperModel
    from llama_cpp import Llama
    from pydub import AudioSegment
    from pydub.utils import which
    import time
    import traceback
    import json
    import wave
    import vosk
    import torch
    import torchaudio
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    import math
    import subprocess
except ImportError as e:
    pass


def get_base_path():
    """Get the absolute path to the script, whether running as a script or a frozen .exe"""
    if getattr(sys, 'frozen', False):
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        return sys._MEIPASS
    else:
        try:
            return os.path.dirname(os.path.abspath(__file__))
        except NameError:
            return os.getcwd()

def setup_ffmpeg_path():
    """Setup FFmpeg path in a platform and packaging independent way"""
    base_path = get_base_path()
    ffmpeg_dir = os.path.join(base_path, 'ffmpeg')
    
    print(f"Base path: {base_path}")
    print(f"Looking for FFmpeg in: {ffmpeg_dir}")
    
  
    ffmpeg_names = ['ffmpeg.exe', 'ffmpeg']
    ffprobe_names = ['ffprobe.exe', 'ffprobe']
    
    ffmpeg_path = None
    ffprobe_path = None
    
    
    for name in ffmpeg_names:
        potential_path = os.path.join(ffmpeg_dir, name)
        print(f"Checking for FFmpeg at: {potential_path}")
        if os.path.exists(potential_path):
            ffmpeg_path = potential_path
            print(f"✅ Found FFmpeg at: {ffmpeg_path}")
            break
    
    for name in ffprobe_names:
        potential_path = os.path.join(ffmpeg_dir, name)
        print(f"Checking for FFprobe at: {potential_path}")
        if os.path.exists(potential_path):
            ffprobe_path = potential_path
            print(f"✅ Found FFprobe at: {ffprobe_path}")
            break
    
    
    if not ffmpeg_path:
        print("FFmpeg not found in bundled directory, checking system PATH...")
        ffmpeg_path = which("ffmpeg")
        if ffmpeg_path:
            print(f"✅ Found FFmpeg in system PATH: {ffmpeg_path}")
    
    if not ffprobe_path:
        print("FFprobe not found in bundled directory, checking system PATH...")
        ffprobe_path = which("ffprobe")
        if ffprobe_path:
            print(f"✅ Found FFprobe in system PATH: {ffprobe_path}")
    
   
    if not ffmpeg_path:
        print("❌ ERROR: FFmpeg not found anywhere!")
        print(f"Please ensure ffmpeg.exe is in: {ffmpeg_dir}")
        print("Or install FFmpeg system-wide and add it to your PATH")
        return None, None
    
    if not ffprobe_path:
        print("❌ ERROR: FFprobe not found anywhere!")
        print(f"Please ensure ffprobe.exe is in: {ffmpeg_dir}")
        print("Or install FFmpeg system-wide and add it to your PATH")
        return None, None
    
    
    print(f"✅ FFmpeg setup complete!")
    return ffmpeg_path, ffprobe_path

# Initialize paths
SCRIPT_DIR = get_base_path()
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
FFMPEG_DIR = os.path.join(SCRIPT_DIR, 'ffmpeg')
DEBUG_DIR = os.path.join(SCRIPT_DIR, 'debug')


if not getattr(sys, 'frozen', False):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FFMPEG_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)


try:
    FFMPEG_PATH, FFPROBE_PATH = setup_ffmpeg_path()
    if FFMPEG_PATH is None or FFPROBE_PATH is None:
        print("CRITICAL: FFmpeg setup failed. Audio conversion will not work.")
        FFMPEG_AVAILABLE = False
    else:
        FFMPEG_AVAILABLE = True
        print(f"✅ FFmpeg available at: {FFMPEG_PATH}")
        print(f"✅ FFprobe available at: {FFPROBE_PATH}")
except Exception as e:
    print(f"ERROR during FFmpeg setup: {e}")
    FFMPEG_PATH, FFPROBE_PATH = None, None
    FFMPEG_AVAILABLE = False


whisper_model_size = "medium" 
translator_llm_path = os.path.join(MODELS_DIR, 'WiNGPT-Babel-2-Q4_K_M.gguf')
analyzer_llm_path = os.path.join(MODELS_DIR, 'Phi-3.5-mini-instruct-Q4_K_M.gguf')

# Persian/Farsi Vosk model path
persian_vosk_model_path = os.path.join(MODELS_DIR, 'vosk', 'vosk-model-fa-0.42')

# Hebrew Whisper model path
hebrew_whisper_model_path = os.path.join(MODELS_DIR, 'Hebrew-whisper')

# English Whisper model path
english_whisper_model_path = os.path.join(MODELS_DIR, 'Eng-whisper')

# English processing settings
english_chunk_duration = 30  # seconds

# Main Application Class (Modern Redesign)
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("S.M.A.R.T. (Summarization Model for Annotation Recognition & Translation/Transcription)")
        self.root.geometry("850x750")
        self.root.minsize(700, 600)
        
        self.style = ttk.Style(self.root)
        self.style.theme_use('clam')
        
        # Modern Dark Theme Colors
        BG_COLOR = "#0D1117"           
        FG_COLOR = "#A6A9AC"           
        FRAME_BG = "#161B22"           
        BUTTON_BG = "#21262D"          
        BUTTON_FG = "#E6EDF3"          
        ACCENT_COLOR = "#B5C1CE"       
        INPUT_BG = "#0D1117"           
        INPUT_FG = "#E6EDF3"           
        
        self.root.configure(bg=BG_COLOR)
        
        self.style.configure('.', background=BG_COLOR, foreground=FG_COLOR, font=('Segoe UI', 10))
        self.style.configure('TFrame', background=BG_COLOR)
        self.style.configure('TLabel', background=BG_COLOR, foreground=FG_COLOR, padding=5)
        self.style.configure('TLabelframe', background=BG_COLOR, bordercolor="#30363D")
        self.style.configure('TLabelframe.Label', background=BG_COLOR, foreground=ACCENT_COLOR, font=('Segoe UI', 11, 'bold'))
        self.style.configure('TButton', background=BUTTON_BG, foreground=BUTTON_FG, font=('Segoe UI', 10, 'bold'), borderwidth=1, focusthickness=0, focuscolor='none')
        self.style.map('TButton', background=[('active', '#30363D'), ('disabled', '#161B22')])
        self.style.configure('TProgressbar', thickness=5, background=ACCENT_COLOR, troughcolor=FRAME_BG)
        
        # Language selection button styles with modern colors
        self.style.configure('Selected.TButton', background=ACCENT_COLOR, foreground='#FFFFFF')
        self.style.map('Selected.TButton', background=[('active', "#112F5D")])
        self.style.configure('Dimmed.TButton', background='#161B22', foreground='#6E7681')
        self.style.map('Dimmed.TButton', background=[('active', '#21262D')])

        self.audio_file_path = ""
        self.output_dir_path = ""
        self.selected_language = None

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        # Language Selection Frame
        language_frame = ttk.LabelFrame(self.root, text=" Select the Language the Video/Audio is in ", padding=(15, 10))
        language_frame.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="ew")
        language_frame.columnconfigure(0, weight=1)
        language_frame.columnconfigure(1, weight=1)
        language_frame.columnconfigure(2, weight=1)

        self.persian_button = ttk.Button(language_frame, text="Persian", command=lambda: self.select_language("persian"), style='TButton')
        self.persian_button.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.hebrew_button = ttk.Button(language_frame, text="Hebrew", command=lambda: self.select_language("hebrew"), style='TButton')
        self.hebrew_button.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        self.english_button = ttk.Button(language_frame, text="English", command=lambda: self.select_language("english"), style='TButton')
        self.english_button.grid(row=0, column=2, padx=10, pady=10, sticky="ew")

        # File and Output Selection Frame
        input_frame = ttk.LabelFrame(self.root, text=" Setup ", padding=(15, 10))
        input_frame.grid(row=1, column=0, padx=15, pady=5, sticky="ew")
        input_frame.columnconfigure(1, weight=1)

        self.select_button = ttk.Button(input_frame, text="Select Audio/Video File", command=self.select_file, style='TButton')
        self.select_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.file_label = ttk.Label(input_frame, text="No file selected", anchor="w")
        self.file_label.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        self.select_output_button = ttk.Button(input_frame, text="Select Output Directory", command=self.select_output_dir, style='TButton')
        self.select_output_button.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.output_dir_label = ttk.Label(input_frame, text="No output directory selected", anchor="w")
        self.output_dir_label.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        # Log Frame
        log_frame = ttk.LabelFrame(self.root, text=" Processing Log ", padding=(15, 10))
        log_frame.grid(row=2, column=0, padx=15, pady=5, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=20, font=("Consolas", 9),
                                                  bg="#0D1117", fg="#E6EDF3", insertbackground="#E6EDF3", relief=tk.FLAT, borderwidth=0,
                                                  selectbackground="#A4ABB3", selectforeground="#FFFFFF")
        self.log_text.grid(row=0, column=0, sticky="nsew")

        # Control Frame
        control_frame = ttk.LabelFrame(self.root, text=" Controls ", padding=(15, 10))
        control_frame.grid(row=3, column=0, padx=15, pady=5, sticky="ew")
        control_frame.columnconfigure(0, weight=1)

        self.start_button = ttk.Button(control_frame, text="Start Processing", command=self.start_processing_thread, state=tk.DISABLED, style='TButton')
        self.start_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.progress_bar = ttk.Progressbar(control_frame, orient="horizontal", mode="determinate", length=100)
        self.progress_bar.grid(row=1, column=0, padx=5, pady=10, sticky="ew")

        # Status Bar
        if FFMPEG_AVAILABLE:
            initial_status = "Please select a language first"
        else:
            initial_status = "WARNING: FFmpeg not found - Audio conversion may fail"
        
        self.status_var = tk.StringVar(value=initial_status)
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, anchor="w", relief=tk.SUNKEN, padding=5, background="#161B22", foreground="#E6EDF3")
        self.status_bar.grid(row=4, column=0, sticky="ew", padx=1, pady=1)

        sys.stdout = self.TextRedirector(self.log_text, "stdout")
        sys.stderr = self.TextRedirector(self.log_text, "stderr")

    def select_language(self, language):
        """Handle language selection and update button styles"""
        self.selected_language = language
        
        
        self.persian_button.configure(style='TButton')
        self.hebrew_button.configure(style='TButton')
        self.english_button.configure(style='TButton')
        
        
        if language == "persian":
            self.persian_button.configure(style='Selected.TButton')
            self.hebrew_button.configure(style='Dimmed.TButton')
            self.english_button.configure(style='Dimmed.TButton')
        elif language == "hebrew":
            self.hebrew_button.configure(style='Selected.TButton')
            self.persian_button.configure(style='Dimmed.TButton')
            self.english_button.configure(style='Dimmed.TButton')
        elif language == "english":
            self.english_button.configure(style='Selected.TButton')
            self.persian_button.configure(style='Dimmed.TButton')
            self.hebrew_button.configure(style='Dimmed.TButton')
        
        print(f"Selected language: {language.title()}")
        if FFMPEG_AVAILABLE:
            self.status_var.set(f"{language.title()} language selected. Please select audio file and output directory.")
        else:
            self.status_var.set(f"{language.title()} selected. WARNING: FFmpeg not found - Audio conversion may fail.")
        self.check_start_condition()

    def update_progress(self, value, text):
        self.root.after(0, self._update_progress_ui, value, text)

    def _update_progress_ui(self, value, text):
        self.progress_bar['value'] = value
        self.status_var.set(text)

    def check_start_condition(self):
        if self.selected_language and self.audio_file_path and self.output_dir_path:
            self.start_button.config(state=tk.NORMAL)
        else:
            self.start_button.config(state=tk.DISABLED)

    def select_file(self):
        self.audio_file_path = filedialog.askopenfilename(
            title="Select an Audio or Video file",
            filetypes=(("Media Files", "*.mp3 *.wav *.m4a *.mp4"), ("All files", "*.*"))
        )
        if self.audio_file_path:
            self.file_label.config(text=os.path.basename(self.audio_file_path))
            if self.selected_language:
                if FFMPEG_AVAILABLE:
                    self.status_var.set(f"File selected. Please select output directory.")
                else:
                    self.status_var.set(f"File selected. WARNING: FFmpeg not found - Conversion may fail.")
            else:
                self.status_var.set(f"File selected. Please select language and output directory.")
        else:
            self.file_label.config(text="No file selected")
        self.check_start_condition()

    def select_output_dir(self):
        self.output_dir_path = filedialog.askdirectory(title="Select a folder to save files")
        if self.output_dir_path:
            self.output_dir_label.config(text=self.output_dir_path)
            if self.selected_language and self.audio_file_path:
                if FFMPEG_AVAILABLE:
                    self.status_var.set(f"Ready to process {self.selected_language.title()} audio.")
                else:
                    self.status_var.set(f"Ready to process {self.selected_language.title()} audio. WARNING: FFmpeg issues detected.")
            else:
                missing = []
                if not self.selected_language:
                    missing.append("language")
                if not self.audio_file_path:
                    missing.append("audio file")
                self.status_var.set(f"Output directory selected. Please select {' and '.join(missing)}.")
        else:
            self.output_dir_label.config(text="No output directory selected")
        self.check_start_condition()

    def start_processing_thread(self):
        if not self.selected_language:
            self.status_var.set("Please select a language first!")
            return
            
        self.start_button.config(state=tk.DISABLED)
        self.select_button.config(state=tk.DISABLED)
        self.select_output_button.config(state=tk.DISABLED)
        self.persian_button.config(state=tk.DISABLED)
        self.hebrew_button.config(state=tk.DISABLED)
        self.english_button.config(state=tk.DISABLED)
        self.log_text.delete('1.0', tk.END)
        self.update_progress(0, f"Starting {self.selected_language.title()} processing...")
        
        thread = threading.Thread(target=self.run_main_logic)
        thread.daemon = True
        thread.start()

    def run_main_logic(self):
        try:
            main_logic(self.audio_file_path, self.output_dir_path, self.selected_language, self)
            self.update_progress(100, "ALL TASKS HAVE BEEN PERFORMED SUCCESSFULLY!")
            print("\n" + "="*80)
            print("🎉 ALL TASKS HAVE BEEN PERFORMED SUCCESSFULLY!")
            print(f"📁 PLEASE FIND YOUR FILES IN THE SELECTED DIRECTORY:")
            print(f"   {self.output_dir_path}")
            print("="*80)
        except Exception as e:
            self.update_progress(0, "An error occurred. Check log for details.")
            print("\n--- A CRITICAL ERROR OCCURRED ---")
            print(traceback.format_exc())
        finally:
            self.start_button.config(state=tk.NORMAL)
            self.select_button.config(state=tk.NORMAL)
            self.select_output_button.config(state=tk.NORMAL)
            self.persian_button.config(state=tk.NORMAL)
            self.hebrew_button.config(state=tk.NORMAL)
            self.english_button.config(state=tk.NORMAL)

    class TextRedirector(object):
        def __init__(self, widget, tag="stdout"):
            self.widget = widget
            self.tag = tag
        def write(self, str):
            self.widget.insert(tk.END, str, (self.tag,))
            self.widget.see(tk.END)
        def flush(self): pass

# --- Backend Logic ---
def convert_to_clean_wav_ffmpeg(input_path, output_filename="temp_english_clean.wav"):
    """
    Uses direct ffmpeg call to convert any input audio/video into a clean,
    standardized WAV file optimized for English Whisper processing.
    """
    print(f"--- Creating clean WAV for English processing: {os.path.basename(input_path)} ---")
    clean_wav_path = os.path.join(get_base_path(), output_filename)

    if not FFMPEG_AVAILABLE or not FFMPEG_PATH:
        print("ERROR: FFmpeg not available. Cannot convert audio.")
        print("Please ensure ffmpeg.exe is in the ffmpeg folder or install FFmpeg system-wide.")
        return None

    try:
        # Command to convert any input to 16kHz mono WAV
        command = [
            FFMPEG_PATH,
            '-i', input_path,    # Input file
            '-ar', '16000',      # Set audio rate to 16kHz
            '-ac', '1',          
            '-y',                
            clean_wav_path
        ]
        
        print(f"Running FFmpeg command: {' '.join(command[:3])} ... (input: {os.path.basename(input_path)})")
        
        # Run the command and capture output
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if os.path.exists(clean_wav_path):
            print("-> Clean WAV created successfully for English processing.")
            return clean_wav_path
        else:
            print("ERROR: FFmpeg completed but output file not found.")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"\n--- FFMPEG CONVERSION ERROR ---")
        try:
            error_msg = e.stderr if e.stderr else "Unknown ffmpeg error"
            print(f"FFmpeg error: {error_msg}")
        except:
            print("Error decoding ffmpeg error message")
        return None
    except Exception as e:
        print(f"\n--- AUDIO CONVERSION ERROR --- \nError: {e}")
        return None

def convert_audio_to_wav_pydub(input_path, output_filename="temp_audio.wav"):
    """Convert any audio/video file to WAV format using pydub with ffmpeg"""
    output_path = os.path.join(get_base_path(), output_filename)
    print(f"--- Converting '{os.path.basename(input_path)}' to WAV format ---")
    
    if not FFMPEG_AVAILABLE or not FFMPEG_PATH:
        print("ERROR: FFmpeg not available. Cannot convert audio.")
        print("Please ensure ffmpeg.exe is in the ffmpeg folder or install FFmpeg system-wide.")
        return None
    
    try:
        # Force pydub to use our specific FFmpeg paths
        print(f"Configuring pydub with FFmpeg: {FFMPEG_PATH}")
        print(f"Configuring pydub with FFprobe: {FFPROBE_PATH}")
        
        # Set the paths directly in AudioSegment
        AudioSegment.converter = FFMPEG_PATH
        AudioSegment.ffmpeg = FFMPEG_PATH
        AudioSegment.ffprobe = FFPROBE_PATH
        
        # Also set environment variables as backup
        os.environ['PATH'] = os.path.dirname(FFMPEG_PATH) + os.pathsep + os.environ.get('PATH', '')
        
        print("Loading audio file...")
        audio = AudioSegment.from_file(input_path)
        print("Converting to mono 16kHz...")
        audio = audio.set_channels(1).set_frame_rate(16000)
        print("Exporting WAV file...")
        audio.export(output_path, format="wav")
        
        if os.path.exists(output_path):
            print("-> Conversion successful.")
            return output_path
        else:
            print("ERROR: Conversion completed but output file not found.")
            return None
            
    except Exception as e:
        print(f"\n--- AUDIO CONVERSION ERROR --- \nError: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Try fallback method using direct FFmpeg subprocess
        print("Attempting fallback conversion using direct FFmpeg...")
        try:
            fallback_result = convert_to_clean_wav_ffmpeg(input_path, output_filename)
            if fallback_result:
                print("-> Fallback conversion successful.")
                return fallback_result
        except Exception as fallback_error:
            print(f"Fallback conversion also failed: {fallback_error}")
        
        traceback.print_exc()
        return None

def perform_persian_transcription(audio_path):
    """Persian-specific transcription using Vosk model"""
    print("\n--- PERSIAN TRANSCRIPTION (Vosk Model) ---")
    
    try:
        # Check if Vosk model exists
        if not os.path.exists(persian_vosk_model_path):
            print(f"ERROR: Persian Vosk model not found at: {persian_vosk_model_path}")
            print("Please download the vosk-model-fa-0.42 model and place it in the models/vosk/ directory")
            return None, None
        
        # Convert audio to proper format for Vosk (if not already done)
        print("Converting audio for Vosk processing...")
        persian_wav_path = convert_audio_to_wav_pydub(audio_path, "temp_persian.wav")
        if not persian_wav_path:
            return None, None
        
        print("Loading Persian Vosk model...")
        model = vosk.Model(persian_vosk_model_path)
        rec = vosk.KaldiRecognizer(model, 16000)
        
        print("Processing Persian audio in chunks...")
        wf = wave.open(persian_wav_path, 'rb')
        chunk_size = 1000  # Adjust as needed
        results = []
        
        # Create word objects to match the expected format
        class Word:
            def __init__(self, word, start_time=0.0):
                self.word = word
                self.start = start_time
        
        word_objects = []
        current_time = 0.0
        
        while True:
            data = wf.readframes(chunk_size)
            if len(data) == 0:
                break
            
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if 'text' in result and result['text'].strip():
                    results.append(result['text'])
                    # Create word objects for each word in the result
                    words_in_chunk = result['text'].strip().split()
                    for word in words_in_chunk:
                        word_objects.append(Word(word, current_time))
                        current_time += 0.5  # Approximate timing
            else:
                partial = json.loads(rec.PartialResult())
                # Update timing for partial results
                current_time += 0.1
        
        # Get final result
        final_result = json.loads(rec.FinalResult())
        if 'text' in final_result and final_result['text'].strip():
            results.append(final_result['text'])
            # Add final words
            final_words = final_result['text'].strip().split()
            for word in final_words:
                word_objects.append(Word(word, current_time))
                current_time += 0.5
        
        wf.close()
        
        full_text = ' '.join(results).strip()
        print(f"-> Persian transcription completed. Total words: {len(word_objects)}")
        print(f"-> Sample text: {full_text[:100]}..." if len(full_text) > 100 else f"-> Full text: {full_text}")
        
        # Clean up temporary file
        try:
            if os.path.exists(persian_wav_path):
                os.remove(persian_wav_path)
        except:
            pass
        
        # Return Persian language code and word objects
        return "fa", word_objects
        
    except Exception as e:
        print(f"\n--- PERSIAN TRANSCRIPTION ERROR ---")
        print(f"Error: {e}")
        traceback.print_exc()
        return None, None

def perform_hebrew_transcription(audio_path):
    """Hebrew-specific transcription using custom Hebrew Whisper model"""
    print("\n--- HEBREW TRANSCRIPTION (Hebrew Whisper Model) ---")
    
    try:
        # Check if Hebrew Whisper model exists
        if not os.path.exists(hebrew_whisper_model_path):
            print(f"ERROR: Hebrew Whisper model not found at: {hebrew_whisper_model_path}")
            print("Please ensure the Hebrew-whisper model is in the models/Hebrew-whisper/ directory")
            return None, None
        
        print("Loading Hebrew Whisper model...")
        model = WhisperModel(hebrew_whisper_model_path, compute_type="int8", device="cpu")
        
        print(f"Transcribing Hebrew audio from: {os.path.basename(audio_path)}")
        print("Forcing Hebrew language detection...")
        
        # Transcribe with forced Hebrew language
        segments_generator, info = model.transcribe(audio_path, beam_size=5, language="he")
        segments = list(segments_generator)
        
        print(f"-> Detected language: {info.language}")
        print(f"-> Language probability: {info.language_probability:.2f}")
        
        # Create word objects from segments
        class Word:
            def __init__(self, word, start_time=0.0):
                self.word = word
                self.start = start_time
        
        word_objects = []
        all_text_parts = []
        
        for segment in segments:
            segment_text = segment.text.strip()
            if segment_text:
                all_text_parts.append(segment_text)
                # Split segment into words and create word objects
                words_in_segment = segment_text.split()
                segment_duration = segment.end - segment.start
                word_duration = segment_duration / len(words_in_segment) if words_in_segment else 0
                
                for i, word in enumerate(words_in_segment):
                    word_start_time = segment.start + (i * word_duration)
                    word_objects.append(Word(word, word_start_time))
        
        # Collect all text
        all_text = " ".join(all_text_parts)
        
        print(f"-> Hebrew transcription completed. Total segments: {len(segments)}")
        print(f"-> Total words: {len(word_objects)}")
        print(f"-> Sample text: {all_text[:100]}..." if len(all_text) > 100 else f"-> Full text: {all_text}")
        
        # Return Hebrew language code and word objects
        return "he", word_objects
        
    except Exception as e:
        print(f"\n--- HEBREW TRANSCRIPTION ERROR ---")
        print(f"Error: {e}")
        traceback.print_exc()
        return None, None

def perform_english_transcription(audio_path):
    """English-specific transcription using custom English Whisper model"""
    print("\n--- ENGLISH TRANSCRIPTION (Custom English Whisper Model) ---")
    
    try:
        # Check if English Whisper model exists
        if not os.path.exists(english_whisper_model_path):
            print(f"ERROR: English Whisper model not found at: {english_whisper_model_path}")
            print("Please ensure the Eng-whisper model is in the models/Eng-whisper/ directory")
            return None, None
        
        # Convert to clean WAV for optimal processing
        clean_audio_path = convert_to_clean_wav_ffmpeg(audio_path)
        if not clean_audio_path:
            print("ERROR: Failed to convert audio to clean WAV format")
            return None, None
        
        print("Loading English Whisper model and processor...")
        processor = AutoProcessor.from_pretrained(english_whisper_model_path)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(english_whisper_model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"-> Model loaded successfully on {device}")
        
        print("Loading audio file with torchaudio...")
        waveform, sample_rate = torchaudio.load(clean_audio_path)
        
        # Split into chunks for processing
        num_samples = waveform.shape[1]
        chunk_samples = english_chunk_duration * sample_rate
        num_chunks = math.ceil(num_samples / chunk_samples)
        print(f"Audio split into {num_chunks} chunks of {english_chunk_duration} seconds each.")
        
        # Create word objects to match expected format
        class Word:
            def __init__(self, word, start_time=0.0):
                self.word = word
                self.start = start_time
        
        word_objects = []
        all_text_parts = []
        
        # Process each chunk
        for i in range(num_chunks):
            start_sample = i * chunk_samples
            end_sample = min((i + 1) * chunk_samples, num_samples)
            chunk = waveform[:, start_sample:end_sample]
            chunk_start_time = i * english_chunk_duration
            
            print(f"Processing chunk {i+1}/{num_chunks}...")
            
            # Preprocess the chunk
            inputs = processor(chunk.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features.to(device)
            
            # Perform inference
            with torch.no_grad():
                generated_tokens = model.generate(inputs)
                text = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            
            chunk_text = text.strip()
            if chunk_text:
                all_text_parts.append(chunk_text)
                
                # Create word objects for this chunk
                words_in_chunk = chunk_text.split()
                chunk_duration = (end_sample - start_sample) / sample_rate
                word_duration = chunk_duration / len(words_in_chunk) if words_in_chunk else 0
                
                for j, word in enumerate(words_in_chunk):
                    word_start_time = chunk_start_time + (j * word_duration)
                    word_objects.append(Word(word, word_start_time))
            
            print(f"✅ Transcribed chunk {i+1}/{num_chunks}")
        
        # Combine all text
        full_text = " ".join(all_text_parts)
        
        print(f"-> English transcription completed. Total chunks: {num_chunks}")
        print(f"-> Total words: {len(word_objects)}")
        print(f"-> Sample text: {full_text[:100]}..." if len(full_text) > 100 else f"-> Full text: {full_text}")
        
        # Clean up temporary file
        try:
            if os.path.exists(clean_audio_path):
                os.remove(clean_audio_path)
                print("Temporary audio file cleaned up.")
        except:
            pass
        
        # Return English language code and word objects
        return "en", word_objects
        
    except Exception as e:
        print(f"\n--- ENGLISH TRANSCRIPTION ERROR ---")
        print(f"Error: {e}")
        traceback.print_exc()
        
        # Clean up temporary file in case of error
        try:
            clean_audio_path = os.path.join(get_base_path(), "temp_english_clean.wav")
            if os.path.exists(clean_audio_path):
                os.remove(clean_audio_path)
        except:
            pass
        
        return None, None

def perform_audio_analysis(audio_path, selected_language):
    """Route to appropriate transcription method based on selected language"""
    print(f"\n--- STAGE 1: Transcription ({selected_language.title()}) ---")
    
    if selected_language == "persian":
        return perform_persian_transcription(audio_path)
    elif selected_language == "hebrew":
        return perform_hebrew_transcription(audio_path)
    elif selected_language == "english":
        return perform_english_transcription(audio_path)
    else:
        print(f"ERROR: Unsupported language '{selected_language}'")
        return None, None

def align_transcript(words):
    """Simple transcript alignment without speaker diarization"""
    if not words:
        return ""
    
    return " ".join(word.word for word in words).strip()

def run_llm_task(llm, system_prompt, user_prompt, task_name="Processing"):
    print(f"\n--- Starting AI Task: {task_name} ---")
    start_time = time.time()
    try:
        response = llm.create_chat_completion(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_tokens=4096, temperature=0.1
        )
        result = response['choices'][0]['message']['content'].strip() if response and 'choices' in response else ""
        end_time = time.time()
        print(f"--- Finished AI Task in {end_time - start_time:.2f} seconds ---")
        return result
    except Exception as e:
        print(f"ERROR during {task_name}: {e}")
        traceback.print_exc()
        return ""

def translate_in_chunks(llm, text_to_translate, lang_name, task_name, chunk_size=300):
    """Splits text into chunks and translates each one."""
    words = text_to_translate.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    
    translated_chunks = []
    print(f"Translating text for '{task_name}' in {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        if not chunk.strip(): continue
        prompt = f"Your only task is to translate the following text into {lang_name}. Preserve formatting and headings exactly. Provide only the translated text.\n\nTEXT TO TRANSLATE:\n{chunk}"
        translated_chunk = run_llm_task(llm, "You are a professional translator.", prompt, f"{task_name} (Chunk {i+1})")
        translated_chunks.append(translated_chunk)
    
    separator = "\n\n" if "Report" in task_name else " "
    return separator.join(translated_chunks)

def main_logic(input_audio_file, output_dir, selected_language, app_instance):
    if not os.path.exists(input_audio_file):
        print(f"ERROR: Audio file '{input_audio_file}' does not exist.")
        return

    # Check if FFmpeg is available before starting
    if not FFMPEG_AVAILABLE:
        print("CRITICAL ERROR: FFmpeg not found!")
        print("Cannot proceed with audio conversion.")
        print(f"Please place ffmpeg.exe in: {FFMPEG_DIR}")
        print("Or install FFmpeg system-wide.")
        app_instance.update_progress(0, "ERROR: FFmpeg not found - Cannot convert audio")
        return

    app_instance.update_progress(5, "Converting audio to WAV...")
    
    # Try direct FFmpeg conversion first (more reliable)
    print("Attempting direct FFmpeg conversion...")
    wav_audio_file = convert_to_clean_wav_ffmpeg(input_audio_file, "temp_audio.wav")
    
    # If direct FFmpeg fails, try pydub as fallback
    if not wav_audio_file:
        print("Direct FFmpeg failed, trying pydub fallback...")
        wav_audio_file = convert_audio_to_wav_pydub(input_audio_file)
    
    if not wav_audio_file: 
        print("CRITICAL ERROR: Failed to convert audio file with both methods")
        app_instance.update_progress(0, "ERROR: Failed to convert audio file")
        return

    app_instance.update_progress(15, f"Transcribing using {selected_language.title()} model...")
    native_lang_code, all_words = perform_audio_analysis(wav_audio_file, selected_language)
    if not all_words: 
        print("ERROR: No transcription words returned from audio analysis")
        return
    
    full_transcript_native = " ".join(word.word for word in all_words).strip()

    path_full_native = os.path.join(output_dir, f"transcript_full_{native_lang_code}.txt")
    with open(path_full_native, "w", encoding="utf-8") as f: f.write(full_transcript_native)
    print(f"-> Native transcript saved.")

    app_instance.update_progress(40, "Loading translation model (WiNGPT)...")
    try:
        translator_llm = Llama(model_path=translator_llm_path, n_ctx=4096, n_gpu_layers=0, verbose=False)
        print("Translation model loaded successfully.")
    except Exception as e:
        print(f"\nCRITICAL ERROR: Failed to load Translation LLM. Error: {e}")
        return

    target_languages = {"en": "English", "he": "Hebrew", "fa": "Farsi"}
    
    for lang_code, lang_name in target_languages.items():
        if lang_code != native_lang_code:
            app_instance.update_progress(50, f"Translating to {lang_name}...")
            translation_full = translate_in_chunks(translator_llm, full_transcript_native, lang_name, f"Full Transcript to {lang_name}")
            path_full_translated = os.path.join(output_dir, f"transcript_full_{lang_code}.txt")
            with open(path_full_translated, "w", encoding="utf-8") as f: f.write(translation_full)
            print(f"-> {lang_name} full transcript saved.")
    
    del translator_llm

    app_instance.update_progress(70, "Loading analysis model (Phi-3.5 Mini)...")
    try:
        analyzer_llm = Llama(model_path=analyzer_llm_path, n_ctx=4096, n_gpu_layers=0, verbose=False)
        print("Analysis model loaded successfully.")
    except Exception as e:
        print(f"\nCRITICAL ERROR: Failed to load Analysis LLM. Error: {e}")
        return

    path_full_english = os.path.join(output_dir, "transcript_full_en.txt")
    if os.path.exists(path_full_english):
        app_instance.update_progress(75, "Generating English detail extraction...")
        with open(path_full_english, "r", encoding="utf-8") as f:
            english_full_transcript = f.read()

        english_analysis_prompt = f"""Analyze the following English transcript and generate a detailed report with these exact headings:
SUMMARY AND ACTION ITEMS
SENTIMENT ANALYSIS
NAMED ENTITY RECOGNITION (NER)
TIMELINE EXTRACTION
RECOMMENDATIONS

Transcript:
{english_full_transcript}"""
        english_report_text = run_llm_task(analyzer_llm, "You are an expert intelligence analyst.", english_analysis_prompt, "English Detail Extraction")
        path_report_en = os.path.join(output_dir, "Detailed_Report_ENG.txt")
        with open(path_report_en, "w", encoding="utf-8") as f: f.write(english_report_text)
        print(f"-> English Detail Report Saved.")
    else:
        english_report_text = ""

    del analyzer_llm

    app_instance.update_progress(90, "Translating detail report...")
    path_report_en_to_read = os.path.join(output_dir, "Detailed_Report_ENG.txt")
    if os.path.exists(path_report_en_to_read) and os.path.getsize(path_report_en_to_read) > 0:
        with open(path_report_en_to_read, "r", encoding="utf-8") as f:
            english_report_to_translate = f.read()
        
        try:
            translator_llm = Llama(model_path=translator_llm_path, n_ctx=4096, n_gpu_layers=0, verbose=False)
            print("Translation model re-loaded successfully.")
        except Exception as e:
            print(f"\nCRITICAL ERROR: Failed to re-load Translation LLM. Error: {e}")
            return
            
        report_lines = english_report_to_translate.splitlines()
        num_lines = len(report_lines)
        chunk_size = (num_lines + 2) // 3
        if chunk_size == 0: chunk_size = 1
        chunks = ["\n".join(report_lines[i:i + chunk_size]) for i in range(0, num_lines, chunk_size)]

        for lang_code, lang_name in target_languages.items():
            if lang_code != 'en':
                translated_chunks = []
                print(f"Translating report to {lang_name} in {len(chunks)} chunks...")
                for i, chunk in enumerate(chunks):
                    if not chunk.strip(): continue
                    prompt_report_chunk = f"Your only task is to translate the following text into {lang_name}. Preserve the formatting and headings exactly.\n\nTEXT TO TRANSLATE:\n{chunk}"
                    translated_chunk = run_llm_task(translator_llm, "You are a professional translator.", prompt_report_chunk, f"Translating Report Chunk {i+1} to {lang_name}")
                    translated_chunks.append(translated_chunk)
                
                full_translated_report = "\n\n".join(translated_chunks)
                path_report_translated = os.path.join(output_dir, f"Detailed_report_{lang_code}.txt")
                with open(path_report_translated, "w", encoding="utf-8") as f: f.write(full_translated_report)
                print(f"-> {lang_name} detailed report saved.")
        
        del translator_llm
    else:
        print("WARNING: English Detailed report not found or is empty. Skipping translation of report.")

    # Clean up temporary WAV files
    try:
        temp_files = [
            os.path.join(get_base_path(), "temp_audio.wav"),
            os.path.join(get_base_path(), "temp_english_clean.wav"),
            os.path.join(get_base_path(), "temp_persian.wav")
        ]
        
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Cleaned up: {os.path.basename(temp_file)}")
    except Exception as e:
        print(f"Note: Could not clean up some temporary files: {e}")

    print("\n--- PROCESSING COMPLETE ---")

if __name__ == "__main__":
    try:
        from faster_whisper import WhisperModel
        from llama_cpp import Llama
        from pydub import AudioSegment
        
        root = tk.Tk()
        app = App(root)
        root.mainloop()
    except ImportError as e:
        root = tk.Tk()
        root.withdraw()
        import tkinter.messagebox
        tkinter.messagebox.showerror("Fatal Error", f"A required library is missing:\n\n{e}\n\nPlease close the application, activate your .venv, and run the pip install commands.")
    except Exception as e:
        import tkinter.messagebox
        tkinter.messagebox.showerror("Fatal Error", f"A fatal error occurred on startup:\n\n{e}\n\nPlease check that all models are in the 'models' folder.")