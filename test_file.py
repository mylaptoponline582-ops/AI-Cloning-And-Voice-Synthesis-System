# importing libraries
import re
import numpy as np
import streamlit as st
import torch
from TTS.api import TTS
import os
import platform

if platform.system() == "Windows":
    os.add_dll_directory(
        r"C:\ffmpeg8\ffmpeg-8.1.2-full_build-shared\ffmpeg-8.1.2-full_build-shared\bin"
    )
#to keep everytime code to run in same environment
# os.add_dll_directory(
#     r"C:\ffmpeg8\ffmpeg-8.1.2-full_build-shared\ffmpeg-8.1.2-full_build-shared\bin"
# )
import shutil
import subprocess
from streamlit_mic_recorder import mic_recorder
import librosa
import soundfile as sf


st.set_page_config(
    page_title=("Ai voice and synthesis system"),
    layout="wide"
)
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "m1" not in st.session_state:
    st.session_state.m1=0
if "m2" not in st.session_state:
    st.session_state.m2 = "Not Ready"
if "m3" not in st.session_state:
    st.session_state.m3 = 0
if "m4" not in st.session_state:
    st.session_state.m4 = "Ready"


# Loading tts model


@st.cache_resource
def load_tts_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TTS(
        "tts_models/multilingual/multi-dataset/xtts_v2"
    ).to(device)

    return model, device


#audio pre-processing
def audio_processing(input_file):
    audio, sr = librosa.load(input_file, sr=22050, mono=True)
    audio, _ = librosa.effects.trim(audio)
    audio = librosa.util.normalize(audio)
    sf.write("processed_voice.wav", audio, sr)
    return "processed_voice.wav"


def generate_long_text(tts_model, text, voice, output_file):

# keep chunks safely below XTTS token limit
    MAX_CHARS = 700

    text = text.strip()
    chunks = []

# split into paragraphs first
    paragraphs = re.split(r'\n+', text)

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # Split paragraph into sentences
        sentences = re.split(
            r'(?<=[.!?])\s+',
            paragraph
        )

        current = ""

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue
# normal sentence fits
            if len(current) + len(sentence) + 1 <= MAX_CHARS:

                if current:
                    current += " " + sentence
                else:
                    current = sentence

            else:

                # Save current chunk
                if current:
                    chunks.append(current)
                    current = ""

                # If sentence itself is too long,
                # split it by words
                words = sentence.split()
                small_chunk = ""

                for word in words:

                    test = (
                        small_chunk + " " + word
                    ).strip()

                    if len(test) <= MAX_CHARS:

                        small_chunk = test

                    else:

                        if small_chunk:
                            chunks.append(small_chunk)

                        small_chunk = word

                if small_chunk:
                    current = small_chunk

        if current:
            chunks.append(current)

    audio_parts = []

    for chunk in chunks:

        audio = tts_model.tts(
            text=chunk,
            speaker_wav=voice,
            language="en",
            split_sentences=False
        )

        audio_parts.append(
            np.asarray(audio)
        )

    if not audio_parts:
        raise ValueError("No text was available for speech generation.")

    final_audio = np.concatenate(audio_parts)

    sf.write(
        output_file,
        final_audio,
        24000
    )


# css styling
st.markdown(
    """
<style>

/* 
   main titles
   */

h1{
    color:#a855f7 !important;
}

[data-testid="stHeading"] h1{
    color:#a855f7 !important;
}


/* 
   center alignment
   */

.center_caption{
    text-align:center;
}

.center_title{
    text-align:center;
}


/* 
  all containers borders
   */

.st-key-metric1_container,
.st-key-metric2_container,
.st-key-metric3_container,
.st-key-metric4_container,
.st-key-voice_container,
.st-key-speech_container{
    border:2px solid #a855f7 !important;
    border-radius:8px !important;
    box-sizing:border-box !important;
}


/* 
   normal buttons
  */

.stButton > button{
    background-color:#c084fc !important;
    color:white !important;
    border:1px solid #c084fc !important;
}

.stButton > button:hover{
    background-color:#a855f7 !important;
    color:white !important;
    border:1px solid #a855f7 !important;
}


/*
download button
   */

[data-testid="stDownloadButton"] button{
    background-color:#c084fc !important;
    color:white !important;
    border:1px solid #c084fc !important;
}

[data-testid="stDownloadButton"] button:hover{
    background-color:#a855f7 !important;
    color:white !important;
    border:1px solid #a855f7 !important;
}


/* 
   button text
    */

.stButton > button p,
[data-testid="stDownloadButton"] button p{
    color:white !important;
}


/* 
  button focus
   */

.stButton > button:focus,
.stButton > button:focus-visible,
[data-testid="stDownloadButton"] button:focus,
[data-testid="stDownloadButton"] button:focus-visible{
    border-color:#a855f7 !important;
    box-shadow:0 0 0 2px rgba(168,85,247,0.25) !important;
}

</style>
""",
    unsafe_allow_html=True
)




# main title


st.markdown(
    '<h1 class=center_title>Your Voice,Reimagined </h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="center_caption">Create a personalized voice profile and transform text into natural-sounding speech.</div>',
    unsafe_allow_html=True
)


#metrics(dashboard)


m1, m2, m3, m4 = st.columns(4)
with m1:
    with st.container(border=True, key="metric1_container"):
        m1_metric = st.empty()

        m1_metric.metric(
            label="Voice Recordings",
            value=1 if "voice_sample" in st.session_state else 0
        )


with m2:
    with st.container(border=True, key="metric2_container"):
        m2_metric = st.empty()

        m2_metric.metric(
            label="Voice Profile",
            value=st.session_state.m2,
        )

with m3:
    with st.container(border=True, key="metric3_container"):
        m3_metric = st.empty()

        m3_metric.metric(
            label="Generated Audio",
            value=st.session_state.m3,
        )

with m4:
    with st.container(border=True, key="metric4_container"):
        m4_metric = st.empty()

        m4_metric.metric(
            label="System Status",
            value=st.session_state.m4
        )


# two columns ( main containers)


col1, col2 = st.columns(2)


# column 1 , voice profile


with col1:
    with st.container(border=True, key="voice_container"):

        st.title("Create Your Voice Profile")

        st.caption(
            "Record or upload an authorized voice sample.The system will process the recording and create your personalized voice profile."
        )

        st.write("Voice Recording")

        st.info(
            "Ready to Record?\n\n"
            "Speak clearly in a quite environment..."
        )
        #
        # st.button("Start Recording")
        audio=mic_recorder(
            start_prompt="Start Recording",
            stop_prompt="Stop Recording",
            format="wav",
            just_once=True,
            use_container_width=True
        )
        if audio is not None:
            with open("voice_recording.wav","wb") as f:
                f.write(audio["bytes"])

            st.session_state.voice_sample=audio_processing("voice_recording.wav")
            st.session_state.m1=1
            st.success("Voice recording saved successfully.")
            st.audio(audio["bytes"],format="audio/wav")


        st.write("OR Upload an audio file")

        file_upload = st.file_uploader(
            "Upload an audio file",
            type=[
                "mp3",
                "wav",
                "m4a",
                "flac",
                "ogg",
                "mp4"
            ],
            key=f"upload_{st.session_state.uploader_key}"
        )

        # saving uploaded voice profile


        if file_upload is not None and "voice_sample" not in st.session_state:

            # Keep the actual uploaded extension
            file_extension = file_upload.name.split(".")[-1].lower()

            voice_sample = f"voice_sample.{file_extension}"

            with open(voice_sample, "wb") as f:
                f.write(file_upload.getbuffer())

            st.session_state.voice_sample =audio_processing(voice_sample)
            st.session_state.m1=1

            st.success("Voice sample uploaded successfully.")


        consent=st.checkbox(
            "I confirm that i am the owner of this recording or have obtained explicit permission from the voice owner to use it for voice cloning and speech synthesis."
        )
        voice_profile=st.button("Create Voice Profile")
        if voice_profile:
            if "voice_sample" not in st.session_state:
                st.warning("Please record or upload a voice sample first.")

            elif not consent:
                st.warning("Please confirm that you have permission to use this voice.")

            else:
                st.session_state.m4 = "Processing"
                try:
                    with st.spinner("Voice Profile Processing..."):
                     st.session_state.voice_profile = st.session_state.voice_sample
                     st.session_state.m2 = "Ready"
                     st.session_state.m4 = "Ready"
                    st.success("Voice profile is ready")
                except Exception as e:
                      st.session_state.m4="Error"
                      st.error("Voice generation failed.")
                      st.exception(e)

        if "voice_sample" in st.session_state:
                if st.button("Delete Recording"):
                    os.remove(st.session_state.voice_sample)
                    del st.session_state.voice_sample
                    st.success("Recording Deleted")

                    st.session_state.uploader_key += 1
                    st.session_state.m1=0
                    st.session_state.m2 = "Not Ready"
                    st.rerun()
        m1_metric.metric(
            label="Voice Recordings",
            value=1 if "voice_sample" in st.session_state else 0
        )
        m2_metric.metric(
            label="Voice Profile",
            value=st.session_state.m2,
        )

# column 2 ,text to speech


with col2:
    with st.container(border=True, key="speech_container"):

        st.title("Generate Speech")

        st.caption(
            "Enter your text and generate natural speech.using your personalized voice profile."
        )

        st.text_area(
            label="Text to Speech",
            placeholder="Start typing here or paste any text you want to turn into speech",
            key="speech_text"
        )

        text = st.session_state.speech_text

        generate_voice = st.button(
            "Generate Voice",
            key="generate_button"
        )


        # generating voice


        if generate_voice:

            # check text
            if not text.strip():

                st.warning(
                    "Please enter some text before generating speech."
                )
            elif not consent:
                st.warning(
                    "Please confirm that you have permission to use this voice before generating speech."
                )
            # check voice sample
            elif "voice_sample" not in st.session_state:

                st.warning(
                    "Please upload a voice sample before generating speech."
                )

            # check voice file
            elif not os.path.exists(
                st.session_state.voice_sample
            ):

                st.warning(
                    "Voice sample file was not found. Please upload it again."
                )

            else:

                try:

                    with st.spinner(
                        "Generating your voice..."
                    ):

                        tts_model, device = load_tts_model()

                        output_file_path = "generated.wav"

                        generate_long_text(
                            tts_model,
                            text,
                            st.session_state.voice_sample,
                            output_file_path
                        )


                    st.success(
                        "Voice generated successfully!"
                    )
                    st.session_state.m3 += 1

                    st.audio(
                        output_file_path
                    )

                except Exception as e:

                    st.error(
                        "Voice generation failed."
                    )

                    st.exception(e)


        # download generated button


        if os.path.exists("generated.wav"):

            with open(
                "generated.wav",
                "rb"
            ) as f:

                audio_data = f.read()

            st.download_button(
                "Download generated voice file",
                data=audio_data,
                file_name="Generated audio.wav",
                mime="audio/wav",
            )

        m3_metric.metric(
            label="Generated Audio",
            value=st.session_state.m3,
        )
        m4_metric.metric(
            label="System Status",
            value=st.session_state.m4,
        )
