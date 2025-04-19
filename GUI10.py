import streamlit as st
import concurrent.futures
import requests
import json
import base64
from main import Main
from main import Main, convert_midi_to_audio, list_soundfonts
from midi2audio import FluidSynth
from pydub import AudioSegment

#python -m streamlit run GUI10.py

st.set_page_config(page_title="Natural Audio to MIDI Converter", page_icon="", layout="wide")

# Define a global variable to track the state
conversion_started = False

# Header Section
with st.container():
    st.subheader("BCS PGD Level Project")
    st.title("Natural Audio to MIDI Converter")
    st.title("Withanage Linuk Dinendra Perera 995067448")
    st.write("Audio to MIDI converter using Fast Fourier Transform FFT based on Python for BCS PGD Level Project")

# About the project
with st.container():
    st.write("---")  # divider
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("Upload a file for conversion")
        file = st.file_uploader("Please Upload MP3 or Wav file", type=["mp3", "wav"])
        if file is not None:
            st.audio(file, format='audio/wav' if file.name.endswith('.wav') else 'audio/mpeg')
        #threshhold = st.text_input("Select the threshold from 0.1 to 1 based on instrument noise: ")
        def validate_input(threshhold):
            if threshhold < 0.1 or threshhold > 1:
                st.error("Error: Input value must be between 0.1 and 1.")
                return False
            return True

        #st.title("User Input Validation")

        threshhold = st.number_input("Select the threshold from 0.1 to 1 based on instrument noise: ", min_value=0.1, max_value=1.0, step=0.01)

        if validate_input(threshhold):
            st.success(f"Valid input: {threshhold}")        
        st.write("You entered:", threshhold)
        numberOfNotes = st.text_input("Maximum number of notes to detect: ")
        st.write("You entered:", numberOfNotes)

        if st.button("Start Conversion"):
            conversion_started = True

    # Process file and parameters only if conversion has started
    if conversion_started:
        if file is not None:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Extract the file name from the UploadedFile object
                file_name = file.name

                m = Main(file_name)

                splitLengthinSeconds = 0.2
                # multithreading
                song = m.split(splitLengthinSeconds, sampleBeginning=0, sampleEnd=60)

                notes = list(executor.map(lambda data: m.thread(data, float(threshhold), int(numberOfNotes)), song))

                # every list in the list is a triad or more notes played at the same time [[0,12]] for example is an " a' " and an " a'' "
                notes = m.removeRepetitions(notes)
                notes = m.noteNames(notes)
                
    # Radio buttons
    output_option = st.radio("Select an output option:", ("MIDI", "Play with Fluid Synth", "Tablature","Accuracy"))
    if conversion_started:
        if output_option == "MIDI":
            st.write("Find converted MIDI file below")
            output_file = "output.mid"
            m.notes_to_midi(notes,output_file)
            st.write("Conversion to MIDI Complete!")
            st.write("Download the MIDI file:")
            st.markdown(f" [Download {output_file}](data:audio/midi;base64,{base64.b64encode(open(output_file, 'rb').read()).decode()})")
        elif output_option == "Play with Fluid Synth":
            st.write("Find converted MIDI file and Soundfont below")
            output_file = "output.mid"
            m.notes_to_midi(notes,output_file)
            st.write("Conversion to MIDI Complete!")
            st.write("Select a Soundfont:")
            soundfont = st.selectbox("Choose a Soundfont",["Soundfont1.sf2"])
            audio_file = convert_midi_to_audio,list_soundfonts(output_file, soundfont)
            audio = AudioSegment.from_wav(audio_file)
            st.audio(audio_file, format="audio/wav")
            st.write("Download the MIDI file:")
            st.markdown(f" [Download {output_file}](data:audio/midi;base64,{base64.b64encode(open(output_file, 'rb').read()).decode()})")
        elif output_option == "Tablature":
            st.write("Find tablature below")
            st.write("Length of notes:", len(notes))
            st.write("Notes:", notes)
        elif output_option == "Accuracy":
            st.write("Pitch accuracy of the input sample is :")
            st.write("")

    st.write("##")