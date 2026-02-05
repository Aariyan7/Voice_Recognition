import librosa
import numpy as np
import csv
import os
import soundfile
import pandas as pd


# pitch_jitter
# pitch_std
# energy_slope_variance
# energy_gap_variance
# speech_rate_variance
# pause_mean_duration
# pause_std_duration
# spectral_flux_spike_rate
# spectral_flux_std

frame_length = 400
hop_length = 160

def load_audio (audio_path):
    waveform, sampling_rate = librosa.load(audio_path, sr=16000, mono=True)
    return waveform, sampling_rate

# Extract pitch form the audio
#       -> pitch_mean
#       -> pitch_std
#       -> pitch_min
#       -> pitch_max
#       -> pitch_range
#       -> pitch_jitter
    
def extract_pitch (waveform, sampling_rate):
    f0 = librosa.yin(
        waveform,
        fmin = 50,
        fmax=400,
        sr=sampling_rate,
        hop_length=160
    )
    
    valid_f0 = f0[~np.isnan(f0)] # Remove all the NaN values
    
    if len(valid_f0) == 0:
        return {
            "pitch_mean":0.0,
            "pitch_std":0.0,
            "pitch_min":0.0,
            "pitch_max":0.0,
            "pitch_mean":0.0,
            "pitch_jitter":0.0,
        }
        
    return {
        "pitch_mean":float(np.mean(valid_f0)),
        "pitch_std":float(np.std(valid_f0)),
        "pitch_min":float(np.min(valid_f0)),
        "pitch_max":float(np.max(valid_f0)),
        "pitch_range":float((np.max(valid_f0) - (np.min(valid_f0)))),
        "pitch_jitter":float(np.mean(np.abs(np.diff(valid_f0)))),  
    }


# Extract the energy of the speaker for every single frame
#       -> speech_rate_variance     “On average, how long does the speaker stay silent?”
#       -> pause_mean_duration      “Are the pauses consistent or chaotic?”
#       -> pause_std_duration       Speech rate = how fast someone talks.
#       -> energy_gap_variance      Energy gap = difference in energy between adjacent frames.
#       -> energy_mean              
#       -> energy_std
#       -> energy_slope_variance
#       -> energy_max

def extract_energy(waveform, sample_rating):
    energy = librosa.feature.rms(
        y=waveform,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]
    
    energy_mean = np.mean(energy)
    energy_std = np.std(energy)
    energy_max = np.max(energy)
    energy_slope = np.diff(energy)
    energy_slope_variance = np.var(energy_slope)
    
    threshold = np.percentile(energy, 30)
    
    is_speech = energy > energy
    is_pause = energy <= energy
    
    speech_rates = []
    
    frame_per_second = sample_rating // hop_length   # Approx 100 frames per second
    
    for i in range(0, len(is_speech), frame_per_second):
        window = is_speech[i:i + frame_per_second]
        if len(window) > 0:
            speech_rates.append(np.mean(window))
        
    speech_rate_variance = np.var(speech_rates) 
    
    # Pass Mean Duration
    pause_durations = []
    current_pause = 0
    
    for frame in is_pause:
        if frame:
            current_pause += 1
        else:
            if current_pause > 0:
                pause_durations.append(current_pause)
                current_pause = 0
            
    if current_pause > 0 :
        pause_durations.append(current_pause)
    
    # Convert frames into seconds
    pause_durations_sec = np.array(pause_durations) * hop_length / sample_rating
    
    if len(pause_durations_sec) > 0:
        pause_mean_duration = np.mean(pause_durations_sec)
        pause_std_duration = np.std(pause_durations_sec)
    else:
        pause_mean_duration = 0.0
        pause_std_duration = 0.0
        
    # energy_gap_variance
    energy_gaps = []
    
    for i in range(1, len(energy)):
        if is_pause[i] and is_speech[i-1]:
            energy_gaps.append(energy[i - 1] - energy[i])
    
    if len(energy_gaps) > 0:
        energy_gap_variance = np.var(energy_gaps)
    else:
        energy_gap_variance = 0.0
    
    return {
        "energy_mean": energy_mean,
        "energy_std": energy_std,
        "energy_slope_variance": energy_slope_variance,
        "energy_max": energy_max,
        "speech_rate_variance": speech_rate_variance,
        "pause_mean_duration": pause_mean_duration,
        "pause_std_duration": pause_std_duration,
        "energy_gap_variance": energy_gap_variance 
    }
    
# Extract MFCC -> Mel-Frequency Cepstral Coefficients

def extract_mcff(waveform, sample_rating):
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sample_rating,
        n_mfcc= 13, 
        n_fft = 400,
        hop_length = hop_length
    )
    
    mfcc_mean = np.mean(mfcc)
    mfcc_std = np.std(mfcc)
    
    return {
        "mfcc_mean": mfcc_mean,
        "mfcc_std": mfcc_std
    }


# Extract Spectral flatness
#   -> Spectral_flatness_mean
#   -> Spectral_flatness_std

def extract_spectral_flatness(waveform):
    spectral_faltness = librosa.feature.spectral_flatness(
        y=waveform,
        n_fft = 400,
        hop_length=hop_length
    )[0]
    
    spectral_flatness_mean = np.mean(spectral_faltness)
    spectral_flatness_std = np.std(spectral_faltness)
    
    return {
        "Spectral_flatness_mean": spectral_flatness_mean,
        "Spectral_flatness_std": spectral_flatness_std
    }


# Extract Spectral Flux
#   -> spectral_flux_mean
#   -> spectral_flux_std
#   -> spectral_flux_max
#   -> spectral_flux_percentile
#   -> spectral_flux_spike_rate

def extract_spectral_flux (waveform, sample_rating):
    flux = librosa.onset.onset_strength(
        y=waveform,
        sr=sample_rating,
        hop_length = hop_length
    )
    
    spectral_flux_mean = np.mean(flux)
    spectral_flux_std = np.std(flux)
    spectral_flux_max = np.max(flux)
    spectral_flux_percentile = np.percentile(flux, 95)
    threshold = spectral_flux_mean + 2 * spectral_flux_std
    spectral_flux_spike_rate = np.mean(flux > threshold)
    
    return {
        "spectral_flux_mean": spectral_flux_mean,
        "spectral_flux_std": spectral_flux_std,
        "spectral_flux_max": spectral_flux_max,
        "spectral_flux_percentile": spectral_flux_percentile,
        "spectral_flux_spike_rate": spectral_flux_spike_rate   
    }

def extract_data_as_chunks(waveform, sample_rating):
    pitch_data = extract_pitch(waveform, sample_rating)
    energy_data = extract_energy(waveform=waveform, sample_rating=sample_rating)
    mfcc_data = extract_mcff(waveform=waveform, sample_rating=sample_rating)
    spectral_flatness_data = extract_spectral_flatness(waveform=waveform)
    spectral_flux_data = extract_spectral_flux(waveform=waveform, sample_rating=sample_rating)
    
    audio_data = {
        "pitch_data": pitch_data,
        "energy_data": energy_data,
        "mfcc_data": mfcc_data,
        "spectral_flatness_data": spectral_flatness_data,
        "spectral_flux_data": spectral_flux_data
    }
    
    return audio_data 

def extract_data(path):
    waveform, sample_rating = load_audio(path)
    pitch_data = extract_pitch(waveform, sample_rating)
    energy_data = extract_energy(waveform=waveform, sample_rating=sample_rating)
    mfcc_data = extract_mcff(waveform=waveform, sample_rating=sample_rating)
    spectral_flatness_data = extract_spectral_flatness(waveform=waveform)
    spectral_flux_data = extract_spectral_flux(waveform=waveform, sample_rating=sample_rating)
    
    audio_data = {
        "pitch_data": pitch_data,
        "energy_data": energy_data,
        "mfcc_data": mfcc_data,
        "spectral_flatness_data": spectral_flatness_data,
        "spectral_flux_data": spectral_flux_data
    }
    
    return audio_data    

def write_data_to_csv():
    directory_name = "training_dataset"
    data_num = 0
    test_data = {}
    col_names = ["label"]

    try:
        with open("training_data_eleven_labs.csv", "a+") as file:
            print("CSV file created Sucessfully!!!")
            writer = csv.writer(file)
            
            print("------------------------- Starting the Extraction of Data -------------------------")
            for dir in os.listdir(directory_name):
                if dir == "AI":
                    label = 1
                elif dir == "Human":
                    label = 0
            
                for audio in os.listdir(directory_name + "/" + dir):
                    path = directory_name + "/" + dir + "/" + audio
                    
                    try:
                        duration = soundfile.info(path).duration
                        if duration > 4 and duration <= 15:
                            audio_data = extract_data(path=path)
                            row = [label]
                            
                            for data in audio_data.keys():
                                for key in audio_data[data].keys():
                                    row.append(audio_data[data][key])
                                    if data_num == 0:
                                        col_names.append(key)        
                            if data_num == 0:
                                test_data["first_row"] = row
                            elif data_num == 25:
                                test_data["25"] = row
                                
                            elif data_num == 50:
                                test_data["50"] = row
                                
                            data_num += 1
                                    
                            writer.writerow(row)
                            print(f"Sucessfully added a row to the CSV for {audio}")
                            
                        elif duration > 15:
                            SR = 16000
                            CHUNK_SIZE = 15
                            MIN_SEC = 4
                            
                            waveform, sample_rating = load_audio(audio_path=path)
                            chunk_samples = SR * CHUNK_SIZE
                            min_samples = MIN_SEC * SR
                            
                            num_samples = len(waveform)
                            
                            for start in range(0, num_samples, chunk_samples):
                                end = start + chunk_samples
                                chunk = waveform[start:end]
                                
                                if len(chunk) < min_samples:
                                    continue
                                
                                audio_data = extract_data_as_chunks(waveform, sample_rating)
                                row = [label]
                            
                                for data in audio_data.keys():
                                    for key in audio_data[data].keys():
                                        row.append(audio_data[data][key])
                                        if data_num == 0:
                                            col_names.append(key)
                                                               
                                if data_num == 0:
                                    test_data["first_row"] = row
                                elif data_num == 25:
                                    test_data["25"] = row
                                elif data_num == 50:
                                    test_data["50"] = row
                                    
                                data_num += 1
                                        
                                writer.writerow(row)
                                print(f"Sucessfully added a row to the CSV for {audio}")
                    
                    except soundfile.LibsndfileError as er:
                        print("Something went wrong while calculating the duration of the audio: ", er)
                    
            print("------------------------- Data has been Extracted and uploaded Sucessfully -------------------------")
                
    except Exception as er:
        print("Something went wrong: ", er)
    
    for key in test_data.keys():
        print(test_data[key])
    
    return col_names


def extract_row_data(waveform, sample_rating):
    pitch_data = extract_pitch(waveform, sample_rating)
    energy_data = extract_energy(waveform=waveform, sample_rating=sample_rating)
    mfcc_data = extract_mcff(waveform=waveform, sample_rating=sample_rating)
    spectral_flatness_data = extract_spectral_flatness(waveform=waveform)
    spectral_flux_data = extract_spectral_flux(waveform=waveform, sample_rating=sample_rating)
    
    audio_data = {
        "pitch_data": pitch_data,
        "energy_data": energy_data,
        "mfcc_data": mfcc_data,
        "spectral_flatness_data": spectral_flatness_data,
        "spectral_flux_data": spectral_flux_data
    }

    row = [] 
    for data in audio_data.keys():
        for key in audio_data[data].keys():
            row.append(audio_data[data][key])
    
    return row 

     

def main():    
    col_names = write_data_to_csv()

    df = pd.read_csv('training_data_eleven_labs.csv', names=col_names)
    print("Head: ", df.head())
    print("Tail: ", df.tail())
    
    df.to_csv('training_data_eleven_labs.csv', index=False)
    
    print("Data generated sucessfully!!!")
    
if __name__ == "__main__":
    main()

    


