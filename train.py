import numpy as np
import pandas as pd
import os
import soundfile
import base64
import io
import librosa

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

from data_extraction import extract_row_data, load_audio

def audio_to_base64(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    return audio_base64


def base64_mp3_to_waveform(audio_base64, target_sr=16000):
    # 1️⃣ Decode Base64 → raw MP3 bytes
    mp3_bytes = base64.b64decode(audio_base64)

    # 2️⃣ Wrap bytes as a file-like object
    audio_buffer = io.BytesIO(mp3_bytes)

    # 3️⃣ Decode MP3 → waveform
    waveform, sr = librosa.load(
        audio_buffer,
        sr=target_sr,
        mono=True
    )

    return waveform, sr

def check_audio(byte46_audio):

    # =====================================================
    # 1. LOAD TRAINING DATA
    # =====================================================

    df = pd.read_csv("training_data_eleven_labs.csv")
    df = df.drop_duplicates()

    X = np.array(df.drop("label", axis=1))
    y = np.array(df["label"])

    print("Training samples:", len(df))


    # =====================================================
    # 2. TRAIN / TEST SPLIT
    # =====================================================

    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        stratify=y,
        random_state=42
    )


    # =====================================================
    # 3. FEATURE SCALING
    # =====================================================

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)


    # =====================================================
    # 4. TRAIN LOGISTIC REGRESSION
    # =====================================================

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train_scaled, y_train)


    # =====================================================
    # 5. EVALUATION
    # =====================================================

    y_pred = model.predict(x_test_scaled)
    y_prob = model.predict_proba(x_test_scaled)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC :", roc_auc_score(y_test, y_prob))
    print("LogLoss :", log_loss(y_test, y_prob))


    # =====================================================
    # 6. FEATURE NAMES
    # =====================================================

    feature_names = [
        "pitch_std",
        "pitch_jitter",
        "energy_slope_variance",
        "energy_gap_variance",
        "speech_rate_variance",
        "pause_mean_duration",
        "pause_std_duration",
        "spectral_flux_spike_rate",
        "spectral_flux_std",
        "mfcc_std"
    ]


    # =====================================================
    # 7. FEATURE → HUMAN MEANING MAP
    # =====================================================

    FEATURE_EXPLANATIONS = {
        "pitch_jitter": "unstable vocal fold vibration",
        "pitch_std": "natural pitch variation",
        "energy_slope_variance": "irregular loudness changes",
        "energy_gap_variance": "abrupt emphasis patterns",
        "speech_rate_variance": "inconsistent speaking speed",
        "pause_mean_duration": "natural thinking pauses",
        "pause_std_duration": "irregular pause timing",
        "spectral_flux_spike_rate": "frequent articulation changes",
        "spectral_flux_std": "uneven spectral transitions",
        "mfcc_std": "variation in vocal tract characteristics"
    }


    # =====================================================
    # 8. TESTING + EXPLAINABILITY
    # =====================================================

    def test_audio_sample():

        def explain_prediction(x_scaled):
            """
            Returns top contributing feature and explanation sentence
            """
            weights = model.coef_[0]
            contributions = x_scaled * weights

            ranked = sorted(
                zip(feature_names, contributions),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            return ranked


        def generate_explanation(reasons, ai_score, human_score):
            if ai_score > human_score:
                label = "AI-generated"
                direction = "AI"
            else:
                label = "Human-generated"
                direction = "Human"

            for feature, contrib in reasons:
                if (contrib > 0 and direction == "AI") or (contrib < 0 and direction == "Human"):
                    meaning = FEATURE_EXPLANATIONS.get(feature, "distinct speech characteristics")
                    return (
                        f"This audio is classified as {label} because it exhibits "
                        f"{meaning}, which strongly influenced the model’s decision."
                    )

            return f"This audio is classified as {label} based on overall speech behavior."


        base_path = "testing_dataset"

        # for label_dir in os.listdir(base_path):
        #     folder_path = os.path.join(base_path, label_dir)

        #     for audio in os.listdir(folder_path):
        #         file_path = os.path.join(folder_path, audio)

        waveform, sr = base64_mp3_to_waveform(byte46_audio)
        # duration = soundfile.info(file_path).duration

        ai_score = None
        human_score = None
        reasons = None

        # -------- Case 1: 4s – 15s --------
        # if 4 < duration <= 15:
        row = extract_row_data(waveform, sr)
        x_test = np.array(row).reshape(1, -1)
        x_test_scaled = scaler.transform(x_test)

        prob = model.predict_proba(x_test_scaled)[0]
        ai_score = round(prob[1] * 100, 2)
        human_score = round(prob[0] * 100, 2)

        reasons = explain_prediction(x_test_scaled[0])

        # -------- Case 2: >15s (chunking) --------
        # elif duration > 15:
        #     SR = 16000
        #     CHUNK_SEC = 15
        #     MIN_SEC = 4

        #     chunk_samples = SR * CHUNK_SEC
        #     min_samples = SR * MIN_SEC

        #     ai_probs = []
        #     human_probs = []
        #     last_scaled = None

        #     for start in range(0, len(waveform), chunk_samples):
        #         chunk = waveform[start:start + chunk_samples]

        #         if len(chunk) < min_samples:
        #             continue

        #         row = extract_row_data(chunk, sr, audio)
        #         x_test = np.array(row).reshape(1, -1)
        #         x_test_scaled = scaler.transform(x_test)

        #         prob = model.predict_proba(x_test_scaled)[0]
        #         ai_probs.append(prob[1] * 100)
        #         human_probs.append(prob[0] * 100)

        #         last_scaled = x_test_scaled

        #     if len(ai_probs) == 0:
        #         continue

        #     ai_score = round(np.mean(ai_probs), 2)
        #     human_score = round(np.mean(human_probs), 2)

        #     reasons = explain_prediction(last_scaled[0])

        explanation = generate_explanation(reasons, ai_score, human_score)

        return {
            "ai_score":ai_score,
            "human_score":human_score,
            "explanation": explanation
        }


    # =====================================================
    # 9. RUN TESTING
    # =====================================================

    result = test_audio_sample()
    return result
     

