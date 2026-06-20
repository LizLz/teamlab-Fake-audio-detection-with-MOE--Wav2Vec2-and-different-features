# Audio Deepfake & Spoofing Detection

**Authors:** Zheng, Felipe  
**Course:** Team Lab — Phonetics Track  
**Presentation:** Final results and insights are available in [`team_lab_poster_portrait.pdf`](./team_lab_poster_portrait.pdf).

---

## 📌 Research Overview
This project focuses on identifying synthetic voice and deepfake audio using the **ASVspoof 2019 Logical Access (LA)** dataset. Our core research explores:
* **Model Complexity:** Can a shallow CNN achieve acceptable performance for this task?
* **Feature Optimization:** Which acoustic features are most effective at identifying spoofed signals?
* **Preprocessing Impact:** How do targeted feature processing techniques influence overall model performance?

---

## ⚡ Quick Start

1. **Download the Data:** Obtain the required ASVspoof 2019 corpus from the [Official ASVspoof Portal](https://www.asvspoof.org/index2019.html).
2. **Feature Extraction:** Run the respective processing scripts inside the `feature_extraction/` folder to pre-process your raw audio files.
3. **Execute Core Baselines:**
   * Run `baseline_model/lfcc_delta_v5_base_model.ipynb` to evaluate our top-performing baseline using hand-crafted acoustic features.
   * Run `baseline_model/wav2vec_updated_base_model.ipynb` to evaluate the system using SSL-derived Wav2Vec 2.0 features.
4. **Explore Advanced Architectures:**
   * Navigate to `advanced_baseline_model/LSTM+attention/lfcc_lstm_attention_advanced_model.ipynb` to run the temporal Bi-LSTM + Attention architecture.
   * Review our macro-ensemble architectures and routing mechanisms inside the `LCNN+MOE/` directory.

---

## 📂 Repository Structure

```text
Deepfake_Detection/
├── advanced_baseline_model/
│   ├── lCNN+MOE/             # Light CNN blocks, Mixture of Experts (MoE) wrappers, and training 
│   └── LSTM+attention/       # Sequential modeling baseline featuring Bi-LSTM + Attention layers
├── baseline_model/           # Shared neural structural variations evaluated across diverse audio features
│   └── ...
├── feature_extraction/       # Scripts dedicated to converting raw waveforms to target feature representations
├── utility/                  # Shared helper scripts, custom loss criteria, and evaluation functions
├── report/                   # LaTeX source files and fully compiled research reports
├── README.md                 # Project roadmap and repository documentation
└── requirements.txt          # Python dependency specifications and package listings