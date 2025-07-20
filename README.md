RLHF-MathTutor

Personalised voice‑first mathematics tutor for K‑12 learners, built on Llama‑3 (or Mistral‑7B) fine‑tuned with RLHF – teraz z obsługą mowy (STT ↔ TTS) zamiast czatu z człowiekiem‑nauczycielem



Table of Contents

Overview

Key Features

Project Structure

Quick Start

Training & Evaluation

Web & Voice UI

Monitoring

Roadmap

Contributing

License

Citation

Overview

RLHF‑MathTutor to badawczy system AI Teaching Assistant do nauki matematyki od klas 4 SP po liceum.  Środowisko działa w dwóch warstwach pomocy:

Mini‑lekcja (pre‑brief) – krótkie wprowadzenie teoretyczne + przykład.

Spersonalizowane zadania i feedback generowane przez fine‑tuned Llama/Mistral (SFT → RLHF).

💡 NOWOŚĆ: interfejs głosowy – uczeń może mówić do tutora, a ten odpowiada syntetyzowaną mową.  STT=Whisper‑large‑v3 (transkrypcja), TTS=Coqui‑XTTS (neuronalna synteza). Dokładność rozpoznania wzmacniana kontekstem zadania.

Key Features

End‑to‑end pipeline: ETL → SFT → Reward Model → RLHF → API + UI.

Matematyczne zbiory danych: GSM8K, MATH (JSON mirror), MathQA‑CSV.

Whiteboard mode: AI rysuje równania / wykresy (SVG, Plotly) i odczytuje pismo ucznia (Pix2LaTeX / TrOCR).

Voice interface: WebSocket kanał audio 16 kHz PCM ↔ ASR ↔ LLM ↔ TTS → strumień audio (OPAQUE / Opus).

fallback: klasyczny chat tekstowy.

Algorytmy RL: PPO oraz Bandit‑DPO (konfigurowalne YAML‑em).

FastAPI back‑end (vLLM / ggml) + Next.js front‑end (LaTeX editor + whiteboard + voice widget).

Monitoring: Superset / Grafana + Weights & Biases.

Project Structure

.
├── data/                # raw & processed math datasets
├── notebooks/           # EDA & reports
├── src/
│   ├── etl/             # ingest & cleaning scripts
│   ├── training/        # SFT & RLHF loops
│   ├── reward/          # reward model training / eval
│   ├── api/             # FastAPI server
│   ├── voice/           # STT (Whisper), TTS (XTTS) micro‑services
│   ├── ui/              # Next.js front‑end (LaTeX + whiteboard + voice)
│   └── ocr/             # Pix2LaTeX / TrOCR service
├── configs/             # YAML configs for experiments
├── dashboard/           # Superset dashboard definitions
├── tests/               # unit & integration tests
└── README.md


Training & Evaluation

Metryki: final‑answer Accuracy, Step Accuracy, BLEU‑Explain, Avg ⭐.

Reproducibility: seeds & hyper‑params in configs/.

Tracking: W&B + MLflow artifacts.

Web & Voice UI

# API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
# Voice micro‑services
python src/voice/stt_server.py  --model whisper-large-v3 --port 9000
python src/voice/tts_server.py  --model xtts-v2         --port 9001
# Front‑end
yarn --cwd src/ui install
yarn --cwd src/ui dev

Monitoring

Superset dashboards live in dashboard/; import JSON definitions.

Roadmap



Contributing

Fork & create feature branch.

pre-commit run --all-files before PR.

Describe motivation, link to an open issue.

License

MIT License – see LICENSE.

Citation
@misc{rlhf_mathtutor_2025,
  title        = {RLHF Math Tutor},
  author       = {KcorUx},
  year         = {2025},
  howpublished = {\url{https://github.com/Kacper-KcorUx/rlhf-mathtutor}}
}
