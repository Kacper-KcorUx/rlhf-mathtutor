RLHF-MathTutor
Personalised mathematics tutor for K‑12 learners, built on Llama‑3 fine‑tuned with RLHF
– now with human‑teacher chat escalation
Table of Contents
Overview
Key Features
Project Structure
Quick Start
Training & Evaluation
Web UI
Monitoring
Roadmap
Contributing
License
Citation
Overview
RLHF‑MathTutor to badawczy system AI Teaching Assistant przeznaczony do nauki matematyki od klas
4 SP po maturalne. Łączy trzy warstwy pomocy:
Mini‑lekcja przed zadaniem (tryb pre‑brief w wersji hybrydowej).
Spersonalizowane zadania i feedback generowane przez fine‑tuned Llama‑3 (SFT → RLHF).
Kanał czatu z nauczycielem (Human‑in‑the‑loop) – uczeń może jednym kliknięciem eskalować
rozmowę; nauczyciel widzi kontekst tablicy i odpowiada w tym samym interfejsie.
Key Features
End‑to‑end pipeline: ETL → SFT → Reward Model → RLHF → API + UI.
Matematyczne zbiory danych: GSM8K, MATH, MathQA‑PC oraz syntetyczne zadania
krok‑po‑kroku.
Whiteboard mode: AI rysuje równania / wykresy (SVG, Plotly) i odczytuje pismo ucznia
(Pix2LaTeX / TrOCR).
Teacher Chat: WebSocket kanał „uczeń ↔ nauczyciel” z pełnym kontekstem historyjek i tablicy.
Algorytmy RL: PPO oraz Bandit‑DPO (konfigurowalne YAML‑em).
FastAPI back‑end (vLLM / ggml) + Next.js front‑end (z LaTeX editor + whiteboard).
Monitoring: Superset / Grafana + Weights & Biases.
Project Structure
.
├── data/ # raw & processed math datasets
├── notebooks/ # EDA & reports
├── src/
•
•
•
•
•
•
•
•
•
•
•
1.
2.
3.
•
•
•
•
•
•
•
1
│ ├── etl/ # ingest & cleaning scripts
│ ├── training/ # SFT & RLHF loops
│ ├── reward/ # reward model training / eval
│ ├── api/ # FastAPI server
│ ├── ui/ # Next.js front‑end (LaTeX editor + whiteboard)
│ ├── teacher_chat/ # WebSocket gateway & teacher dashboard
│ └── ocr/ # Pix2LaTeX / TrOCR service
├── configs/ # YAML configs for experiments
├── dashboard/ # Superset dashboard definitions
├── tests/ # unit & integration tests
└── README.md
Quick Start
1. Clone & install
git clone https://github.com/<your-org>/rlhf-mathtutor.git
cd rlhf-mathtutor
conda env create -f environment.yml
conda activate rlhf-mathtutor
pre-commit install
2. Download datasets
python scripts/get_datasets.py --names gsm8k mathqa_pc math
3. Train baseline SFT
python src/training/train_sft.py --config configs/sft_llama3.yml
4. Train reward model
python src/reward/train_reward.py --config configs/reward.yml
5. RLHF
python src/training/train_rlhf.py --config configs/ppo.yml
Training & Evaluation
Metryki: final‑answer Accuracy, Step Accuracy, BLEU‑Explain, Avg .
Reproducibility: seeds & hyper‑params in configs/ .
Tracking: Weights & Biases + MLflow artifacts.
•
•
•
2
Web UI
# API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
# Front‑end
yarn --cwd src/ui install
yarn --cwd src/ui dev
# Teacher dashboard
yarn --cwd src/teacher_chat install
yarn --cwd src/teacher_chat dev
Monitoring
Superset dashboards live in dashboard/ ; import JSON definitions.
Roadmap
-
Contributing
Fork & create feature branch.
pre-commit run --all-files before PR.
Describe motivation, link to an open issue.
License
MIT License – see LICENSE .
Citation
@misc{rlhf_mathtutor_2025,
title = {RLHF Math Tutor},
author = {Your Name},
year = {2025},
howpublished = {\url{https://github.com/<your-org>/rlhf-mathtutor}
