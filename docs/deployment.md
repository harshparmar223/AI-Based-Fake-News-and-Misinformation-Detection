# Deployment and Running Guide

## 1. Local Development Setup

### Prerequisites
- Python 3.10+ (tested up to 3.13)
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/AI-Fake-News-Detection.git
cd AI-Fake-News-Detection

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download required NLTK corpora
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Configure environment variables
cp .env.example .env
```

### Data Preparation & Model Training
```bash
# 1. Clean raw data and generate train/test splits
python -m src.preprocessing.prepare_dataset

# 2. Train and evaluate all models (SVM, RF, Logistic)
python -m src.ml.trainer
```

### Launching Streamlit Web Application
```bash
streamlit run app.py
```
Access the web dashboard in your browser at `http://localhost:8501`.

---

## 2. Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

COPY . .

# Run pre-training if models are not pre-packaged
RUN python -m src.preprocessing.prepare_dataset && python -m src.ml.trainer

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build & Run Container
```bash
docker build -t truthlens-ai .
docker run -p 8501:8501 truthlens-ai
```

---

## 3. Testing & Code Quality

```bash
# Run all unit and integration tests
pytest -v

# Run with test coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```
