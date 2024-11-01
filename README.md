# S1.3

## Setup

1. Clone the repository:
```
git clone https://github.com/142CodeGreen/RAG-with-NeMo-Guardrails.git
cd RAG-with-NeMo-Guardrails
```

2. Install the required packages:
```
pip install --upgrade -r requirements.txt
```

3. Export API keys. NVIDIA_API_KEY is for NVIDIA NIM, while OpenAI API Key is needed for Nemo Guardrails. 
```
export NVIDIA_API_KEY="your-api-key-here"
echo $NVIDIA_API_KEY

export OPENAI_API_KEY="your-openai-key-here"
echo $OPENAI_API_KEY
```

4. Run the app.py:
```
python3 app.py
```
