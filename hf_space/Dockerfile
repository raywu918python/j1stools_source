FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY ai_chat.py .
COPY ai_support.py .
COPY llm_provider.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
