FROM python:3.11-slim

# deps do sistema (inclui zstd)
RUN apt-get update && apt-get install -y \
    curl ca-certificates bash zstd \
 && rm -rf /var/lib/apt/lists/*

# instala ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# código
COPY . /app

# Railway
ENV PORT=8080
EXPOSE 8080

# modelo default (leve)
ENV OLLAMA_MODEL=llama3.2:3b

# inicia ollama + baixa modelo + inicia api
CMD ["bash", "-lc", "ollama serve & sleep 2 && ollama pull ${OLLAMA_MODEL} && uvicorn main:app --host 0.0.0.0 --port ${PORT}"]