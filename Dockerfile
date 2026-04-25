FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install fastapi uvicorn pydantic python-dotenv openai gradio

EXPOSE 8000

CMD ["uvicorn", "auto_sre_env.server:app", "--host", "0.0.0.0", "--port", "7860"]