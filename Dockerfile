FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "auto_sre_env.server:app", "--host", "0.0.0.0", "--port", "7860"]