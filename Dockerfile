FROM python:3.13-slim

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user . .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu .

USER user
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
