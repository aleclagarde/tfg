FROM python:3.9
ENV PYTHONUNBUFFERED 1 
EXPOSE 8000 
WORKDIR /app
COPY . /app
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "app:app"]