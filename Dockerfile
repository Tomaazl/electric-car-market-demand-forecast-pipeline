FROM python:3.11-slim

RUN apt-get update && apt-get install -y gcc libpq-dev &&     pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
COPY . .

CMD ["bash"]