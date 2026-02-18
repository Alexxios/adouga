FROM python:3.14-slim

RUN apt-get update && apt-get install -y curl

RUN pip install poetry

WORKDIR /project
COPY common /project/common/

WORKDIR /project/app
COPY backend/pyproject.toml backend/poetry.lock /project/app/
RUN poetry config virtualenvs.create false
RUN poetry install -v --no-interaction --no-ansi

COPY backend /project/app/
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
