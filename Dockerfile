FROM python:3.11.9

WORKDIR /app

ADD . /app

RUN pip install poetry

RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi

CMD ["/bin/bash"]
