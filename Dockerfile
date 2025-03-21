FROM python:3.13

RUN useradd -m -u 1000 user

WORKDIR /code

RUN chown -R user:user /code

USER user

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt && \
    export PATH=$PATH:/home/user/.local/bin

COPY . /code

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
