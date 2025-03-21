FROM python:3.13


RUN useradd -m -u 1000 user
USER user

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN chown -R user:user /code

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# COPY . .

COPY --chown=user:user . /code

# Start the FastAPI app on port 7860, the default port expected by Spaces
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
