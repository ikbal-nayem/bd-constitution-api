FROM python:3.13


RUN useradd -m -u 1000 user
USER user

WORKDIR /

COPY requirements.txt /

RUN chown -R user:user /

# Install requirements.txt 
RUN pip install --no-cache-dir --upgrade -r /requirements.txt

# COPY . .

COPY --chown=user:user . .

# Start the FastAPI app on port 7860, the default port expected by Spaces
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
