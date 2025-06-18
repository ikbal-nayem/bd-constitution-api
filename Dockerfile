FROM python:3.13

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
		PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/code

# Try and run pip command after setting the user with `USER user` to avoid permission issues with Python
RUN pip install --no-cache-dir --upgrade pip

COPY --chown=user ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY --chown=user . $HOME/code

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
