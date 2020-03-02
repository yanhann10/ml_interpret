FROM python:3.7.3-stretch

# Working Directory
WORKDIR /app

# Copy source code to working directory
COPY . app.py /app/

EXPOSE 8080

# Install packages from requirements.txt
# hadolint ignore=DL3013
RUN pip install --upgrade pip &&\
    pip install --trusted-host pypi.python.org -r requirements.txt


# ENTRYPOINT ['streamlit', 'run']

# CMD ["app.py"]

CMD streamlit run --server.port 8080 --server.enableCORS false app.py