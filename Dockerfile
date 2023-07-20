# Stage 1: Build the binary in uniapps baseimage
FROM 164775872696.dkr.ecr.ap-south-1.amazonaws.com/uniapps-base:2020.02.02 as build_env

# Use Python v3.10 image for compatibility
FROM python:3.10 as builder

# Copy the model and classification script
COPY ./classify.py /app/classify.py
COPY ./tf_files /app/tf_files

# Install tensorflow and pyinstaller dependencies
RUN python3.10 -m pip install tensorflow pyinstaller

# Run binary creation script
RUN pyinstaller --add-data "/app/tf_files:tf_files" --onefile /app/classify.py --distpath /app/dist

ENTRYPOINT [ "/app/dist/classify" ]