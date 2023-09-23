#  Base image - python 3.9
FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# add libgomp1 for lightgbm to work
# check https://stackoverflow.com/questions/55036740/lightgbm-inside-docker-libgomp-so-1-cannot-open-shared-object-file
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app files
COPY . /app

# Expose the port
EXPOSE 8501

# Run the app when the container launches
CMD ["streamlit", "run", "app.py", "--server.port", "8501"]