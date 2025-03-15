FROM apache/airflow:2.10.5
#Create aodels directory and set ownership
USER root
RUN apt-get update && apt-get install -y git && apt-get install -y vim && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /app/models
USER airflow
#Copy requirenenta file
COPY requirements.txt /tmp/requirements.txt
# Install Python packages as root (systan-aide)
RUN pip install -r /tmp/requirements.txt
#Switch back to airflow user for safety
USER airflow