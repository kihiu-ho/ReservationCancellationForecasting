# Reservation Cancellation Forecasting

A machine learning project for predicting hotel reservation cancellations using **Apache Airflow, MLflow, PostgreSQL, MinIO, and Docker**.

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Environment Variables](#environment-variables)
- [Project Structure](#project-structure)
- [Contributors](#contributors)
- [License](#license)

---

## 📖 Project Overview

This project aims to **predict hotel reservation cancellations** using machine learning models. It is designed to be scalable and easy to deploy using **Docker and Apache Airflow**. The project includes:

- **Data ingestion & preprocessing**
- **Model training & evaluation**
- **Tracking & logging using MLflow**
- **Deployment using Apache Airflow & MinIO**

### 🔧 Tech Stack

- **Python** (Machine Learning & Data Processing)
- **Apache Airflow** (Workflow Orchestration)
- **PostgreSQL** (Database)
- **MinIO** (Object Storage)
- **MLflow** (Model Tracking)
- **Docker & Docker Compose** (Containerization)

---

## 🚀 Features

- **Automated Data Pipeline**: Uses Apache Airflow for scheduling and execution.
- **Model Tracking & Experimentation**: MLflow tracks model performance.
- **Scalable Architecture**: Deployable using Docker.
- **Object Storage Support**: MinIO stores model artifacts.

---

## 🛠 Installation

### Prerequisites

Ensure you have the following installed:

- **Docker & Docker Compose**
- **Python 3.8+**
- **Git**

### Setup Instructions

1. **Clone the Repository**
   ```sh
   git clone https://github.com/kihiu-ho/ReservationCancellationForecasting.git
   cd reservation-cancellation-forecasting
   ```

2. **Set Up Environment Variables**
   Create a `.env` file in the root directory with the necessary environment variables:
   ```ini
   AWS_ACCESS_KEY_ID=your-access-key
   AWS_SECRET_ACCESS_KEY=your-secret-key
   MINIO_USERNAME=minio
   MINIO_PASSWORD=minio123
   ```

3. **Start Docker Containers**
   ```sh
   docker-compose up -d
   ```

4. **Verify Running Services**
   - Airflow UI: [http://localhost:8080](http://localhost:8080)
   - MLflow UI: [http://localhost:5500](http://localhost:5500)
   - MinIO UI: [http://localhost:9001](http://localhost:9001)

### Starting and Stopping Docker Services

To start the services with Flower for monitoring:
```sh
docker compose --profile flower up -d --build
```

To stop and remove all containers, networks, and volumes:
```sh
docker compose down -v
```

To stop only the Flower service:
```sh
docker compose --profile flower down
```

---

## 📊 Usage

### Running the Pipeline
To trigger an Airflow DAG for training the model:
1. **Login to the Airflow UI**
2. **Enable the DAG** (e.g., `reservation_cancellation_dag`)
3. **Trigger the DAG manually**



### Accessing MLflow
To track experiments and model performance:
```sh
mlflow ui --port 5500
```

---

## ⚙️ Environment Variables

| Variable | Description |
|----------|------------|
| `AIRFLOW_IMAGE_NAME` | Docker image for Airflow |
| `AIRFLOW_UID` | User ID for Airflow containers |
| `AWS_ACCESS_KEY_ID` | MinIO Access Key |
| `AWS_SECRET_ACCESS_KEY` | MinIO Secret Key |
| `MLFLOW_TRACKING_URI` | MLflow tracking server URL |

---

## 📂 Project Structure

```
reservation-cancellation-forecasting/
│── dags/                      # Airflow DAGs
│── models/                    # Trained models
│── data/                      # Dataset directory
│── config/                    # Configuration files
│── logs/                      # Log files
│── scripts/                   # Helper scripts
│── docker-compose.yaml        # Docker services
│── README.md                  # Project documentation
│── main.py                    # Entry point for ML pipeline
│── requirements.txt            # Python dependencies
```

---