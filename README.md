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
   git clone https://github.com/your-username/reservation-cancellation-forecasting.git
   cd reservation-cancellation-forecasting