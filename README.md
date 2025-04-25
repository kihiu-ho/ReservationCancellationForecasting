# Reservation Cancellation Forecasting

A machine learning project for predicting hotel reservation cancellations using **Apache Airflow, MLflow, PostgreSQL, MinIO, and Docker**.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Technical Stack](#technical-stack)
- [Machine Learning Model Theory](#machine-learning-model-theory)
- [Airflow Implementation](#airflow-implementation)
- [Model Serving & Deployment](#model-serving--deployment)
- [MLflow Integration](#mlflow-integration)
- [Data Pipeline](#data-pipeline)
- [CI/CD Pipeline](#cicd-pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Environment Variables](#environment-variables)
- [Project Structure](#project-structure)

---

## 📖 Project Overview

This project aims to **predict hotel reservation cancellations** using machine learning models. It is designed to be scalable and easy to deploy using **Docker and Apache Airflow**. The project includes:

- **Data ingestion & preprocessing**
- **Model training & evaluation**
- **Tracking & logging using MLflow**
- **Deployment using Apache Airflow & MinIO**

## 🏗 System Architecture

### High-level System Architecture

```mermaid
graph TB
    subgraph Data Sources
        DB[(PostgreSQL)] 
        API[External APIs]
        Stream[Streaming Data]
    end

    subgraph Data Pipeline
        Airflow[Apache Airflow]
        Spark[Apache Spark]
        Kafka[Apache Kafka]
    end

    subgraph ML Pipeline
        Preprocessing[Data Preprocessing]
        FeatureEng[Feature Engineering]
        Training[Model Training]
        Eval[Model Evaluation]
    end

    subgraph Model Management
        MLflow[MLflow Tracking]
        Registry[Model Registry]
        Serving[Model Serving]
    end

    subgraph Storage
        MinIO[(MinIO Storage)]
        DB2[(PostgreSQL)]
    end

    subgraph CI/CD
        GitHub[GitHub Actions]
        Docker[Docker Registry]
        Deploy[Deployment]
    end

    DB --> Airflow
    API --> Airflow
    Stream --> Kafka --> Airflow
    Airflow --> Spark
    Spark --> Preprocessing
    Preprocessing --> FeatureEng
    FeatureEng --> Training
    Training --> Eval
    Eval --> MLflow
    MLflow --> Registry
    Registry --> Serving
    Serving --> MinIO
    Serving --> DB2
    GitHub --> Docker
    Docker --> Deploy
```

### Detailed Component Architecture

```mermaid
graph LR
    subgraph Data Ingestion Layer
        A1[Batch Data] --> B1[Airflow DAGs]
        A2[Streaming Data] --> B2[Kafka Topics]
        B2 --> B1
    end

    subgraph Processing Layer
        B1 --> C1[Data Validation]
        C1 --> C2[Data Cleaning]
        C2 --> C3[Feature Engineering]
    end

    subgraph ML Layer
        C3 --> D1[Model Training]
        D1 --> D2[Model Evaluation]
        D2 --> D3[Model Registry]
    end

    subgraph Serving Layer
        D3 --> E1[Model Serving API]
        E1 --> E2[Prediction Service]
        E2 --> E3[Monitoring]
    end

    subgraph Storage Layer
        F1[(MinIO)] --> F2[(PostgreSQL)]
        F2 --> F3[(MLflow Artifacts)]
    end
```

## 🔧 Technical Stack

### Core Technologies
- **Python 3.9+**: Primary programming language
- **Apache Airflow 2.7+**: Workflow orchestration
- **MLflow 2.8+**: Model tracking and management
- **PostgreSQL 14+**: Relational database
- **MinIO**: Object storage
- **Docker & Docker Compose**: Containerization

### ML & Data Processing
- **Scikit-learn**: Machine learning algorithms
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Apache Spark**: Distributed computing
- **Apache Kafka**: Stream processing

### Monitoring & Logging
- **Prometheus**: Metrics collection
- **Grafana**: Visualization
- **ELK Stack**: Log management

### CI/CD Tools
- **GitHub Actions**: CI/CD pipeline
- **Docker Registry**: Container registry
- **Kubernetes**: Container orchestration

## 🤖 Machine Learning Model Theory

### Model Architecture

```mermaid
graph TD
    subgraph Feature Engineering
        A[Raw Features] --> B[Feature Selection]
        B --> C[Feature Transformation]
        C --> D[Feature Scaling]
    end

    subgraph Model Training
        D --> E[Train/Test Split]
        E --> F[Model Training]
        F --> G[Cross Validation]
        G --> H[Hyperparameter Tuning]
    end

    subgraph Model Evaluation
        H --> I[Performance Metrics]
        I --> J[Model Comparison]
        J --> K[Model Selection]
    end
```

### Key Algorithms
1. **Random Forest Classifier**
   - Ensemble learning method
   - Handles non-linear relationships
   - Robust to overfitting

2. **XGBoost**
   - Gradient boosting framework
   - High performance on structured data
   - Built-in feature importance

3. **LightGBM**
   - Light Gradient Boosting Machine
   - Faster training speed
   - Lower memory usage

### Model Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

## 🎯 Airflow Implementation

### DAG Structure

```mermaid
graph TD
    subgraph Data Pipeline DAG
        A[Start] --> B[Data Validation]
        B --> C[Data Cleaning]
        C --> D[Feature Engineering]
        D --> E[Model Training]
        E --> F[Model Evaluation]
        F --> G[Model Registration]
    end

    subgraph Monitoring DAG
        H[Start] --> I[Data Quality Check]
        I --> J[Model Performance]
        J --> K[Alert Generation]
    end
```

### Task Dependencies
1. **Data Ingestion Tasks**
   - Validate input data
   - Check data quality
   - Transform data format

2. **Processing Tasks**
   - Clean data
   - Engineer features
   - Prepare training data

3. **ML Tasks**
   - Train models
   - Evaluate performance
   - Register best model

## 🚀 Model Serving & Deployment

### Deployment Architecture

```mermaid
graph TB
    subgraph Model Serving
        A[API Gateway] --> B[Load Balancer]
        B --> C1[Model Server 1]
        B --> C2[Model Server 2]
        C1 --> D[Model Cache]
        C2 --> D
        D --> E[Database]
    end

    subgraph Monitoring
        F[Prometheus] --> G[Grafana]
        G --> H[Alert Manager]
    end
```

### Deployment Strategies
1. **Blue-Green Deployment**
   - Zero-downtime updates
   - Easy rollback
   - Traffic switching

2. **Canary Deployment**
   - Gradual rollout
   - Risk mitigation
   - Performance monitoring

## 📊 MLflow Integration

### MLflow Components

```mermaid
graph LR
    subgraph MLflow Architecture
        A[MLflow Tracking] --> B[MLflow Models]
        B --> C[Model Registry]
        C --> D[Model Serving]
        
        E[Experiments] --> A
        F[Metrics] --> A
        G[Parameters] --> A
    end
```

### Key Features
1. **Experiment Tracking**
   - Parameter logging
   - Metric tracking
   - Artifact storage

2. **Model Registry**
   - Version control
   - Stage transitions
   - Model lineage

## 📈 Data Pipeline

### Pipeline Architecture

```mermaid
graph LR
    subgraph Data Flow
        A[Data Sources] --> B[Ingestion]
        B --> C[Processing]
        C --> D[Storage]
        D --> E[Analysis]
        E --> F[Visualization]
    end

    subgraph Quality Control
        G[Data Validation] --> H[Quality Checks]
        H --> I[Error Handling]
    end
```

### Pipeline Components
1. **Data Ingestion**
   - Batch processing
   - Stream processing
   - API integration

2. **Data Processing**
   - Cleaning
   - Transformation
   - Feature engineering

3. **Data Storage**
   - Raw data
   - Processed data
   - Model artifacts

## 🔄 CI/CD Pipeline

### Pipeline Architecture

```mermaid
graph LR
    subgraph CI/CD Flow
        A[Code Push] --> B[Build]
        B --> C[Test]
        C --> D[Package]
        D --> E[Deploy]
        
        F[Quality Gates] --> B
        G[Security Scan] --> C
        H[Performance Test] --> D
    end
```

### Pipeline Stages
1. **Continuous Integration**
   - Code review
   - Unit testing
   - Integration testing

2. **Continuous Deployment**
   - Container building
   - Image pushing
   - Deployment automation

## 🛠 Installation
Clone the repository:
``` bash
git clone https://github.com/kihiu-ho/ReservationCancellationForecasting.git
cd reservation-cancellation-forecasting
```
Build and start the Docker environment:
``` bash
docker-compose build
docker-compose up -d
```
## ▶️ Usage & Examples
After successfully installing and starting the project:
- **Airflow Web Interface** is running at:
``` 
http://localhost:8080
```
- **MLflow Web Interface** is running at:
``` 
http://localhost:5000
```
- Trigger workflow directly from Airflow UI or CLI:
``` bash
docker-compose exec airflow-scheduler airflow dags trigger your_dag_id
```
## 📑 Environment Variables
Customize the project behavior via environment variables in the `.env` file:
- `AIRFLOW_IMAGE_NAME` (Docker image for Airflow)
- `_AIRFLOW_WWW_USER_USERNAME` & `_AIRFLOW_WWW_USER_PASSWORD` (Airflow Admin login)
- `_PIP_ADDITIONAL_REQUIREMENTS` (Any additional Python packages)

For full details, check the `.env.example` file provided in the source repository.
## 📂 Project Structure
``` 
.
├── airflow                 # Airflow-related code (DAGs, plugins, requirements)
│   └── dags                   
│   └── plugins              
├── data                    # Raw & processed data
├── notebooks               # Jupyter notebooks for exploratory analysis and modeling
├── scripts                 # Data/Model-related Python scripts
├── mlflow                  # Configuration and tracking files related to MLflow
├── .env                    # Environment variables file
├── docker-compose.yaml     # Docker Compose specifications
└── README.md
```
## 🌀 Airflow UI
- URL: [http://localhost:8080](http://localhost:8080)
- Default username & password:
``` 
  username: airflow
  password: airflow
```
## 🔍 MLflow UI
Track your experiments and models:
- MLflow is accessible at:
``` 
http://localhost:5500
```
## 📂 Project Structure
Clear and organized project structure for efficient collaboration:
``` 
reservation-cancellation-forecasting/
├── airflow/
│   ├── dags/
│   └── plugins/
├── data/
├── notebooks/
├── mlflow
├── scripts
├── docker-compose.yml              
├── Dockerfile             
├── requirements.txt        
├── README.md                
├── .env                     
```
## ⚠️ Important Notes
- This configuration is intended only for local development and testing.
- Do NOT use it directly in production without security hardening and proper resource scaling.
- For advanced use during development or production deployments, build customized Docker images that include all required dependencies.

---

## Airflow UI
![img.png](png/img.png)
![img_1.png](png/img_1.png)
## MLflow UI
![img.png](img5.png)