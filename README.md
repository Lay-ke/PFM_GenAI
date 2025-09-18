# Predictive Maintenance Forecaster (PFM_GenAI)

## Purpose

This repository contains a serverless prototype for predictive maintenance, designed to run on AWS Lambda. The flow includes simulated device sensor data, an inference Lambda that loads a PyTorch model and scaler, and a Bedrock agent handler for higher-level prompts and responses.

---

## Architecture

Here is the full architecture diagram.
![PFM_GenAI Architecture Diagram](./PMF%20-%20architecture.jpg)

**Core runtime responsibilities:**
- **Sensor simulation:** Generates periodic sensor payloads.
- **IoT / ingestion:** (Assumed external) Accepts sensor stream and forwards to Lambda.
- **Inference Lambda (`conveyor_inference_function.py`):** Loads model and scaler, runs inference, writes results to storage.
- **Agent handler (`bedrock_agent_query_handler.py`):** Interacts with a Bedrock-like agent for queries and explanations.

---

## Repository Layout (Lambda-first)

- `conveyor_inference_function.py` — Lambda handler for processing incoming sensor payloads and returning inference results.
- `simulation_device_function.py` — Lambda-style or local script for generating simulated sensor data on a schedule.
- `bedrock_agent_query_handler.py` — Lambda handler that communicates with a Bedrock agent (or similar LLM service).
- `ml_model/` — Model artifacts and helper utilities:
    - `best_model.pth` — Serialized PyTorch model.
    - `inference.py` — Functions to load the model, scaler, and run inference.
    - `scaler.pkl` — Scikit-learn scaler for features.
    - `classes.txt` — Label/class mapping.
    - `features.txt` — Expected feature names and order.
    - `metrics.json` — Training/evaluation metrics.
    - `train.py`, `pipeline.ipynb` — Training artifacts (not required for runtime Lambda).

---

## Lambda Deployment Notes

- **Runtime environment:** Python 3.9+.
- **Dependency packaging:** PyTorch is not included in Lambda by default. Two options:
    1. **Container image:** Build a Docker image with required Python packages (recommended).
    2. **Lambda layer / zipped package:** Bundle dependencies into a Lambda layer (challenging for large native libs like PyTorch).

---

## Workflow (End-to-End)

Each step maps a cloud component to repository files and handlers:

1. **Data collection (Sensor / Data lake)**
     - Raw sensor data and historical datasets stored centrally (e.g., S3).
     - Repo: Training artifacts in `ml_model/` (`train.py`, `pipeline.ipynb`, `metrics.json`).

2. **Model training**
     - Offline model training produces a model artifact for inference.
     - Repo: `train.py`, `pipeline.ipynb`, `best_model.pth`.

3. **Sensor simulation (scheduled)**
     - Scheduled job triggers function to simulate telemetry and publish to ingestion endpoint.
     - Repo: `simulation_device_function.py`.

4. **Ingestion (IoT / message filter)**
     - AWS IoT or API Gateway ingests streaming sensor data and routes to inference Lambda.
     - Repo: Ingestion is external; `conveyor_inference_function.py` consumes messages.

5. **Routing / rules**
     - IoT Rules or message routing forwards relevant messages to inference Lambda.
     - Repo: Configuration not included.

6. **Inference Lambda**
     - Receives payload, extracts features, applies scaler, runs model, persists results.
     - Repo: `conveyor_inference_function.py`, `ml_model/inference.py`, `ml_model/best_model.pth`, `ml_model/scaler.pkl`.

7. **Inference storage**
     - Results written to storage for analysis/visualization (e.g., S3, DynamoDB).
     - Repo: Storage logic in `conveyor_inference_function.py`.

8. **Data aggregation and visualization**
     - Aggregated data used for dashboarding (e.g., QuickSight, Grafana).
     - Repo: Not included; use `ml_model/` outputs and `metrics.json`.

9. **Model retraining trigger**
     - Retrain model periodically or on data drift.
     - Repo: Retraining orchestration not included; `train.py` provides logic.

10. **Bedrock agent (reasoning and user prompts)**
        - Agent consumes inference history and answers user queries.
        - Repo: `bedrock_agent_query_handler.py`.

11. **User interaction / UI**
        - Front-end or API exposes query and visualization capabilities.
        - Repo: UI out-of-scope; agent handler ready for API Gateway or front-end.

---

## Implementation Notes

- **Deployment infrastructure:** EventBridge schedules, IoT rules, Lambda roles, S3 buckets required for full deployment. SAM or Terraform templates can be added.
- **Observability:** Add CloudWatch metrics/logs for inference Lambda; consider a dead-letter queue (DLQ).
- **Security:** Use least privilege IAM roles for Lambdas. For ECR containers, ensure proper repository policy and lifecycle rules.

---

## Notes

- This repo does not include IaC templates.