import boto3
import json
import os
import logging
import uuid
from datetime import datetime

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# initialize s3 client
s3_client = boto3.client("s3")

# Set your bucket name in env var (recommended)
S3_BUCKET = os.environ.get("S3_BUCKET_NAME", "relu-quicksight")

# Initialize SageMaker runtime client
sagemaker_client = boto3.client('sagemaker-runtime')

ENDPOINT_NAME = os.environ.get('SAGEMAKER_ENDPOINT_NAME', 'pytorch-inference-2025-09-11-14-15-37-612')

# Incoming fields (model expects these exact names)
REQUIRED_FIELDS = [
    "Speed (rpm)",
    "Load (kg)",
    "Temperature (℃)",
    "Vibration (m/s²)",
    "Current (A)"
]

# Mapping for output cleanup
FIELD_MAPPING = {
    "Speed (rpm)": "speed_rpm",
    "Load (kg)": "load_kg",
    "Temperature (℃)": "temperature_c",
    "Vibration (m/s²)": "vibration_ms2",
    "Current (A)": "current_a"
}

# Fault dictionary
FAULT_KB = {
    "ball bearing": {
        "label": "Ball Bearing Fault detected",
        "severity": "high",
        "recommendation": "Shut down immediately and replace bearing"
    },
    "central_shaft_fault": {
        "label": "Central Shaft Fault detected",
        "severity": "high",
        "recommendation": "Inspect shaft alignment and lubrication urgently"
    },
    "pulley_fault": {
        "label": "Pulley Fault detected",
        "severity": "medium",
        "recommendation": "Check pulley alignment and wear within 24 hours"
    },
    "drive motor": {
        "label": "Drive Motor Fault detected",
        "severity": "high",
        "recommendation": "Inspect motor windings and insulation immediately"
    },
    "idler_roller_fault": {
        "label": "Idler Roller Fault detected",
        "severity": "low",
        "recommendation": "Schedule roller inspection during next maintenance"
    },
    "belt slippage": {
        "label": "Potential Belt Slippage - recommend inspection",
        "severity": "medium",
        "recommendation": "Inspect belt tension and alignment within 24 hours"
    }
}

def format_payload_as_text(payload: dict) -> str:
    """
    Convert structured payload into a rich text report.
    """
    return f"""[Device Report]

Device ID: {payload['device_id']}
Timestamp: {payload['timestamp']}

Operating Conditions:
- Speed: {payload['speed_rpm']} rpm
- Load: {payload['load_kg']} kg
- Temperature: {payload['temperature_c']} °C
- Vibration: {payload['vibration_ms2']} m/s²
- Current: {payload['current_a']} A

Model Inference:
- Predicted Fault: {payload['ml_predicted_class']}
- Confidence: {payload['ml_confidence']}

Fault Management Refinement:
- Refined Label: {payload['fm_refined_label']}
- Severity: {payload['fm_severity']}
- Recommendation: {payload['fm_recommendation']}

Summary (natural language):
On {payload['timestamp']}, device {payload['device_id']} was running at {payload['speed_rpm']} rpm with a load of {payload['load_kg']} kilograms. 
The operating temperature was {payload['temperature_c']} °C, vibration measured {payload['vibration_ms2']} m/s², 
and current draw was {payload['current_a']} amps. 
The AI model predicted "{payload['ml_predicted_class']}" with a confidence of {payload['ml_confidence']}. 
This was refined to "{payload['fm_refined_label']}" with a severity level of {payload['fm_severity']}. 
The recommended action is: {payload['fm_recommendation']}.
"""


def lambda_handler(event, context):
    logger.info(f"Received event: {json.dumps(event)}")

    try:
        # ✅ Validate input for model
        for field in REQUIRED_FIELDS:
            if field not in event:
                raise KeyError(f"Missing required field: '{field}'")
            if event[field] is None or not isinstance(event[field], (int, float)):
                raise ValueError(f"Invalid value for '{field}': {event[field]}")

        # ✅ Payload sent to SageMaker (raw field names)
        payload_str = json.dumps({"instances": [{k: event[k] for k in REQUIRED_FIELDS}]})
        logger.info(f"Payload for model: {payload_str}")

        response = sagemaker_client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=payload_str
        )

        result = response['Body'].read().decode('utf-8')
        parsed_result = json.loads(result)

        prediction = parsed_result.get("predictions", [{}])[0]
        # print('Prediction::', prediction)
        predicted_class = prediction.get("predicted_class", "unknown")
        confidence = prediction.get("confidence", 0.0)

        # ✅ Build final clean output (remapped fields)
        clean_fields = {FIELD_MAPPING[k]: event[k] for k in REQUIRED_FIELDS}
        clean_fields["device_id"] = event.get("DeviceId", "unknown_device")
        clean_fields["timestamp"] = event.get("Timestamp", datetime.utcnow().isoformat() + "Z")

        # Normalize key for lookup
        # normalized_class = predicted_class.lower().replace(" ", "_")

        fault_info = FAULT_KB.get(predicted_class, {
            "label": f"Unmapped fault type: {predicted_class}",
            "severity": "unknown",
            "recommendation": "Further analysis required"
        })

        print('Fault Info::', fault_info)

        combined_payload = {
            **clean_fields,
            "ml_predicted_class": predicted_class,
            "ml_confidence": round(confidence, 2),
            "fm_refined_label": fault_info["label"],
            "fm_severity": fault_info["severity"],
            "fm_recommendation": fault_info["recommendation"]
        }

        logger.info(f"Final payload: {json.dumps(combined_payload)}")

        # Parse timestamp from payload (ensure it's in ISO format)
        ts = datetime.fromisoformat(combined_payload["timestamp"].replace("Z", ""))

        # Partitioned path
        partition_path = f"inference_analytics/{ts.year}/{ts.month:02d}/{ts.day:02d}/"

        # Unique file name
        file_name = f"{combined_payload['device_id']}_{ts.strftime('%H%M%S')}_{uuid.uuid4().hex}.json"

        # Full S3 key
        s3_key = f"{partition_path}{file_name}"

        # Unique file name using device_id + timestamp + random id
        file_name = f"{combined_payload['device_id']}_{combined_payload['timestamp']}_{uuid.uuid4().hex}.json"

        # Upload JSON to S3
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key, 
            Body=json.dumps(combined_payload, indent=2),
            ContentType="application/json"
        )

        # ✅ Create TXT representation
        txt_content = format_payload_as_text(combined_payload)

        # TXT partitioned path
        txt_partition_path = f"knowledge/{ts.year}/{ts.month:02d}/{ts.day:02d}/"

        # TXT file name (parallel to JSON)
        txt_file_name = f"{combined_payload['device_id']}_{ts.strftime('%H%M%S')}_{uuid.uuid4().hex}.txt"
        txt_s3_key = f"{txt_partition_path}{txt_file_name}"

        # Upload TXT to S3
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=txt_s3_key,
            Body=txt_content,
            ContentType="text/plain"
        )

        logger.info(f"Stored TXT report in s3://{S3_BUCKET}/{txt_s3_key}")

        logger.info(f"Stored result in s3://{S3_BUCKET}/{s3_key}")
        print(f"Stored result in s3://{S3_BUCKET}/sagemaker_inference/{file_name}")

        return {
            "statusCode": 200,
            "body": json.dumps(combined_payload)
        }

    except Exception as e:
        logger.exception("Unhandled exception occurred.")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Internal server error.", "details": str(e)})
        }
