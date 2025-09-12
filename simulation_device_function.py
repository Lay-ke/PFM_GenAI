import json
import random
import time
import math
from datetime import datetime
import boto3
from botocore.exceptions import ClientError, BotoCoreError
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_iot_client():
    """Initialize AWS IoT Core client with error handling"""
    try:
        return boto3.client('iot-data')
    except (BotoCoreError, ClientError) as e:
        logger.error(f"Failed to initialize IoT Core client: {str(e)}")
        raise Exception(f"IoT client initialization failed: {str(e)}")

# Initialize IoT Core client
try:
    iot_client = initialize_iot_client()
except Exception as e:
    logger.error(f"Critical error initializing IoT client: {str(e)}")
    raise

def lambda_handler(event, context):
    """
    Lambda function to generate synthetic sensor data for predictive maintenance
    Matches the format: temperature, pressure, vibration, humidity, equipment, location, faulty
    """
    try:
        # Validate event input
        if not isinstance(event, dict):
            raise ValueError("Event must be a dictionary")

        # Get current timestamp
        try:
            timestamp = datetime.utcnow().isoformat()
        except Exception as e:
            logger.error(f"Failed to generate timestamp: {str(e)}")
            raise ValueError(f"Timestamp generation failed: {str(e)}")

        # Generate sensor readings
        try:
            sensor_data = generate_realistic_sensor_data()
        except Exception as e:
            logger.error(f"Failed to generate sensor data: {str(e)}")
            raise RuntimeError(f"Sensor data generation failed: {str(e)}")

        # Create complete sensor payload
        try:
            payload = {
                'DeviceId': f"conveyor_motor_001",
                'Timestamp': timestamp,
                'Speed (rpm)': round(sensor_data['speed'], 1),
                'Load (kg)': round(sensor_data['load'], 1),
                'Current (A)': round(sensor_data['current'], 2),
                'Temperature (℃)': round(sensor_data['temperature'], 2),
                'Vibration (m/s²)': round(sensor_data['vibration'], 2),
                'Pressure (Pa)': round(sensor_data['pressure'], 2),
                'Humidity (g/m³)': round(sensor_data['humidity'], 2),
                'Equipment': sensor_data['equipment'],
                'Location': sensor_data['location'],
                'Faulty': sensor_data['faulty']
            }
        except KeyError as e:
            logger.error(f"Missing required sensor data key: {str(e)}")
            raise ValueError(f"Invalid sensor data format: {str(e)}")

        # Validate payload before publishing
        if not all(key in payload for key in ['DeviceId', 'Timestamp', 'Speed (rpm)', 'Load (kg)', 'Current (A)', 
                                            'Temperature (℃)', 'Vibration (m/s²)', 'Pressure (Pa)', 'Humidity (g/m³)', 
                                            'Equipment', 'Location', 'Faulty']):
            raise ValueError("Incomplete payload data")

        # Send to IoT Core
        try:
            response = iot_client.publish(
                topic='predictive-maintenance/sensor-data-2',
                qos=1,
                payload=json.dumps(payload)
            )
            logger.info(f"Data sent successfully: {payload}")
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Sensor data published successfully',
                    'payload': payload
                })
            }
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Error publishing to IoT Core: {str(e)}")
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'error': f"Failed to publish to IoT Core: {str(e)}"
                })
            }
        except json.JSONEncodeError as e:
            logger.error(f"JSON serialization error: {str(e)}")
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'error': f"Payload serialization failed: {str(e)}"
                })
            }

    except Exception as e:
        logger.error(f"Unexpected error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': f"Internal server error: {str(e)}"
            })
        }

def generate_realistic_sensor_data():
    """
    Generate conveyor motor sensor data matching the patterns observed
    in the dataset (Speed, Load, Temp, Vibration, Current + env sensors).
    """
    try:
        # Equipment and location (metadata)
        equipment_types = ['ConveyorMotor', 'PumpMotor', 'FanMotor']
        locations = ['Atlanta', 'Chicago', 'San Francisco', 'New York', 'Houston']
        
        if not equipment_types or not locations:
            raise ValueError("Equipment types or locations list is empty")

        equipment = random.choice(equipment_types)
        location = random.choice(locations)

        # Generate sensor readings
        speed = random.gauss(120, 3)          # Mean ~120 rpm, range ~115–125
        load = random.gauss(495, 10)          # Mean ~495 kg, range ~480–510
        current = 2.9 + (load - 480) * 0.01   # Correlate current with load (~3.0–3.4 A)
        current += random.uniform(-0.05, 0.05)

        temperature = random.gauss(41, 3)     # From dataset, ~37–43℃
        vibration = 0.65 + (speed - 115) * 0.01  # Base ~0.65–0.85, slight correlation
        vibration += random.uniform(-0.05, 0.05)

        pressure = random.gauss(35, 5)        # Simulated
        humidity = random.gauss(50, 10)

        # Clamp values within realistic dataset bounds
        try:
            speed = max(110, min(130, speed))
            load = max(470, min(520, load))
            current = max(2.8, min(3.5, current))
            temperature = max(35, min(45, temperature))
            vibration = max(0.6, min(0.9, vibration))
            pressure = max(20, min(80, pressure))
            humidity = max(20, min(90, humidity))
        except TypeError as e:
            logger.error(f"Error clamping sensor values: {str(e)}")
            raise ValueError(f"Invalid sensor value type: {str(e)}")

        # Fault injection
        faulty = 1 if random.random() < 0.05 else 0
        if faulty:
            try:
                load *= 1.1
                current *= 1.15
                vibration *= 1.3
                temperature += 5
            except TypeError as e:
                logger.error(f"Error applying fault injection: {str(e)}")
                raise ValueError(f"Fault injection calculation error: {str(e)}")

        return {
            'speed': speed,
            'load': load,
            'current': current,
            'temperature': temperature,
            'vibration': vibration,
            'pressure': pressure,
            'humidity': humidity,
            'equipment': equipment,
            'location': location,
            'faulty': faulty
        }

    except Exception as e:
        logger.error(f"Error in generate_realistic_sensor_data: {str(e)}")
        raise

def generate_batch_data(num_records=50):
    """
    Generate multiple sensor readings for testing/seeding
    Returns data in the same format as the sample CSV
    """
    try:
        if not isinstance(num_records, int) or num_records <= 0:
            raise ValueError("num_records must be a positive integer")

        batch_data = []
        for _ in range(num_records):
            try:
                data = generate_realistic_sensor_data()
                
                record = {
                    'temperature': round(data['temperature'], 13),
                    'pressure': round(data['pressure'], 13),
                    'vibration': round(data['vibration'], 15),
                    'humidity': round(data['humidity'], 13),
                    'speed': round(data['speed'], 1),
                    'load': round(data['load'], 1),
                    'current': round(data['current'], 2),
                    # Note: motor_temperature and motor_vibration were not in original data
                    # Using temperature and vibration as substitutes
                    'motor_temperature': round(data['temperature'], 1),
                    'motor_vibration': round(data['vibration'], 2),
                    'equipment': data['equipment'],
                    'location': data['location'],
                    'faulty': data['faulty']
                }
                batch_data.append(record)
            except Exception as e:
                logger.error(f"Error generating record: {str(e)}")
                continue  # Skip failed record but continue with others
        
        if not batch_data:
            raise RuntimeError("Failed to generate any valid batch data")
        
        return batch_data

    except Exception as e:
        logger.error(f"Error in generate_batch_data: {str(e)}")
        raise

def simulate_degradation_over_time():
    """
    Advanced function to simulate equipment degradation patterns over time
    This creates more realistic long-term trends
    """
    try:
        current_time = time.time()
        
        cycles = {
            'Turbine': 45 * 24 * 3600,
            'Compressor': 30 * 24 * 3600,
            'Pump': 60 * 24 * 3600
        }
        
        if not cycles:
            raise ValueError("Cycles configuration is empty")
            
        equipment = random.choice(list(cycles.keys()))
        cycle_time = current_time % cycles[equipment]
        degradation_progress = cycle_time / cycles[equipment]
        
        base_fault_prob = {'Turbine': 0.05, 'Compressor': 0.03, 'Pump': 0.08}
        if equipment not in base_fault_prob:
            raise KeyError(f"No fault probability defined for {equipment}")
            
        current_fault_prob = base_fault_prob[equipment] * (1 + degradation_progress * 3)
        return min(current_fault_prob, 0.25)

    except Exception as e:
        logger.error(f"Error in simulate_degradation_over_time: {str(e)}")
        raise

def generate_csv_output(num_records=65):
    """
    Generate data in CSV format matching the sample structure
    """
    try:
        batch_data = generate_batch_data(num_records)
        
        csv_output = "temperature\tpressure\tvibration\thumidity\tequipment\tlocation\tfaulty\n"
        
        for record in batch_data:
            try:
                row = f"{record['temperature']}\t{record['pressure']}\t{record['vibration']}\t{record['humidity']}\t{record['equipment']}\t{record['location']}\t{record['faulty']}\n"
                csv_output += row
            except KeyError as e:
                logger.error(f"Missing key in CSV record: {str(e)}")
                continue
                
        return csv_output

    except Exception as e:
        logger.error(f"Error in generate_csv_output: {str(e)}")
        raise

def validate_trigger_config(config_name):
    """Validate trigger configuration"""
    try:
        if config_name not in TRIGGER_CONFIGS:
            raise KeyError(f"Invalid trigger config: {config_name}")
        return TRIGGER_CONFIGS[config_name]
    except Exception as e:
        logger.error(f"Error validating trigger config: {str(e)}")
        raise

# Configuration for different trigger rates
TRIGGER_CONFIGS = {
    "high_frequency": "rate(30 seconds)",
    "normal": "rate(2 minutes)",
    "low_frequency": "rate(5 minutes)"
}