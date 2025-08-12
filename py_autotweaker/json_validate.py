import json
from jsonschema import validate
from pathlib import Path

def json_validate(config_data):
    # Validate JSON data against schema
    schema_path = Path(__file__).parent.parent / 'schema' / 'job-config.schema.json'
    with open(schema_path) as f:
        schema = json.load(f)
    validate(instance=config_data, schema=schema)