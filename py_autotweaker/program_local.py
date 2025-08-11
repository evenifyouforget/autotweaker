import json
from jsonschema import validate
from pathlib import Path
from get_design import retrieveDesign, designDomToStruct
from save_design import fc_login, save_design
from .measure_design import measure_design

def program_local(args):
    if args.do_generate_config:
        print('Generate config pass not currently supported')
        exit(1)

    if args.do_autotweak:
        # Load config from JSON file
        with open(args.config_file) as f:
            config_data = json.load(f)

        # Validate JSON data against schema
        schema_path = Path(__file__).parent.parent / 'schema' / 'job-config.schema.json'
        with open(schema_path) as f:
            schema = json.load(f)
        validate(instance=config_data, schema=schema)

        # Fetch design data
        design_struct = designDomToStruct(retrieveDesign(args.design_id))
        # debug placeholder run
        run_result = measure_design(design_struct=design_struct, config_data=config_data)
        print(run_result)
        # debug save
        user_id = fc_login(input('Enter username: '), input('Enter password: ')).user_id
        print(f'You user id: {user_id}')
        saved_design_id = save_design(design_struct, user_id)
        print(f'Saved to https://ft.jtai.dev/?designId={saved_design_id}')