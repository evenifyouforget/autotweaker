import json
from get_design import retrieveDesign, designDomToStruct
from .measure_design import measure_design

def program_local(args):
    if args.do_generate_config:
        print('Generate config pass not currently supported')
        exit(1)

    if args.do_autotweak:
        # Load config from JSON file
        with open(args.config_file) as f:
            config_data = json.load(f)
        # TODO validate JSON data against schema
        # Fetch design data
        design_struct = designDomToStruct(retrieveDesign(args.design_id))
        # debug placeholder run
        run_result = measure_design(design_struct=design_struct, config_data=config_data)
        print(run_result)