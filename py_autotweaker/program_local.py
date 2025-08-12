import itertools
import json
from jsonschema import validate
from pathlib import Path
import time
from get_design import retrieveDesign, designDomToStruct
from save_design import save_design
from .auto_login import auto_login_get_user_id
from .job_struct import Creature, Garden
from .performance import get_thread_count

def program_local(args):
    if args.do_generate_config:
        print('Generate config pass not currently supported')
        exit(1)

    if args.do_autotweak:
        # Start the clock
        start_time = time.time()

        # Get user ID, needed for saving
        user_id = auto_login_get_user_id()

        # Configure threads
        max_threads = get_thread_count(args.max_threads)

        # Other config
        timeout_seconds = args.timeout_seconds
        stop_on_win = args.stop_on_win
        upload_best_k = args.upload_best_k
        max_garden_size = args.max_garden_size
        if not max_garden_size:
            max_garden_size = max(upload_best_k, max_threads) * 3

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

        # Set up garden
        garden = Garden([Creature(design_struct)], max_garden_size, config_data)
        
        # Loop until end condition reached
        for loop_count in itertools.count():
            time_elapsed = time.time() - start_time
            print(f'# Loop #{loop_count + 1} begins, total time {time_elapsed:.2f} seconds since start')
            if time_elapsed > timeout_seconds:
                print(f'Stopping due to time limit reached ({timeout_seconds} seconds)')
                break
            garden_status = garden.checkup()
            print(f'Best score so far: {garden_status.best_score}')
            if stop_on_win and garden_status.best_score <= 0:
                print('Stopping early since a solve was found')
                break
            garden.evolve(max_threads, upload_best_k)
            garden.start(max_threads - garden_status.num_active_threads)
            print(f'Statistics: {garden.num_kills} creatures have been culled so far')

        # Save results
        print(f'Uploading up to {upload_best_k} best creatures')
        for rank, creature in enumerate(garden.creatures[:upload_best_k]):
            saved_design_id = save_design(creature.design_struct, user_id)
            print(f'{rank+1}. Saved to https://ft.jtai.dev/?designId={saved_design_id} achieving score of {creature.best_score}')