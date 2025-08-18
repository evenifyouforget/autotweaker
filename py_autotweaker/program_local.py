import itertools
import json
from pathlib import Path
import pickle
import time
import numpy as np
from PIL import Image
from get_design import retrieveDesign, designDomToStruct
from save_design import save_design
from .auto_login import auto_login_get_user_id
from .job_struct import Creature, Garden
from .json_validate import json_validate
from .screenshot import screenshot_design
from .performance import get_thread_count

def program_local(args):
    # Fetch design data
    design_struct = designDomToStruct(retrieveDesign(args.design_id))
        
    # Generate thumbnail if requested
    if args.output_thumbnail_image:
        
        # Generate screenshot
        image_dimensions = (args.thumbnail_width, args.thumbnail_height)
        screenshot = screenshot_design(design_struct, image_dimensions, use_rgb=True)
        
        # Save image
        pil_image = Image.fromarray(screenshot, mode='RGB')
        pil_image.save(args.output_thumbnail_image)
        print(f"Thumbnail saved to: {args.output_thumbnail_image}")

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
        json_validate(config_data)

        # Set up garden
        garden = Garden([Creature(design_struct)], max_garden_size, config_data)
        
        # Set up silenceable printing
        silence_loop_print = False
        def loop_print(*args, **kwargs):
            if silence_loop_print:
                return
            print(*args, **kwargs)
        # Loop until end condition reached
        try:
            for loop_count in itertools.count():
                # Don't print every loop
                silence_loop_print = loop_count > 100 and loop_count % 10**(len(str(loop_count)) - 2)
                # Do the loop
                time_elapsed = time.time() - start_time
                loop_print(f'# Loop #{loop_count + 1} begins, total time {time_elapsed:.2f} seconds since start, using {max_threads} threads and {max_garden_size} creatures')
                if time_elapsed > timeout_seconds:
                    print(f'Stopping due to time limit reached ({timeout_seconds} seconds)')
                    break
                garden_status = garden.checkup()
                loop_print(f'Best score so far: {garden_status.best_score}')
                if stop_on_win and garden_status.best_score <= 0:
                    print('Stopping early since a solve was found')
                    break
                garden.evolve(max_threads, upload_best_k)
                garden.start(max_threads - garden_status.num_active_threads)
                loop_print(f'Statistics: {garden.num_kills} creatures have been culled so far')
        except KeyboardInterrupt:
            print('Stopping early due to user interrupt')

        # Save results locally
        print('Making local quicksave')
        quicksave_dir = Path(__file__).parent / '.quicksave'
        quicksave_dir.mkdir(parents=True, exist_ok=True)
        for rank, creature in enumerate(garden.creatures):
            quicksave_path = quicksave_dir / f'{args.design_id}_{rank}.pickle'
            with open(quicksave_path, 'wb') as file:
                pickle.dump(creature.design_struct, file, pickle.HIGHEST_PROTOCOL)
            print(f'{rank+1}. Saved to {quicksave_path} achieving score of {creature.best_score}')
        
        # Save results online
        if not user_id:
            print('Skipping upload due to user_id not being set')
        else:
            print(f'Uploading up to {upload_best_k} best creatures')
            for rank, creature in enumerate(garden.creatures[:upload_best_k]):
                saved_design_id = save_design(creature.design_struct, user_id, name=f'Auto {rank+1}', description=f'Based on {args.design_id}, score {creature.best_score}')
                print(f'{rank+1}. Saved to https://ft.jtai.dev/?designId={saved_design_id} achieving score of {creature.best_score}')