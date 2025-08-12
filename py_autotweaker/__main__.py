import argparse
from pathlib import Path
from .program_local import program_local

def main():
    # Create the top-level parser
    parser = argparse.ArgumentParser(
        description='General entry point to all things autotweaker'
    )
    subparsers = parser.add_subparsers(dest='command', help='* local for the local runner')

    # Create the parser for the "local" sub-command
    parser_local = subparsers.add_parser('local', help='Run the autotweaker locally on a design')
    parser_local.add_argument('-d', '--design-id', type=int, required=True, help='Design ID to autotweak')
    parser_local.add_argument('-c', '--config-file', type=Path, required=True, help='Config JSON file to use for this job')
    parser_local.add_argument('-g', '--do-generate-config', action='store_true', help='Enable the generate config pass')
    parser_local.add_argument('-a', '--do-autotweak', action='store_true', help='Enable the autotweak pass')
    parser_local.add_argument('-t', '--timeout-seconds', type=float, default=5.0, help='Stop if the program has been running for this many seconds')
    parser_local.add_argument('-w', '--stop-on-win', action='store_true', help='Stop if the design finds a solve')
    parser_local.add_argument('-n', '--max-threads', default='auto', help='Override the max threads that will be used')
    parser_local.add_argument('-k', '--upload-best-k', type=int, default=3, help='Upload this many of the top performers')
    parser_local.add_argument('-m', '--max-garden-size', help='Override number of creatures to keep in garden')
    parser_local.add_argument('-o', '--output-thumbnail-image', type=Path, help='We generate a thumbnail image for computer vision. With this option you can save it to a file.')
    parser_local.add_argument('-W', '--thumbnail-width', type=int, default=200, help='Thumbnail width in pixels (default: 200)')
    parser_local.add_argument('-H', '--thumbnail-height', type=int, default=145, help='Thumbnail height in pixels (default: 145)')
    parser_local.set_defaults(func=program_local)

    # Parse the arguments
    args = parser.parse_args()

    # Call the appropriate function
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()