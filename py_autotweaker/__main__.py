import argparse
from pathlib import Path

def program_local(args):
    pass # TODO placeholder

def main():
    # Create the top-level parser
    parser = argparse.ArgumentParser(
        description='General entry point to all things autotweaker'
    )
    subparsers = parser.add_subparsers(dest='command', help='* local for the local runner')

    # Create the parser for the "local" sub-command
    parser_local = subparsers.add_parser('local', help='Run the autotweaker locally on a design')
    parser_local.add_argument('-d', '--design', type=int, required=True, help='Design ID to autotweak')
    parser_local.add_argument('-c', '--config-file', type=Path, required=True, help='Config file to use for this job')
    parser_local.add_argument('-g', '--generate-config', action='store_true', help='Enable the generate config pass')
    parser_local.add_argument('-a', '--autotweak', action='store_true', help='Enable the autotweak pass')
    parser_local.add_argument('-t', '--timeout', type=float, default=5.0, help='Stop if the program has been running for this many seconds')
    parser_local.add_argument('-w', '--stop-on-win', action='store_true', help='Stop if the design finds a solve')
    parser_local.add_argument('-n', '--max-threads', default='auto', help='Override the max threads that will be used')
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