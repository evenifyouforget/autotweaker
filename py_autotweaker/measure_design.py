from collections import namedtuple
from pathlib import Path
import shlex
import subprocess

RunDesignResult = namedtuple('RunDesignResult', ['proc', 'real_solve_ticks', 'real_end_ticks', 'best_score'])

def measure_design(design_struct, config_data, command_prepend=None, command_append=None):
    # generate serialized input
    serialized_input = []
    serialized_input.append(config_data['tickMaximum'])
    all_pieces = design_struct.goal_pieces + design_struct.design_pieces + design_struct.level_pieces
    serialized_input.append(len(all_pieces))
    for i, piece_struct in enumerate(all_pieces):
        serialized_input.append(piece_struct.type_id)
        serialized_input.append(piece_struct.piece_id if piece_struct.piece_id is not None else i + 1000)
        serialized_input.append(piece_struct.x)
        serialized_input.append(piece_struct.y)
        serialized_input.append(piece_struct.w)
        serialized_input.append(piece_struct.h)
        serialized_input.append(piece_struct.angle)
        joints = list(piece_struct.joints)
        joints += [-1] * (2 - len(joints))
        serialized_input += joints
    serialized_input.append(design_struct.build_area.x)
    serialized_input.append(design_struct.build_area.y)
    serialized_input.append(design_struct.build_area.w)
    serialized_input.append(design_struct.build_area.h)
    serialized_input.append(design_struct.goal_area.x)
    serialized_input.append(design_struct.goal_area.y)
    serialized_input.append(design_struct.goal_area.w)
    serialized_input.append(design_struct.goal_area.h)
    serialized_input.append(config_data['tickMinimum'])
    waypoints_list = config_data['waypoints']
    serialized_input.append(len(waypoints_list))
    for waypoint in waypoints_list:
        serialized_input.append(waypoint['x'])
        serialized_input.append(waypoint['y'])
        serialized_input.append(waypoint['radius'])
    serialized_input = ' '.join(map(str, serialized_input))
    # run the executable
    exec_path = Path(__file__).parent.parent / 'bin' / 'measure_single_design'
    command_prepend = command_prepend or []
    command_append = command_append or []
    command = command_prepend + [exec_path] + command_append
    proc = subprocess.run(command, text=True, input=serialized_input, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        debug_command_text = shlex.join(map(str, command))
        raise AssertionError(f'Process {debug_command_text} exited with return code {proc.returncode}')
    stdout = proc.stdout
    real_solve_ticks, real_end_ticks, best_score = stdout.strip().split()
    real_solve_ticks = int(real_solve_ticks)
    real_end_ticks = int(real_end_ticks)
    best_score = float(best_score)
    return RunDesignResult(
        proc=proc,
        real_solve_ticks=real_solve_ticks,
        real_end_ticks=real_end_ticks,
        best_score=best_score,
        )