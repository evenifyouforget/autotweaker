from collections import namedtuple
import random
import warnings
from get_design import FCPieceStruct, FCDesignStruct

DesignNormalizeResult = namedtuple('DesignNormalizeResult', ['design', 'is_valid', 'is_changed', 'max_diff_ulps'])
DesignNormalizeResult.__doc__ = """Result of normalize_design()
    design: Possibly modified design. Always a new object
    is_valid: False if any of these apply:
    - Any pieces are out of bounds
    - Any pieces are colliding (overlapping and not excluded, according to FC rules)
    - Any pieces have illegal dimensions
    - The resulting joints (indices) are different, or any joints were removed
    - Any pieces are deleted (not a distinct case, just something the backend might do when normalizing other cases)
    - The last design, and last last design, have a field that differs by more than the permitted ulps threshold
    is_changed: False if design is equal to the input design, otherwise True
    max_diff_ulps: The maximum, over all fields, of the difference between the last design and the last last design, measured in ulps (0 = fixed point)
"""

def copy_piece(piece: FCPieceStruct) -> FCPieceStruct:
    return FCPieceStruct(
        type_id=piece.type_id,
        piece_id=piece.piece_id,
        x=piece.x,
        y=piece.y,
        w=piece.w,
        h=piece.h,
        angle=piece.angle,
        joints=list(piece.joints)
    )

def copy_design(design: FCDesignStruct) -> FCDesignStruct:
    return FCDesignStruct(
        name=design.name,
        base_level_id=design.base_level_id,
        goal_pieces=list(map(copy_piece, design.goal_pieces)),
        design_pieces=list(map(copy_piece, design.design_pieces)),
        level_pieces=list(map(copy_piece, design.level_pieces)),
        build_area=copy_piece(design.build_area),
        goal_area=copy_piece(design.goal_area)
    )

def normalize_design(design: FCDesignStruct, max_iters: int = 30, permit_max_diff_ulps: int = 10) -> DesignNormalizeResult:
    """
    Try to normalize a design by simulating loading and resaving the design through fcsim,
    up to a maximum number of iterations.
    This may cause the design to change:
    - All floating point values may change due to the strtod round trip not being exact
    - Joints may be re-snapped
    - Joints may be removed if they are too far apart
    - Joints may be removed if the joint assignment resolves differently ("left rod issue")

    Args:
        design (FCDesignStruct): Original design
        max_iters (int, optional): _description_. Maximum simulated load/save round trips
        permit_max_diff_ulps (int, optional): _description_. Reject if the last round trip changed any floating point value by more than this many ulps

    Returns:
        DesignNormalizeResult: Possibly modified design, tagged with useful info
    """
    if max_iters < 1:
        raise ValueError("max_iters must be at least 1")
    current_design = design
    for _ in range(max_iters):
        last_design = current_design
        current_design = copy_design(current_design) # TODO actual simulated round trip logic
        max_diff_ulps = max(...) # TODO compare fields for current_design and last_design, in ulps
        is_permanently_broken = ... # TODO check for conditions which irreversibly break the design (ex. pieces deleted, since they'll never come back)
        is_valid = not is_permanently_broken and max_diff_ulps <= permit_max_diff_ulps and ... # TODO check for other reject conditions
        if is_permanently_broken or max_diff_ulps == 0:
            break # stop early on ideal result or critical failure
    is_changed = current_design != design # TODO check if you can just use != for this struct (namedtuple, list)
    return DesignNormalizeResult(design=current_design, is_valid=is_valid, is_changed=is_changed, max_diff_ulps=max_diff_ulps)

def is_design_valid(design: FCDesignStruct) -> bool:
    warnings.warn("Please use normalize_design() instead", DeprecationWarning)
    return True

def generate_mutant(design: FCDesignStruct) -> FCDesignStruct:
    """
    Attempt to produce a design that is a slight variation on the input design.
    Resulting design is not guaranteed to be valid.

    Args:
        design (FCDesignStruct): Original design

    Returns:
        FCDesignStruct: Modified design
    """
    mutant = copy_design(design)
    # TODO actual proper mutation
    if mutant.design_pieces:
        index = random.randrange(len(mutant.design_pieces))
        piece = mutant.design_pieces[index]
        mutant.design_pieces[index] = FCPieceStruct(
            type_id=piece.type_id,
            piece_id=piece.piece_id,
            x=piece.x + random.uniform(-1, 1) * 0.1,
            y=piece.y + random.uniform(-1, 1) * 0.1,
            w=piece.w + random.uniform(-1, 1) * 0.1,
            h=piece.h + random.uniform(-1, 1) * 0.1,
            angle=piece.angle + random.uniform(-1, 1) * 0.01,
            joints=list(piece.joints)
        )
    return mutant