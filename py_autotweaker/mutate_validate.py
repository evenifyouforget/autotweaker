import random
from get_design import FCPieceStruct, FCDesignStruct

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

def is_design_valid(design: FCDesignStruct) -> bool:
    # TODO actual validation
    return True

def generate_mutant(design: FCDesignStruct) -> FCDesignStruct:
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