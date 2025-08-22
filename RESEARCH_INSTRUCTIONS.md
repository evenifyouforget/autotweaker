# Your playground

We have a separate branch here.
This is also an R&D project, so the PR probably won't be merged, but we will save the results.
Therefore, feel free to create new files as needed and experiment.
Document your findings thoroughly and make sure the next person after you can always continue your work.

# Introduction

See README.md for general information about the project, and in particular the waypoints format and distance calculation.

See the Python code in py_autotweaker/ and the C++ code in backend/ for how it is actually used.

See example/run_example.sh for an entry point.

See ftlib submodule, and fcsim submodule under that.

## The ideal: tweaking the same way a user does

The ideal for the autotweaker would be to tweak in the same way a real user does in the game, just automated and without the GUI.

This would mean moving joints or whole connected components of the design.

What we currently have is not that.

## The current state

See py_autotweaker/mutate_validate.py

We originally set up a model where generate_mutant() makes a mutant version of a design by changing some (x, y, width, height, rotation) values, and then is_design_valid() just returns true.

The ideal behaviour would be for generate_mutant() to perform a "user-like" action instead, and is_design_valid() has actual validation.

## Does it work?

What we have is the bare minimum needed for a search optimization algorithm - a way to enumerate new search candidates, and a way to measure them for success.

The autotweaker is able to run to a solve on the example job, given enough time (5 minutes is usually plenty).

However, taking the resulting designs into the real game, they don't solve as expected.

## Normalization

ftlib more or less runs exactly the design you give it.

The real game, and fcsim (which is closest to the real game), does some kind of conversion when it loads the design into memory. I'll refer to the process overall as "normalization". So by the time you run it, what you're running is the normalized version, and that's why it doesn't solve.

## The obvious solution - just normalize it?

If the issue is that we're not accounting for the normalization step that would happen in the real game... why not just normalize all the designs we try?

Yes, let's do that. I have already provided a stub, normalize_design() which will do that. We just need to implement it.

In fact, doing all the hard work in normalize_design() means generate_mutant() doesn't need to be fancy - the current basic value shifting should be good enough, though it's still good to check a few things.

# Goals

- [ ] generate_mutant() doesn't wiggle any values that are never wiggleable
- [ ] normalize_design() does its job, and conforms to its interface and functional requirements

## generate_mutant() requirements

I believe this is true, but you should double check.

For rods, the values that can change are: x, y, width, and rotation. This means height is fixed.

For everything else, the values that can change are: x, y. This means width, height, and rotation are fixed.

If there's no way for the real user to edit the design in a way that changes a certain field, you should assume it's fixed. Likewise, if there is a way for a user to edit the design in a way that changes a particular field, assume it is variable (and therefore subject to random wiggling).

generate_mutant() needs to check the piece type ID, and only wiggle fields that can actually change.

## normalize_design() requirements

It needs to conform to the interface.

normalize_design()

```py
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
    ...
```

You will need to investigate what fcsim does, in the full flow, for loading a design, all the way to resaving a design. Note that fcsim is designed for XML input. You may want to temporarily build the wrapper, FCDesignStruct -> XML -> fcsim internal representation -> XML -> FCDesignStruct, and then later simplify out the parts to FCDesignStruct -> fcsim internal representation -> FCDesignStruct, and maybe we will find that we can cut out the fcsim internal representation too.

Implement it correctly first. Worry about clean up and simplification later.

## The things that can change, and the reasons for rejecting

Here are all the things I think fcsim may do when you load and resave:

- [ ] All floating point fields may change since the strtod does not meet the usual requirements (the round trip may not be exact)
- [ ] Joints may be re-snapped (positions recalculated)
- [ ] Joints may be removed if they are too far apart
- [ ] Joints may be removed if the joint assignment resolves differently ("left rod issue" occurs because the joints only specify what piece they joint to, not which joint within that piece, and wiggling the pieces may cause the closest match to change in a way that ends up breaking the design)
- [ ] Flag or delete pieces which are outside the build area, even if they only poke outside a little bit
- [ ] Flag or delete pieces that are colliding (overlapping and not excluded by piece type or jointing, according to FC rules)
- [ ] Flag or delete pieces that have illegal dimensions, or change the dimensions to be legal

Here are all the things we need to additionally check for:

- [ ] Any pieces in the final design were invalid for any reason
- [ ] The resulting joints (indices) are different, or any joints were removed, comparing the final design to the original design
- [ ] Any pieces in the final design are deleted compared to the original design
- [ ] The last design, and last last design, have a field that differs by more than the permitted ulps threshold

Again, while this is a pretty good initial hypothesis based on my understanding of the game, you should double check every item.

## Why not just normalize once?

A sane game would normalize once and normalizing it again would just return the same design.

Unfortunately, FC is not sane. The most obvious reason is that strtod does not respect round trip equality.

That is, if we consider a function f(x) = strtod(dtostr(x)), then f(x) = x is not guaranteed.

Technically, since the cold storage format is XML, we should be looking at strings, so it would be g(y) = dtostr(strtod(y)). But if g(y) = y is not guaranteed, that is not directly harmful, since f(x) = x could still be true. g(g(y)) = g(y) not being true would be a reason to be concerned.

Anyway, we can consider f(x) to be very close to x (the identity function). Sometimes f(x) = x and we have a fixed point, which is great. But this is not always the case.

If we pack all the fields within a design into a vector X, then overall, what we have is something like F(X) = [f(x) for x in X]. We can then measure M(X) = max(|ulps_difference(x, f(x)) for x in X|). If M(X) = 0 then it means F(X) = X and we have a fixed point, and this is ideal. But this won't always be the case. Maybe we'll never reach a fixed point, and that's why we need a maximum number of iterations.