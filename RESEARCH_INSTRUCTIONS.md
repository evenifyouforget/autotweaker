# Your playground

We have a separate branch here.
This is also an R&D project, so the PR probably won't be merged, but we will save the results.
Therefore, feel free to create new files as needed and experiment.
Document your findings thoroughly and make sure the next person after you can always continue your work.

# Introduction

See README.md for general information about the project, and in particular the waypoints format and distance calculation.

See the Python code in py_autotweaker/ and the C++ code in backend/ for how it is actually used.

See example/run_example.sh for an entry point.

## Waypoints guide the distance calculation guides the search algorithm

The search algorithm wants to find a variation of the design that solves.
You can consider the individual pieces' (x, y, angle, and maybe width) all bundled up as a point in some high dimensional space, and some tiny subset of this space is a solve.
If we were just searching for a needle in a haystack, this would be very hard, but fortunately there is some order to this problem.
Intuitively, a design could be further or closer to solving, like how in real world navigation, Toronto is closer to Ottawa than Vancouver.
Even without a full understanding of travel, a plain distance heuristic helps that real world navigation immensely, and likewise there is probably some kind of heuristic in this game.
Unfortunately, the search algorithm doesn't just know that - we need to actively equip it with a good heuristic.
A good heuristic may entail not having local minimums to fall into, or having the gradient lead to the solution.
This is why we invented waypoints.

So that's the motivation, but the search algorithm is way down the pipeline from when we set the waypoints.
What information do we have at the time of waypoint generation, and how might we go about generating good waypoints?

## Screenshots

We need some way to understand the level. There were 2 main ways I could have gone about this:

* A geometric model, where we represent all level objects as meshes, or if I'm fancy about circles, as a combination of polylines and curves.
* A pixel model, where pixels in a grid are coloured in using some algorithm, and the level is reduced to a collection of rectangles with some limited precision.

I made the executive decision to go with a pixel model. We might find out later that I was wrong about this, but for now, let's go with this.

This means the level is effectively rendered to an image. We call this a screenshot.

While the screenshot is a simplification of the level in that it doesn't capture the full geometric information, we shouldn't try to go backwards, or use the original level information. In fact, for any waypoint generation algorithm, we should treat the screenshot as if it is in fact the ground truth - the level is always made up of these axis aligned rectangles.

# A simplified blocky world

As far as our waypoints generation is concerned, the world is made up of these grid-aligned rectangles.

There are 3 colours you need to worry about:

* 1 = wall
* 3 = source
* 4 = sink

Anything that's not a wall is passable (air).

Rather than think about how the game actually works in detail, we can think about it as some abstract unknown dynamic process.
At the start of a "run", each source pixel spawns a goal piece at its location.
That's going to get confusing, so let's call it an "ant" instead.
So to recap so far, each source pixel spawns an ant at the start of the run.
We can treat the ants as having zero or infinitesimally small size, but still unable to go through walls.
So, they can hug walls, but not pass through them.
Treat the edges of the arena as walls as well, as in, ants cannot exit the board entirely and go out of bounds.

Important case to be careful of is that we should assume ants cannot cross a checkerboard corner:

..XX
..XX
XX..
XX..

Even though in a sense, there is a 0 width gap there. Just assume ants can't pass through that, as if walls are also made fatter by some infinetesimal value.
Anyway, that's just an edge case, and I think this convention will save us the trouble of dealing with "diagonal movement" later.

The ants will move through the level through some unknown dynamic process, and the desired result is that every ant reaches a sink pixel.

It's probably fine to assume the ants only move through a grid walk in the 4 cardinal directions, and this makes the graph algorithms like path connectivity checking much simpler.
But it will matter sometimes to remember the ants are not confined to the grid, for example, ants need to pass through every waypoint before reaching a sink, and if there is a 0.5 pixel gap between the wall and the waypoint circle and the ant can squeeze through without triggering the waypoint, that's bad for us.
Therefore, waypoints need to reach all the way to the wall with no gaps even for an infinitesimally small ant.

## The distance (score) metric

To avoid confusion with mathematical distance, I'll refer to it as the score of a run.

The run score is the same as usual, just with ants instead of goal pieces. Every ant must pass through every waypoint in order, then reach a sink.

Working backwards, every ant wants to reach a sink, and that is the original goal, and what we want to do is place waypoints such that if an ant always tries to locally minimize its distance to the next waypoint (with a sink being the implied last waypoint), it should not get stuck in a local minimum.

## World space and pixel space

It will be easier to work in pixel space, and only convert to world space at the very end. You can assume the aspect ratio for the screenshot is about right, and therefore, circles will not get significantly deformed (or if they are, that's an issue with our screenshot dimensions).

# What makes a good waypoints list

## The requirements

For a waypoints list to be useful later in the pipeline, it is required that it is not possible to skip waypoints: Every possible path from a source to a sink must pass through every waypoint in order.

Hint: You can break down this problem into manageable parts. You don't actually have to check every single path separately. See graph algorithms for fancy tricks, in particular cuts and path connectivity will be useful concepts here.

## The nice to have

* No local valleys: It should not be possible to get stuck in a corner that locally minimizes the Euclidean distance to the next waypoint.
* No huge circles: Waypoint circles should not be huge relative to the image. There's no reason why we strictly need to disallow huge circles, but it doesn't seem likely.
* No overlap: Waypoint circles should not overlap with each other, and also not overlap with sources or sinks. Also sort of just a quality check.
* No detours: The path through all waypoints should not be much longer than the shortest path the ants could take if they could move at any angle through continuous Euclidean space. Also just a quality check.
* Less waypoints: Redundant to say this, but excessive waypoints are also a bad smell, so maybe we should reward a shorter list.

Of these, the first one, "no local valleys" is by far the most important and fundamental metric for a list of waypoints to be good or bad, since getting stuck in local minimums has a clear impact on the search algorithm at the end of the pipeline.

## Scoring

This all seems to lead naturally into, "before we think about generating waypoints automatically, let's think about automatically evaluating a list of waypoints".
Yes, that is a good start.

Instructions:

* Create a new function in its own Python file, which takes a screenshot and a list of waypoints, and produces a score (lower is better)
* Also create a function that checks if a list of waypoints meets the requirements of non-skippability.
* Give the score function an option to harshly penalize not meeting the non-skippability requirements

### Getting fancy

Now that we have a score function, we can try to make it better.

* Search the web
* Think about it
* Incorporate other bonuses or penalties for things which are likely to make a waypoints list good or bad
* Adjust curves or formulas

Keep in mind we are going to attach a test rig to this later, that automatically evaluates all our candidate waypoint lists.
Maybe we will even use this to try to auto-optimize waypoint lists.
We will be able to see which extra features are actually helping.
Final acceptance of extra features should be based on evidence.
It may help to make extra features toggle-able with a flag in the function.

Evidence should always win, even if it tells us to use something counterintuitive.

# The tournament

Now that we effectively have a grader, it's time to start making some contestants.

Start with setting up the base class and just the null algorithm as a subclass, and a tournament test rig to gather all the subclasses, run them all, and rank their relative performance.

Considering we are generating a score per contestant per test case, and these scores may be in very different ranges, you will need to come up with some way to combine these into a comprehensive and sensible overall rank and score for each contestant.

## The null algorithm

An empty waypoints list is always valid. It's the baseline. If the empty waypoints list is not valid, either the grader is wrong, or there is genuinely an issue with the level itself (sink not reachable from source, for example).

## The corner turning algorithm

This is a recursive algorithm.

If you can draw a straight line from every source to a sink without hitting a wall, we're done, return an empty list.

Otherwise, we try to make a waypoint.

We run a pathfinding algorithm from the sinks to the sources. We trace along this path until the first point that is "obscured" as in we cannot draw a straight line from it to a sink without hitting a wall.
In this way, we have detected a corner.

We spawn a candidate waypoint at the last non-obscured point, and we try stochastically expanding the waypoint like a balloon to fill the "hallway" where this corner is, always making sure the candidate waypoint is not obscured.
The waypoint wants to expand, but it doesn't want to cut into walls (it is repelled by touching walls).
It is also attracted to the original corner like by a spring.
At some point, we stop this balloon iteration, maybe with an iteration limit.

If turning the all pixels covered by the waypoint into walls results in there no longer being path connectivity from sources to sinks, that means the waypoint should in fact be non-skippable, and we can accept it.

If the waypoint was rejected for any reason, we terminate and return an empty list.

If the waypoint was accepted, do tail recursion like:

```py
return corner_algorithm(source=source, sink=new_waypoint, wall=wall) + [new_waypoint]
```

The resulting flow is we construct the last waypoint first, and work backwards until we are unable to produce more waypoints.

## Other algorithms

If we have the tournament, we might as well make use of it!

* Try all kinds of algorithms. Bad algorithms will just be rejected anyway, there's no harm in experimenting
* Try non-deterministic, stochastic, or optimizing algorithms, rather than only static deterministic algorithms. Maybe guided optimization will produce better results?
* Try variations on current winners. Can we do even better?
* Think about other ways the problem could be approached, given the requirements and information available to the algorithm
* Search online for inspiration
* Try more weird things

You may also want to start timing or profiling.
While this is R&D, excessive runtimes may end up being a concern, and hinder further research.

# Test cases

See maze_like_levels.tsv for a list of level IDs for levels that don't use very weird features, and therefore, should be easy to process in the framework.
Though, easy to process doesn't mean easy to generate waypoints for.

You may want to sanity check each level that a sink is actually reachable from every source, and if not, exclude it from the tournament.