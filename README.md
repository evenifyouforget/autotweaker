# autotweaker

## Motivation

FC players spend a lot of time tweaking, and most of that is very small movements to a rod or perhaps a wheel, in hopes that one of these iterations will solve (or at least make progress).
One is obviously inclined to ask, what if we could do it automatically?
An autotweaker would work in the background and try many tweaks for us.

## Philosophy and scope

FC is a complex game - if it's not obvious already, just look at the high level mechanical engineering needed for impossible-looking levels, or the explicit combinatorial/geometric puzzles in many levels.
A perfect autotweaker understands what constitutes progress, which means it also needs to intelligently understand all these levels... which is extremely difficult.
Possibly NP-hard in the worst case!

A perfect autotweaker is also able to try making major changes to a design, say, moving a rod a whole 10 units away from its original position.
Sometimes that's just what you need for a winning tweak.
Unfortunately, such major tweaks depend on ftlib having full design editing capabilities, which at the time of writing, it doesn't.
These are nontrivial to develop.
We can make them, we're just taking our time to get it right.

A perfect autotweaker also has a simulation that exactly matches the original game.
Unfortunately ftlib is only something like 70% accurate - which means it's very close already, but sometimes a solve will turn out to be a false positive.

In light of these challenges, we are taking a reduced scope for the autotweaker:

1. Design for the common case, but don't design out the outliers. The autotweaker should be able to comprehend the 90% most straightforward levels, and still make an effort (not explode) on the rest.
2. Focus on small tweaks, since they represent probably 99% of all tweaking. This means if you're not already close, the autotweaker won't save you, but that's okay, the tool is already being very useful.
3. Allow it to continue even after it thinks it solved, to make it solve extra strongly, so the final design is likely resilient even to ftlib simulation bugs.
4. Have a local runnable and a Discord bot as the final deliverables. Some people with PC power to spare may want to run it locally, while for others, it is more convenient to take whatever freely available time credits are available on the Discord bot.
5. Permit a reduced standard of code internally, and don't promise a stable API. Such a "move fast and break things" attitude would normally be insane, but the only deliverable most users will care about is the bot and its public-facing interface, and loosening quality controls lets us make use of coding AIs. AI slop is, admittedly, often acceptable to deliver a small product quickly.

## Basic requirements

These apply to both the local runnable and the Discord bot.

* Run the backend simulation at full performance
* Be multithreaded and support maxing all available cores if the user so desires
* Make changes to the design
* Be able to save the final design to the FC servers
* Measure design performance and keep the winners
* Support a configuration with waypoints and other parameters

## Approach

From [previous autotweaker experiments](https://ft.jtai.dev/auto/) we have seen that a poor fitness heuristic can severely hurt the autotweaker's ability to help with levels, and reduce the set of levels it is suitable for.
We believe there is a significant challenge in designing a fitness function, therefore, we are splitting the autotweaker functionality into 2 distinct phases: waypoint definition and automatic optimization.

* Waypoint definition is what produces the list of waypoints. We will have a program that does its best to automatically set good waypoints, but it is also possible to manually define the waypoints.
* Automatic optimization is where the autotweaker does its thing.

### Waypoints

Have you ever played a racing game?
How does the game know your progress within the track?
Why does it not get confused when the start is basically the same location as the lap location, or when the track twists and turns?

There are different implementations, but one of the most obvious is waypoints.
You don't trigger a lap immediately because there is a list of locations along the track you must pass through first.

Our waypoints here use a similar method.
It helps produce an accurate distance metric on all maze-like levels, though it won't really help for puzzle levels.
It's also optimized for a single goal piece, though certain "berry" levels may also be supported.

Before autotweaking, we configure the level with a list of waypoints.
Each waypoint has a position and a radius.
We expect the goal pieces to reach the waypoints in order, before finally reaching the goal.
A goal piece is considered to have reached a waypoint when its center enters the waypoint circle; therefore, if the goal piece needs to pass through an area, the waypoint should cover from wall to wall so there is no way the goal piece can skip the waypoint.

### Distance (error) calculation

We calculate this function for every goal piece, and sum it to get the total error score for that tick.
The score for the design overall is the lowest score it achieved at any tick.
Due to the waypoints behaviour, this calculation is stateful.

If the goal piece is inside the goal already, we add a negative distanced based on how far safely inside the goal area it is. (the gap between the goal piece and the goal area boundary)

If the goal piece is not inside the goal, we add a large penalty.
This makes every "solving" goal piece be considered very valuable.

We add the distance to the next checkpoint's position.

We add the distance from the next checkpoint's position to the next next checkpoint's position, and so on for every hop.

The goal area center is implicitly the last checkpoint.

Due to triangle math, there is implicitly a small bonus for reaching each checkpoint.

These conventions also mean that a negative score means a solve.