import os
import subprocess

# Create environment with same settings as ftlib
env = Environment(CXX='g++', CXXFLAGS='-std=c++17', CPPPATH=[], LIBS=[], LIBPATH=[])
env['godot'] = False

# First build ftlib dependency
print("Building ftlib dependency...")
result = subprocess.run(['scons'], cwd='ftlib', capture_output=True, text=True)
if result.returncode != 0:
    print(f"Error building ftlib: {result.stderr}")
    Exit(1)

# Set up include paths to ftlib
ftlib_src = 'ftlib/src'
env.Append(CPPPATH=[
    ftlib_src,
    f'{ftlib_src}/glib', 
    f'{ftlib_src}/glib_adapter',
    f'{ftlib_src}/openlibm',
    f'{ftlib_src}/sim',
    f'{ftlib_src}/softfloat', 
    f'{ftlib_src}/spectre',
    f'{ftlib_src}/box2d/Include'
])

# Build autotweaker executable (same flags as ftlib's cli_adapter)
backend_env = env.Clone()
backend_env.Append(
    CCFLAGS=["-Wall", "-O2", "-flto"],
    LINKFLAGS=["-O2", "-flto"]
)

autotweaker = backend_env.Program(
    target='bin/autotweaker',
    source=['backend/main.cpp'],
    LIBS=['ftlib'],
    LIBPATH=['ftlib/bin']
)

Default(autotweaker)