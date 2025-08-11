import os
import subprocess
import sys

# Create environment with same settings as ftlib
env = Environment(CXX='g++', CXXFLAGS='-std=c++17', CPPDEFINES=['HARDFLOAT_TOGGLE'])

print("Building ftlib dependency...")
# Pass through all CLI args so scons -c also works
result = subprocess.run(['scons'] + sys.argv[1:], cwd='ftlib')
if result.returncode != 0:
    print(f"ftlib sub-build failed. Exiting.")
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

# Build measure_single_design executable (same flags as ftlib's cli_adapter)
backend_env = env.Clone()
backend_env.Append(
    CCFLAGS=["-Wall", "-O2", "-flto"],
    LINKFLAGS=["-O2", "-flto"],
)

measure_single_design = backend_env.Program(
    target='bin/measure_single_design',
    source=['backend/main.cpp'],
    LIBS=['ftlib'],
    LIBPATH=['ftlib/bin']
)

Default(measure_single_design)