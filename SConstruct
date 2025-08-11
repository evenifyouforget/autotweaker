import os
import subprocess

# Create the main environment with same settings as ftlib
env = Environment(CXX='g++', CXXFLAGS='-std=c++17', CPPPATH=[], LIBS=[], LIBPATH=[])

# Set up paths relative to autotweaker root
ftlib_root = 'ftlib'
backend_root = 'backend'

# Add ftlib headers to include path (using absolute paths)
ftlib_src_base = Dir(ftlib_root).Dir('src').abspath
env.Append(CPPPATH=[
    ftlib_src_base,
    os.path.join(ftlib_src_base, 'glib'),
    os.path.join(ftlib_src_base, 'glib_adapter'), 
    os.path.join(ftlib_src_base, 'openlibm'),
    os.path.join(ftlib_src_base, 'sim'),
    os.path.join(ftlib_src_base, 'softfloat'),
    os.path.join(ftlib_src_base, 'spectre')
])

# Create bin directory if it doesn't exist
bin_dir = Dir('bin').abspath
if not os.path.exists(bin_dir):
    os.makedirs(bin_dir)

# Custom flag (same as ftlib)
env['godot'] = False

# First, we need to build the ftlib library
print("Building ftlib dependency...")

# Build ftlib using its own build system
ftlib_build_result = subprocess.run(['scons'], cwd=ftlib_root, capture_output=True, text=True)
if ftlib_build_result.returncode != 0:
    print(f"Error building ftlib: {ftlib_build_result.stderr}")
    Exit(1)

# Copy the ftlib library to our bin directory
import shutil
ftlib_lib_src = os.path.join(ftlib_root, 'bin', 'libftlib.a')
ftlib_lib_dst = os.path.join('bin', 'libftlib.a')
if os.path.exists(ftlib_lib_src):
    shutil.copy2(ftlib_lib_src, ftlib_lib_dst)
    print(f"Copied libftlib.a to {ftlib_lib_dst}")
else:
    print(f"Warning: {ftlib_lib_src} not found")

# Export environment for SCsub files
Export('env')

# Build our backend executable
SConscript('backend/SCsub')

# Add clean target
if 'clean' in COMMAND_LINE_TARGETS:
    # Clean our build files
    Clean('.', ['bin', 'backend/*.o'])
    # Clean ftlib build files
    import subprocess
    def clean_ftlib(target, source, env):
        print("Cleaning ftlib...")
        subprocess.run(['scons', '-c'], cwd='ftlib')
        return None
    
    clean_action = Action(clean_ftlib, "Cleaning ftlib dependency")
    clean_target = env.AlwaysBuild(env.Alias('clean-ftlib', [], clean_action))
    Depends('clean', clean_target)