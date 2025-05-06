from setuptools import setup,Extension

import os
import sys
import numpy
import platform
import subprocess

proc_name = platform.machine()
system = platform.system()

#===================================================================
# So far only concern linux OS
def is_package_installed_ubuntu(package_name):
    result = subprocess.run(["dpkg", "-s", package_name], capture_output=True, text=True)
    return result.returncode == 0

def is_package_installed_fedora(package_name):
    result = subprocess.run(["rpm", "-q",  package_name], capture_output=True, text=True)
    return result.returncode == 0

def check_and_install(package_name, installed):
    if installed:
        print(f"{package_name} is already installed.")
    else:
        print(f"{package_name} is NOT installed.")
        print(f"To install it, run:\n\n    sudo {'apt-get' if distro_id.lower() in ['ubuntu', 'debian'] else 'dnf'} install {package_name}\n")
        print("Then re-run this setup script.")
        sys.exit(1)

#==================================================================

try:
    if system=='Darwin':   #Mac OS
        #Check MacOS architecture
        if proc_name in ['arm64','x86_64']:
            # Check and install GSL via Homebrew
            try:
                brew_list_output = subprocess.check_output(["brew", "list"]).decode("utf-8")
                if "gsl" not in brew_list_output:
                    raise Exception("GSL not found. User should install GSL before proceeding with the ANTARESS installation.")
                else:
                    print("GSL is already installed.")
            except subprocess.CalledProcessError as e:
                print(f"Failed to check GSL install with brew: {e}")
                sys.exit(1)

            gsl_path = subprocess.check_output(["brew", "--prefix", "gsl"]).decode("utf-8").strip()
            gsl_include_dir = [os.path.join(gsl_path, "include")]
            gsl_lib_dir = [os.path.join(gsl_path, "lib")]

            # Check and install libcerf via Homebrew
            try:
                # Check if libcerf is installed using brew list
                brew_list_output = subprocess.check_output(["brew", "list"]).decode("utf-8")
                if "libcerf" not in brew_list_output:
                    print("libcerf not found. Installing with Homebrew...")
                    subprocess.run(["brew", "install", "libcerf"], check=True)
                    print("libcerf installed successfully.")
                else:
                    print("libcerf is already installed.")
            except subprocess.CalledProcessError as e:
                print(f"Failed to check/install libcerf with brew: {e}")
                sys.exit(1)

            libcerf_path = subprocess.check_output(["brew", "--prefix", "libcerf"]).decode("utf-8").strip()
            libcerf_include_dir = [os.path.join(libcerf_path, 'include')]
            libcerf_lib_dir = [os.path.join(libcerf_path, 'lib')]

        else:
            print("Unknown processor architecture.")
            sys.exit(1)


    elif system == 'Linux':   #Linux
        # Detect distro to distinguish which Linux system user has
        distro_id = ""
        try:
            if os.path.exists('/etc/os-release'):
                with open('/etc/os-release', 'r') as f:
                    for line in f:
                        if line.startswith('ID='):
                            distro_id = line.strip().split('=')[1].strip('"').lower()
                            print("Detected Linux distribution:", distro_id)
                            break
        
        except subprocess.CalledProcessError as e:
            print(f"Failed to detector Linux distribution: {e}")
            sys.exit(1)

        # Define package name
        cerf_pkg = "libcerf-dev" if distro_id in ['ubuntu', 'debian'] else "libcerf-devel"
        gsl_pkg = "libgsl-dev" if distro_id in ['ubuntu', 'debian'] else "gsl-devel"
        # actually more like elif distro_id in ['fedora', 'rhel', 'centos'] else "libcerf-devel"

        # Check installation status
        if distro_id in ['ubuntu', 'debian']:
            cerf_installed = is_package_installed_ubuntu(cerf_pkg)
            gsl_installed = is_package_installed_ubuntu(gsl_pkg)
        elif distro_id in ['fedora', 'rhel', 'centos']:
            cerf_installed = is_package_installed_fedora(cerf_pkg)
            gsl_installed = is_package_installed_fedora(gsl_pkg)
        else:
            print(f"Unsupported Linux distribution: {distro_id}")
            sys.exit(1)
        
        check_and_install(cerf_pkg, cerf_installed)
        check_and_install(gsl_pkg, gsl_installed)


except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)

#Paths
gsl_include_dir = ['/usr/include']
gsl_lib_dir = ['/usr/lib', '/usr/lib/x86_64-linux-gnu', '/usr/lib64'] #add the third path as sometimes gsl is in there for Fedora
libcerf_include_dir = ['/usr/include']
libcerf_lib_dir = ['/usr/lib', '/usr/lib/x86_64-linux-gnu','/usr/lib64']

#Getting numpy location
numpy_include_dir = [numpy.get_include()]

# Define extension module
module1 = Extension( 
    'C_star_grid',
    include_dirs=numpy_include_dir + gsl_include_dir + libcerf_include_dir,
    library_dirs=gsl_lib_dir + libcerf_lib_dir,
    runtime_library_dirs=libcerf_lib_dir,
    libraries=['gsl', 'gslcblas', 'cerf'],
    sources=[os.path.join("src", "antaress", "ANTARESS_analysis", "C_grid", "C_star_grid.c")],
    extra_compile_args=['-Wall', '-O2', '-g'], 
    extra_link_args=['-Wl,-v'] 
)

setup(
    name = 'Co-add stellar grid of profiles',
    version = '1.0',
    description = '',
    ext_modules = [module1]
    )
