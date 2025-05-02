from setuptools import setup,Extension

import os
import sys
import numpy
import platform
import subprocess

proc_name = platform.machine()
system = platform.system()

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
        try:
            distro_id = ""
            if os.path.exists('/etc/os-release'):
                with open('/etc/os-release', 'r') as f:
                    for line in f:
                        if line.startswith('ID='):
                            distro_id = line.strip().split('=')[1].strip('"').lower()
                            break
        
        except subprocess.CalledProcessError as e:
                print(f"Failed to detector Linux distribution: {e}")
                sys.exit(1)

        #Ubuntu/Debian image
        if distro_id in ['ubuntu', 'debian']:
            # Check and install GSL via apt
            try: 
                apt_list_output = subprocess.run(["dpkg", "-s", "libgsl-dev"], capture_output=True, text=True)
                if apt_list_output.returncode != 0:
                    raise Exception("GSL not found. User should install GSL before proceeding with the ANTARESS installation.")
                else:
                    print("libgsl-dev is already installed.")
            except subprocess.CalledProcessError as e:
                print(f"Failed to check GSL install with apt: {e}")
                sys.exit(1)

            # Check and install libcerf via apt
            try:
                apt_list_output = subprocess.run(["dpkg", "-s", "libcerf-dev"], capture_output=True, text=True)
                if apt_list_output.returncode != 0:
                    print("libcerf-dev not found. Installing with apt...")
                    subprocess.run(["apt-get", "update"], check=True)
                    subprocess.run(["apt-get", "install", "-y", "libcerf-dev"], check=True)
                    print("libcerf-dev installed successfully.")
                else:
                    print("libcerf-dev is already installed.")
            except subprocess.CalledProcessError as e:
                print(f"Failed to check/install libcerf with apt: {e}")
                sys.exit(1)
        
        #Fedora/RHEL/Centos image
        elif distro_id in ['fedora', 'rhel', 'centos']:
            # Check and install GSL via dnf
            try: 
                dnf_list_output = subprocess.run(["dnf", "list", "installed", "libgsl-devel"], capture_output=True, text=True)
                if dnf_list_output.returncode != 0:
                    raise Exception("GSL not found. User should install GSL before proceeding with the ANTARESS installation.")
                else:
                    print("libgsl-devel is already installed.")
            except subprocess.CalledProcessError as e:
                print(f"Failed to check GSL install with dnf: {e}")
                sys.exit(1)

            # Check and install libcerf via dnf
            try:
                dnf_list_output = subprocess.run(["dnf", "list", "installed", "libcerf-devel"], capture_output=True, text=True)
                if dnf_list_output.returncode != 0:
                    print("libcerf-devel not found. Installing with dnf...")
                    subprocess.run(["dnf", "makecache"], check=True)
                    subprocess.run(["dnf", "install", "libcerf-devel"], check=True)
                    print("libcerf-dev installed successfully.")
                else:
                    print("libcerf-devel is already installed.")
            except subprocess.CalledProcessError as e:
                print(f"Failed to check/install libcerf with dnf: {e}")
                sys.exit(1)

        else:
            print(f"Unsupported Linux distribution: {distro_id}")
            sys.exit(1)

        #Paths
        gsl_include_dir = ['/usr/include']
        gsl_lib_dir = ['/usr/lib', '/usr/lib/x86_64-linux-gnu', '/usr/lib64'] #add the third path as sometimes gsl is in there for Fedora
        libcerf_include_dir = ['/usr/include']
        libcerf_lib_dir = ['/usr/lib', '/usr/lib/x86_64-linux-gnu','/usr/lib64']

    else:
        print(f"Unsupported operating system: {system}")
        sys.exit(1)

    #Getting numpy location
    numpy_include_dir = [numpy.get_include()]

except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)

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

setup(name = 'Co-add stellar grid of profiles',
        version = '1.0',
        description = '',
        ext_modules = [module1])