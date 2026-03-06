"""
Utilities for R integration via rpy2
"""
import os
import sys
from pathlib import Path
import rpy2

def setup_r_environment():
    """
    Setup R environment with automatic path detection.
    
    This function attempts to locate R installation automatically and
    sets up the necessary environment variables for rpy2 to work.
    
    Raises:
        RuntimeError: If R installation cannot be found
    """
    if 'R_HOME' not in os.environ:
        # Try common R installation paths on Windows
        possible_paths = [
            r'C:\Program Files\R\R-4.5.2',
            r'C:\Program Files\R\R-4.5.1',
            r'C:\Program Files\R\R-4.5.0',
            r'C:\Program Files\R\R-4.4.3',
            r'C:\Program Files\R\R-4.4.2',
            r'C:\Program Files\R\R-4.4.1',
            r'C:\Program Files\R\R-4.4.0',
            r'C:\Program Files\R\R-4.3.3',
            r'C:\Program Files\R\R-4.3.2', 
            r'C:\Program Files\R\R-4.3.1',
            r'C:\Program Files\R\R-4.3.0',
            r'C:\Program Files\R\R-4.2.3',
            r'C:\Program Files\R\R-4.2.2',
            r'C:\Program Files\R\R-4.2.1',
            r'C:\Program Files\R\R-4.2.0',
            r'C:\Program Files\R\R-4.1.3',
            # Alternative installation paths
            r'C:\Program Files (x86)\R\R-4.5.2',
            r'C:\Program Files (x86)\R\R-4.5.1',
            r'C:\Program Files (x86)\R\R-4.4.1',
            r'C:\Program Files (x86)\R\R-4.3.3',
            r'C:\Users\%USERNAME%\Documents\R\R-4.5.2',
            r'C:\Users\%USERNAME%\Documents\R\R-4.5.1',
            r'C:\Users\%USERNAME%\Documents\R\R-4.4.1',
            # Unix/Linux paths (in case this runs on other systems)
            '/usr/lib/R',
            '/usr/local/lib/R',
            '/opt/R',
        ]
        
        r_home = None
        for path in possible_paths:
            # Expand environment variables like %USERNAME%
            expanded_path = os.path.expandvars(path)
            if Path(expanded_path).exists():
                r_home = expanded_path
                break
        
        if r_home is None:
            raise RuntimeError(
                "R installation not found. Please either:\n"
                "1. Install R from https://cran.r-project.org/\n" 
                "2. Set R_HOME environment variable manually\n"
                "3. Add R to your system PATH\n\n"
                f"Searched in the following locations:\n" + 
                "\n".join([f"  - {os.path.expandvars(p)}" for p in possible_paths])
            )
        
        os.environ['R_HOME'] = r_home
        
        # Set up PATH for R binaries
        if sys.platform.startswith('win'):
            bin_path = Path(r_home) / 'bin' / 'x64'
        else:
            bin_path = Path(r_home) / 'bin'
            
        if bin_path.exists():
            os.environ['PATH'] += os.pathsep + str(bin_path)
        
        print(f"Using R installation at: {r_home}")
    else:
        print(f"Using existing R_HOME: {os.environ['R_HOME']}")


def import_r_packages():
    """
    Import required R packages for the analysis.
    
    Returns:
        tuple: (latentcor, fastfrechet) - imported R packages
        
    Raises:
        ImportError: If rpy2 cannot be imported or packages are not available
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
    except ImportError as e:
        raise ImportError(
            "Failed to import rpy2. Please install it with:\n"
            "pip install rpy2\n\n"
            f"Original error: {e}"
        )
    
    try:
        # Import the required R packages
        latentcor = importr('latentcor')
        fastfrechet = importr('fastfrechet')
        return latentcor, fastfrechet
        
    except Exception as e:
        raise ImportError(
            "Failed to import R packages. Please install them in R with:\n"
            "install.packages(c('latentcor', 'fastfrechet'))\n\n"
            f"Original error: {e}"
        )


def get_r_objects():
    """
    Get R objects module.
    
    Returns:
        rpy2.robjects module
    """
    try:
        import rpy2.robjects as ro
        return ro
    except ImportError as e:
        raise ImportError(f"Failed to import rpy2.robjects: {e}")
