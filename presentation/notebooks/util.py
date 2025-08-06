import os

def go_to_project_root():
    """
    Changes the current working directory (CWD) to the project's root directory.
    
    This function assumes the project root is the directory containing a specific
    marker file, like 'pyproject.toml', 'main.py', or '.git'. The function
    searches for this marker by traversing up the directory tree.
    If the marker is not found, the function returns without changing the CWD.
    """
    
    # Define a marker file that identifies the project root.
    # Change this to a file that uniquely exists in your project's root.
    marker_file = "README.md"
    
    # Get the current working directory.
    current_directory = os.getcwd()
    
    # Loop to traverse up the directory tree until the marker is found or
    # we reach the system's root directory.
    while True:
        # Check if the marker file exists in the current directory.
        if marker_file in os.listdir(current_directory):
            # If found, change the CWD and exit the function.
            os.chdir(current_directory)
            print(f"CWD changed to the project root: {os.getcwd()}")
            return
        
        # If not found, move up one directory.
        parent_directory = os.path.dirname(current_directory)
        if parent_directory == current_directory:
            print("Project root marker not found. The CWD was not changed.")
            return
        
        # Update the current directory for the next iteration.
        current_directory = parent_directory