import os

def merge_text_files(input_folder, output_file):
    """
    Copies the contents of all text files in a folder and pastes them into a single output text file.

    :param input_folder: Path to the folder containing text files.
    :param output_file: Path to the output text file.
    """
    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: The folder '{input_folder}' does not exist.")
        return

    # Open the output file in write mode
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Iterate through all files in the input folder
        for filename in os.listdir(input_folder):
            # Check if the file is a text file
            if filename.endswith('.txt'):
                file_path = os.path.join(input_folder, filename)
                try:
                    # Open and read the text file
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        # Write the contents to the output file
                        outfile.write(infile.read())
                        outfile.write("\n\n")  # Add spacing between files
                    print(f"Copied contents from '{filename}' to '{output_file}'")
                except Exception as e:
                    print(f"Error reading file '{filename}': {e}")

    print(f"All text files have been merged into '{output_file}'.")

# Example usage
input_folder = "D:/misc/langs/python/seminar/results"  # Replace with the path to your input folder
output_file = "output.txt"     # Replace with the desired output file path

merge_text_files(input_folder, output_file)