import json
import os

def convert_json_to_txt(input_json_path, output_txt_path):
    """
    Reads a JSON file, pretty-prints its content, and saves it to a text file.

    Args:
        input_json_path (str): The path to the input JSON file.
        output_txt_path (str): The path to the output text file.
    """
    try:
        # Ensure the input JSON file exists
        if not os.path.exists(input_json_path):
            print(f"Error: Input JSON file not found at '{input_json_path}'")
            return

        # Read the JSON data from the input file
        with open(input_json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # Pretty-print the JSON data
        # indent=4 makes the output more readable with 4 spaces for indentation
        # sort_keys=True sorts dictionary keys alphabetically
        pretty_json_text = json.dumps(json_data, indent=4, sort_keys=False)

        # Write the pretty-printed JSON to the output text file
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(pretty_json_text)

        print(f"Successfully converted '{input_json_path}' to '{output_txt_path}'")

    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{input_json_path}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # --- Example Usage ---

    # 1. Create a dummy JSON file for demonstration
    example_json_content = {
        "name": "Alice",
        "age": 30,
        "isStudent": False,
        "courses": [
            {"title": "History I", "credits": 3},
            {"title": "Math II", "credits": 4}
        ],
        "address": {
            "street": "123 Main St",
            "city": "Anytown",
            "zip": "12345"
        }
    }
    dummy_json_file = "example.json"
    with open(dummy_json_file, 'w', encoding='utf-8') as f:
        json.dump(example_json_content, f, indent=4)
    print(f"Created dummy JSON file: '{dummy_json_file}'")

    # Define input and output file paths
    input_file = dummy_json_file
    output_file = "output.txt"

    # Call the conversion function
    convert_json_to_txt(input_file, output_file)

    # You can also test with a non-existent file
    # convert_json_to_txt("non_existent.json", "error_output.txt")

    # Or with a malformed JSON file (uncomment to test)
    # with open("malformed.json", "w") as f:
    #     f.write("{'key': 'value'") # Missing closing brace
    # convert_json_to_txt("malformed.json", "malformed_output.txt")
