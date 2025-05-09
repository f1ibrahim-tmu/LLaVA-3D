import json

# Input and output file paths
input_file = "./qwen2.5vl_3d_test_results.json"
output_file = "./qwen2.5vl_3d_test_results_array.json"

# Function to group lines into JSON objects
def read_multiline_json_objects(file_path):
    objects = []
    current_object = []
    open_braces = 0

    with open(file_path, "r") as infile:
        for line in infile:
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            # Count opening and closing braces
            open_braces += line.count("{")
            open_braces -= line.count("}")

            current_object.append(line)

            # If braces are balanced, we have a complete JSON object
            if open_braces == 0 and current_object:
                objects.append(json.loads("".join(current_object)))
                current_object = []

    return objects

# Read the multiline JSON objects
json_objects = read_multiline_json_objects(input_file)

# Write the JSON array to a new file
with open(output_file, "w") as outfile:
    json.dump(json_objects, outfile, indent=4)

print(f"Converted multiline JSON objects to JSON array and saved to {output_file}")

