import argparse
import subprocess
import sys
import os
import io
import warnings # To help manage warnings from libraries

# Temporarily add LLaVA to path to import its modules
# This might need adjustment based on where the script is run relative to LLaVA_3D
# Get the absolute path of the directory containing the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming the script is in a subdirectory like LLaVA_3D/scripts, and llava is in LLaVA_3D/llava
# Adjust a_path_to_llava_parent_dir as needed if script is elsewhere e.g. ../ for one level up
a_path_to_llava_parent_dir = os.path.join(current_script_dir, '..') 
sys.path.append(os.path.abspath(a_path_to_llava_parent_dir))

try:
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import get_model_name_from_path
except ImportError as e:
    print(f"Error importing LLaVA modules: {e}")
    print("Please ensure that check_model_setup.py is placed correctly within the LLaVA-3D project structure,")
    print("or that the LLaVA package is installed and accessible in your PYTHONPATH.")
    print(f"Attempted to add to sys.path: {os.path.abspath(a_path_to_llava_parent_dir)}")
    sys.exit(1)

def check_git_revision(expected_hash_prefix):
    print("\n--- Checking Git Revision ---")
    try:
        process = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        current_hash = process.stdout.strip()
        print(f"Current Git HEAD (short): {current_hash}")
        if current_hash.startswith(expected_hash_prefix):
            print(f"SUCCESS: Git hash matches expected prefix '{expected_hash_prefix}'.")
            return True
        else:
            print(f"WARNING: Git hash '{current_hash}' does not match expected prefix '{expected_hash_prefix}'.")
            return False
    except FileNotFoundError:
        print("ERROR: Git command not found. Please ensure Git is installed and in your PATH.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Git command failed: {e}")
        print(f"Output: {e.stderr}")
        return False

def check_lora_weights(args, model_name, captured_output):
    print("\n--- Checking LoRA Weights ---")
    lora_loaded_by_log = False
    lora_weights_file_found = False

    if "lora" in model_name.lower() or args.model_base is not None:
        if args.model_base is None:
            print("WARNING: 'lora' is in model_name but no --model-base was provided. LoRA models typically require a base model.")
            # This case is often handled by load_pretrained_model warning as well.
        else:
            print(f"INFO: --model-base '{args.model_base}' provided, suggesting LoRA usage.")
            
            # Check for LoRA weight file existence
            lora_file_path = os.path.join(args.model_path, args.lora_weights_filename)
            if os.path.exists(lora_file_path):
                print(f"INFO: LoRA weights file '{args.lora_weights_filename}' found at '{lora_file_path}'.")
                lora_weights_file_found = True
            else:
                # Try alternative common name if the default was adapter_model.bin and user asked for delta_weights.pt (or vice-versa)
                alt_lora_filename = "delta_weights.pt" if args.lora_weights_filename == "adapter_model.bin" else "adapter_model.bin"
                alt_lora_file_path = os.path.join(args.model_path, alt_lora_filename)
                if os.path.exists(alt_lora_file_path):
                     print(f"INFO: Alternative LoRA weights file '{alt_lora_filename}' found at '{alt_lora_file_path}'.")
                     lora_weights_file_found = True # Consider this a find
                else:
                    print(f"WARNING: Specified LoRA weights file '{args.lora_weights_filename}' (and alternative '{alt_lora_filename}') not found in '{args.model_path}'.")


            # Check logs for merging
            if "Merging LoRA weights..." in captured_output:
                print("INFO: Log message \"Merging LoRA weights...\" found during model loading.")
                lora_loaded_by_log = True
            else:
                print("WARNING: Log message \"Merging LoRA weights...\" NOT found. This might indicate LoRA weights were not merged.")
            
            if lora_weights_file_found and lora_loaded_by_log:
                print("SUCCESS: LoRA weights file found and merging process logged.")
                return True
            elif lora_weights_file_found and not lora_loaded_by_log:
                print("POTENTIAL ISSUE: LoRA weights file found, but merging log not detected. Check model loading verbosity and procedure.")
                return False
            elif not lora_weights_file_found and lora_loaded_by_log: # Less likely scenario
                print("POTENTIAL ISSUE: LoRA merging logged, but specified weights file not found. Check paths and filenames.")
                return False
            else: # Neither found
                print("ERROR: LoRA setup seems problematic. Neither weights file nor merging log detected as expected.")
                return False
                
    else:
        print("INFO: Not a LoRA setup (no 'lora' in model_name and no --model-base provided). Skipping LoRA checks.")
        return True # Not a failure if not a LoRA setup
    return False # Default if conditions aren't met

def check_dtype_warnings(captured_output):
    print("\n--- Checking Dtype Mismatches ---")
    if "Found mismatched dtype" in captured_output or "mismatched dtype" in captured_output.lower(): # More general check
        print("WARNING: \"Found mismatched dtype\" (or similar) warning detected during model loading. This could impact performance/precision.")
        return False
    else:
        print("SUCCESS: No explicit \"Found mismatched dtype\" warnings detected.")
        return True

def main():
    parser = argparse.ArgumentParser(description="Check LLaVa-3D model setup and artifacts.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the LLaVa-3D model checkpoint (directory or file).")
    parser.add_argument("--model-base", type=str, default=None, help="Optional: Path to the base LLM if loading LoRA weights.")
    parser.add_argument("--lora-weights-filename", type=str, default="adapter_model.bin", help="Filename of the LoRA weights (e.g., adapter_model.bin, delta_weights.pt). Looked for in model_path.")
    parser.add_argument("--expected-git-hash-prefix", type=str, default="49cf", help="Expected short Git hash prefix of the LLaVA-3D paper.")
    # Add other args that load_pretrained_model might need, e.g., device, quantization
    parser.add_argument("--device", type=str, default="cuda", help="Device to load the model on.")
    parser.add_argument("--load_4bit", action='store_true', help="Load model in 4-bit.")
    parser.add_argument("--load_8bit", action='store_true', help="Load model in 8-bit.")


    args = parser.parse_args()

    print(f"Running checks for model: {args.model_path}")
    if args.model_base:
        print(f"Using base model: {args.model_base}")

    # 1. Git Revision Check
    git_ok = check_git_revision(args.expected_git_hash_prefix)

    # Capture stdout/stderr for model loading checks
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = captured_stdout = io.StringIO()
    sys.stderr = captured_stderr = io.StringIO()

    model_loaded_successfully = False
    model_name_for_checks = get_model_name_from_path(args.model_path)
    
    try:
        disable_torch_init()
        print(f"Attempting to load model: {args.model_path} with base: {args.model_base}...") # This print goes to StringIO
        # Note: processor and context_len are not used in this script but are returned by the function
        # The torch_dtype in load_pretrained_model defaults to bfloat16 if not 4bit/8bit
        # LLaVa-3D paper might use bf16, but if user has GTX card it might cast to fp16.
        # We are looking for explicit warnings.
        _tokenizer, _model, _processor, _context_len = load_pretrained_model(
            model_path=args.model_path,
            model_base=args.model_base,
            model_name=model_name_for_checks, # Important to pass model_name
            load_8bit=args.load_8bit,
            load_4bit=args.load_4bit,
            device=args.device,
            # use_flash_attn can be added if needed by default by some models
        )
        model_loaded_successfully = True
        print("Model loading call completed.") # This print goes to StringIO
    except Exception as e:
        print(f"ERROR during model loading: {e}") # This print goes to StringIO
    finally:
        stdout_output = captured_stdout.getvalue()
        stderr_output = captured_stderr.getvalue()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        print("\n--- Captured output during model loading ---")
        if stdout_output:
            print("Stdout:\n" + stdout_output)
        if stderr_output:
            print("Stderr:\n" + stderr_output)
        print("--- End of captured output ---")


    if not model_loaded_successfully:
        print("\nERROR: Model could not be loaded. Subsequent checks for LoRA (via logs) and dtype might be unreliable.")
        # Allow checks to proceed with potentially empty/partial logs if user wants to inspect

    combined_output = stdout_output + "\n" + stderr_output

    # 2. LoRA Weights Check
    # Pass model_name_for_checks which is derived before trying to load
    lora_ok = check_lora_weights(args, model_name_for_checks, combined_output)

    # 3. Dtype Mismatch Check
    dtype_ok = check_dtype_warnings(combined_output)

    print("\n--- Summary of Checks ---")
    print(f"Git Revision Check:         {'PASS' if git_ok else 'FAIL/WARN'}")
    print(f"LoRA Weights Check:         {'PASS/INFO' if lora_ok else 'FAIL/WARN'}") # INFO if not LoRA setup
    print(f"Dtype Mismatch Warnings:    {'PASS' if dtype_ok else 'WARN'}")
    
    if git_ok and lora_ok and dtype_ok:
        print("\nAll critical checks passed (or INFO for non-LoRA setup).")
    else:
        print("\nOne or more checks raised warnings or potential issues. Please review the logs above.")

if __name__ == "__main__":
    main() 