import os

def pack_repo(output_file="current_reality.txt"):
    """Packs code, tests, and outputs into a single file for AI review."""
    with open(output_file, "w") as f:
        
        # 1. The Code (Source Logic)
        f.write("# SECTION 1: SOURCE CODE\n")
        for root, _, files in os.walk("src"):
            if "__pycache__" in root: continue
            for file in files:
                if file.endswith(".py"):
                    path = os.path.join(root, file)
                    f.write(f"\n--- FILE: {path} ---\n")
                    with open(path, "r") as source:
                        f.write(source.read())

        # 2. The Truth (Test Results)
        f.write("\n\n# SECTION 2: TEST REPORT\n")
        # Ensure you run 'pytest > reports/test_results.log' before running this script
        if os.path.exists("reports/test_results.log"):
            f.write("--- FILE: reports/test_results.log ---\n")
            with open("reports/test_results.log", "r") as log:
                f.write(log.read())
        else:
            f.write("[WARNING] No test logs found. Run tests first!\n")

        # 3. The Output (Sample Artifacts)
        f.write("\n\n# SECTION 3: SAMPLE OUTPUT\n")
        # Change this filename to whatever your main output file is (e.g., report.md, data.json)
        output_path = "data/reports/latest_output.md" 
        if os.path.exists(output_path):
            f.write(f"--- FILE: {output_path} ---\n")
            with open(output_path, "r") as out:
                f.write(out.read())
        else:
            f.write(f"[WARNING] No sample output found at {output_path}.\n")

if __name__ == "__main__":
    pack_repo()
    print(f"✅ Generated current_reality.txt (Upload this to Gemini)")