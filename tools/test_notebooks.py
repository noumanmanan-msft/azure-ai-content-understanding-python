import os
import sys
from typing import Tuple, Optional, List

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


SINGLE_NOTEBOOK_TIMEOUT = 1200


def should_skip(notebook_path: str, skip_list: List[str]) -> bool:
    return any(skip in notebook_path for skip in skip_list)


def run_notebook(notebook_path: str, root: str) -> Tuple[bool, Optional[str]]:
    """Execute a single notebook."""
    try:
        with open(notebook_path, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(
            timeout=SINGLE_NOTEBOOK_TIMEOUT,
            kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": root}})
        return True, None
    except Exception as e:
        return False, str(e)


def run_all_notebooks(path: str = ".", skip_list: List[str] = None) -> None:
    abs_path = os.path.abspath(path)
    print(f"🔍 Scanning for notebooks in: {abs_path}\n")

    skip_list = skip_list or []

    notebook_found: int = 0
    success_notebooks: List[str] = []
    failed_notebooks: List[Tuple[str, str]] = []

    for root, _, files in os.walk(abs_path):
        for file in files:
            if file.endswith(".ipynb") and not file.startswith("."):
                notebook_path = os.path.join(root, file)

                if should_skip(notebook_path, skip_list):
                    print(f"⏭️ Skipped: {notebook_path}")
                    continue

                notebook_found += 1
                print(f"▶️ Running: {notebook_path}")
                success, error = run_notebook(notebook_path, root)

                if success:
                    print(f"✅ Success: {notebook_path}\n")
                    success_notebooks.append(notebook_path)
                else:
                    print(f"❌ Failed: {notebook_path}\nError: {error}\n")
                    failed_notebooks.append((notebook_path, error))

    # 📋 Summary
    print("🧾 Notebook Execution Summary")
    print(f"✅ {len(success_notebooks)} succeeded")
    print(f"❌ {len(failed_notebooks)} failed\n")

    if failed_notebooks:
        print("🚨 Failed notebooks:")
        for nb, error in failed_notebooks:
            last_line = error.strip().splitlines()[-1] if error else "Unknown error"
            print(f" - {nb}\n   ↳ {last_line}")
        sys.exit(1)

    if notebook_found == 0:
        print("❌ No notebooks were found. Check the folder path or repo contents.")
        sys.exit(1)

    print("🏁 All notebooks completed successfully.")


if __name__ == "__main__":
    args: List[str] = sys.argv[1:]

    # NOTE: Define skip list (can use full paths or substrings)
    skip_list = [
        "build_person_directory.ipynb",  # Skip due to "new_face_image_path" needed to be added manually
    ]

    if not args:
        run_all_notebooks("notebooks", skip_list=skip_list)
    else:
        failed: List[Tuple[str, str]] = []
        for notebook_path in args:
            if should_skip(notebook_path, skip_list):
                print(f"⏭️ Skipped: {notebook_path}")
                continue

            if notebook_path.endswith(".ipynb") and os.path.isfile(notebook_path):
                print(f"▶️ Running: {notebook_path}")
                success, error = run_notebook(notebook_path, os.path.dirname(notebook_path))
                if success:
                    print(f"✅ Success: {notebook_path}\n")
                else:
                    print(f"❌ Failed: {notebook_path}\nError: {error}\n")
                    failed.append((notebook_path, error))
            else:
                print(f"⚠️ Not a valid notebook file: {notebook_path}")
                failed.append((notebook_path, "Invalid path or not a .ipynb file"))

        # Summary
        print("🧾 Execution Summary")
        print(f"✅ {len(args) - len(failed)} succeeded")
        print(f"❌ {len(failed)} failed")

        if failed:
            print("🚨 Failed notebooks:")
            for nb, error in failed:
                last_line = error.strip().splitlines()[-1] if error else "Unknown error"
                print(f" - {nb}\n   ↳ {last_line}")
            sys.exit(1)
        else:
            print("🏁 All selected notebooks completed successfully.")
