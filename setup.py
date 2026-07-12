import os
import sys
import subprocess
import platform

requirements = [
    "pandas",
    "numpy",
    "holidays",
    "scikit-learn",
    "xgboost",
    "statsmodels",
    "matplotlib",
    "seaborn"
]

def main():
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(requirements) + "\n")
    print("[OK] requirements.txt written.")

    venv_dir = ".venv"
    if not os.path.exists(venv_dir):
        print("[INFO] Creating virtual environment (.venv)...")
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
        print("[OK] venv created.")
    else:
        print("[INFO] Reusing existing venv.")

    is_windows = platform.system() == "Windows"
    venv_python = os.path.join(venv_dir, "Scripts", "python.exe") if is_windows else os.path.join(venv_dir, "bin", "python")

    print("[INFO] Installing dependencies...")
    subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([venv_python, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    print("[OK] Setup complete.")

if __name__ == "__main__":
    main()
