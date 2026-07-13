import os
import sys
import subprocess
import platform

def check_python_version():
    major, minor, micro = sys.version_info[:3]
    print(f"[INFO] Detected Python {major}.{minor}.{micro}")
    if major < 3 or minor < 10:
        print("[WARNING] This project was built on Python 3.13. You are using an older version.")
        print("[WARNING] We recommend Python 3.10+ for full compatibility.")
    elif major == 3 and minor != 13:
        print(f"[INFO] Note: Project was built with Python 3.13.x, but you have 3.{minor}.x.")

def create_directories():
    dirs = [
        "data",
        "Model/models/final",
        "Model/models/debug",
        "Model/reports/final",
        "Model/reports/debug"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"[OK] Ensured directory exists: {d}")

def main():
    print("[INFO] Using existing requirements.txt.")
    check_python_version()
    create_directories()

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
