# FIDUCIA — Setup Guide (from scratch)

How to run FIDUCIA on a fresh macOS or Windows machine. No prior setup assumed.

FIDUCIA runs **entirely on your own machine** — the web app is served locally and
the AI runs through **Ollama**, also local. Nothing is sent to the cloud.

---

## What you need (overview)

| Component | Why | Approx size |
|-----------|-----|-------------|
| **Python 3.11+** | runs the web server | ~30 MB |
| **Ollama** | runs the local AI model | ~1 GB app |
| **qwen3:8b** model | the AI the assistant uses | ~5 GB download |
| The FIDUCIA project files | the app itself | small |

**Hardware:** the 8B model wants **~8 GB of free RAM** to run comfortably. It works
on Apple Silicon Macs and on Windows PCs (a GPU helps but is not required — it's
just slower on CPU).

---

## macOS

### 1. Install Homebrew (package manager)
Open **Terminal** (Cmd+Space, type "Terminal") and paste:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Follow the prompts. When it finishes, close and reopen Terminal.

### 2. Install Python
```bash
brew install python@3.11
python3 --version        # should print 3.11.x or higher
```

### 3. Install Ollama (the local AI runtime)
```bash
brew install ollama
```
Then start it (leave this running in its own Terminal window/tab):
```bash
ollama serve
```

### 4. Download the AI model
In a **new** Terminal tab (Cmd+T):
```bash
ollama pull qwen3:8b
```
This downloads ~5 GB once. Test it:
```bash
ollama run qwen3:8b "say hello"
```

### 5. Get the project and set it up
```bash
cd ~/Documents           # or wherever you keep projects
# if you have the folder already, just: cd fiducia
cd fiducia

python3 -m venv .venv         # create an isolated environment
source .venv/bin/activate     # activate it (prompt now shows (.venv))
pip install -r requirements.txt
```

### 6. Run it
```bash
uvicorn main:app --port 8000
```
Open **http://localhost:8000** in your browser. Done.

---

## Windows

### 1. Install Python
- Go to <https://www.python.org/downloads/> and download **Python 3.11 or newer**.
- Run the installer. **Tick "Add python.exe to PATH"** at the bottom before clicking Install.
- Open **PowerShell** (Start menu → type "PowerShell") and check:
```powershell
python --version        # should print 3.11.x or higher
```

### 2. Install Ollama (the local AI runtime)
- Download from <https://ollama.com/download/windows> and run the installer.
- Ollama starts automatically and runs in the background (look for its icon in the
  system tray). No separate "serve" command is needed.

### 3. Download the AI model
In **PowerShell**:
```powershell
ollama pull qwen3:8b
```
This downloads ~5 GB once. Test it:
```powershell
ollama run qwen3:8b "say hello"
```

### 4. Get the project and set it up
```powershell
cd $HOME\Documents          # or wherever you keep projects
cd fiducia

python -m venv .venv                 # create an isolated environment
.\.venv\Scripts\Activate.ps1         # activate it (prompt now shows (.venv))
pip install -r requirements.txt
```
> If activation is blocked by an execution-policy error, run this once, then retry:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

### 5. Run it
```powershell
uvicorn main:app --port 8000
```
Open **http://localhost:8000** in your browser. Done.

---

## First run — what to expect

- On startup the app **preloads the model** so the first message is fast. This can
  take ~10–15 seconds the very first time. If Ollama isn't running yet, the app
  still starts — it just warms up on the first message instead.
- A database file (`fiducia.db`) is **created automatically** in the project folder
  on first run. You don't need to set up any database.

## Everyday use (after the first setup)

You only need two things running:

1. **Ollama** — already running in the background (Windows) or `ollama serve` (macOS).
2. **The app** — from the project folder, each time:
   ```bash
   # macOS
   source .venv/bin/activate
   uvicorn main:app --port 8000
   ```
   ```powershell
   # Windows
   .\.venv\Scripts\Activate.ps1
   uvicorn main:app --port 8000
   ```

Stop the server with **Ctrl+C**.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Browser shows "Network error" / can't connect | The server isn't running. Start it with `uvicorn main:app --port 8000`. |
| Assistant reply errors with "Ollama unavailable" | Ollama isn't running or the model isn't pulled. Run `ollama serve` (macOS) and `ollama pull qwen3:8b`. |
| `command not found: python3` / `python` not recognised | Python isn't installed or not on PATH. Reinstall (Windows: re-tick "Add to PATH"). |
| `pip install` fails | Make sure the virtual environment is **activated** (you should see `(.venv)` in the prompt). |
| Port 8000 already in use | Run on another port: `uvicorn main:app --port 8001`, then open `http://localhost:8001`. |
| First AI reply is very slow | Normal on first use / CPU-only machines — the model is loading into memory. It's faster after that (kept warm for 30 minutes). |
| Model too slow / not enough RAM | Try a smaller model: `ollama pull qwen3:4b`, then change `MODEL = "qwen3:8b"` to `"qwen3:4b"` in `conversation.py`. |

---

## Notes for reviewers

- **Everything is local.** No API keys, no accounts, no external services. The model
  runs via Ollama on the same machine; the score is a fixed formula in `scoring.py`.
- **Model choice** lives in one place: `MODEL` at the top of `conversation.py`.
- **Reset the data** at any time by deleting `fiducia.db` (it's recreated on next run).
