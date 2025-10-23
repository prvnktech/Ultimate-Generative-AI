#!/bin/bash
# ============================================================
# Ultimate Generative AI — Full Cross-Platform Reader-Friendly Setup
# Author: prvnktech
# ============================================================

# -------------------- COLORS --------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# -------------------- VARIABLES --------------------
REPO_NAME="Ultimate-Generative-AI"
REPO_SSH="git@github.com:prvnktech/Ultimate-Generative-AI.git"

# -------------------- SSH CHECK --------------------
echo -e "${CYAN}🔍 Checking SSH access...${NC}"
ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ SSH authentication failed.${NC}"
    echo -e "Please add your key with:"
    echo -e "  ${BOLD}eval \"\$(ssh-agent -s)\"${NC}"
    echo -e "  ${BOLD}ssh-add /path/to/githubprvnktechkey${NC}"
    exit 1
else
    echo -e "${GREEN}✅ SSH authentication verified!${NC}"
fi

# -------------------- PYTHON VERSION CHECK --------------------
PYTHON_CMD=$(command -v python3 || command -v python)
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION=3.10

python_version_ok=$(echo "$PYTHON_VERSION >= $REQUIRED_VERSION" | bc)
if [ "$python_version_ok" -eq 0 ]; then
    echo -e "${RED}⚠️  Python $PYTHON_VERSION detected. Python >= 3.10 is required.${NC}"
    echo -e "${YELLOW}Please install Python 3.10+ manually:${NC}"
    echo -e "  Mac/Linux: https://www.python.org/downloads/"
    echo -e "  Windows: https://www.python.org/downloads/windows/"
    exit 1
else
    echo -e "${GREEN}✅ Python $PYTHON_VERSION detected (>= 3.10)${NC}"
fi

# -------------------- CLONE REPO --------------------
if [ ! -d "$REPO_NAME" ]; then
    echo -e "${BLUE}📦 Cloning repository...${NC}"
    git clone "$REPO_SSH"
fi
cd "$REPO_NAME" || exit
git remote set-url origin "$REPO_SSH"

# -------------------- .GITIGNORE --------------------
if [ ! -f ".gitignore" ]; then
    echo -e "${BLUE}🧹 Creating .gitignore...${NC}"
    cat <<EOL > .gitignore
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.so
*.egg-info/
.env
.venv/
env/
venv/
.ipynb_checkpoints/
.DS_Store
.idea/
.vscode/
*.log
EOL
    echo -e "${GREEN}✅ .gitignore created${NC}"
else
    echo -e "${YELLOW}ℹ️  .gitignore exists — skipping${NC}"
fi

# -------------------- REQUIREMENTS.TXT --------------------
if [ ! -f "requirements.txt" ]; then
    echo -e "${BLUE}📦 Creating requirements.txt...${NC}"
    cat <<EOL > requirements.txt
torch
torchvision
tensorflow
keras
numpy
pandas
scikit-learn
matplotlib
seaborn
opencv-python
pillow
diffusers
transformers
datasets
accelerate
langchain
langchain-core
langchain-community
openai
faiss-cpu
tqdm
notebook
EOL
    echo -e "${GREEN}✅ requirements.txt created${NC}"
else
    echo -e "${YELLOW}ℹ️  requirements.txt exists — skipping${NC}"
fi

# -------------------- README.MD --------------------
echo -e "${BLUE}📝 Updating README.md...${NC}"
cat <<EOL > README.md
# Ultimate Generative AI

This repository contains code and resources for the book **Ultimate Generative AI**, authored by **Praveen (prvnktech)**.

## 📚 Table of Contents
EOL

for d in $(ls -d Chapter_* | sort); do
    echo "- [${d//_/ }](./$d)" >> README.md
done

cat <<EOL >> README.md

## ⚙️ Setup Instructions

### Conda (recommended if installed)
\`\`\`bash
conda create -n ultimate_gen_ai python=3.11
conda activate ultimate_gen_ai
pip install -r requirements.txt
\`\`\`

### venv (fallback)
Mac/Linux:
\`\`\`bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\`

Windows:
\`\`\`cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
\`\`\`

${BOLD}${YELLOW}💡 Tip for Windows users:${NC} Use Git Bash or PowerShell to run commands. If 'python' is not recognized, check your PATH or install Python from https://www.python.org/downloads/windows/${NC}

## 🤝 Contributing
Pull requests welcome. Discuss major changes first.

## 📜 License
MIT License © 2025 Praveen (prvnktech)
EOL

# -------------------- ENV DETECTION & INSTALL --------------------
echo -e "${CYAN}🐍 Detecting environment...${NC}"
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${GREEN}ℹ️  Conda environment detected: $CONDA_DEFAULT_ENV${NC}"
    pip install --upgrade pip
    pip install -r requirements.txt
elif [ -n "$VIRTUAL_ENV" ]; then
    echo -e "${GREEN}ℹ️  venv environment detected: $VIRTUAL_ENV${NC}"
    pip install --upgrade pip
    pip install -r requirements.txt
else
    if [ ! -d "venv" ]; then
        echo -e "${BLUE}ℹ️  Creating venv...${NC}"
        $PYTHON_CMD -m venv venv
    fi
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo -e "${BLUE}📥 Activating venv on Windows...${NC}"
        source venv/Scripts/activate
    else
        echo -e "${BLUE}📥 Activating venv on Mac/Linux...${NC}"
        source venv/bin/activate
    fi
    pip install --upgrade pip
    pip install -r requirements.txt
    deactivate
fi

# -------------------- COMMIT & PUSH --------------------
echo -e "${CYAN}📤 Committing and pushing updates...${NC}"
git add .gitignore requirements.txt README.md
git commit -m "Reader-friendly cross-platform setup: gitignore, requirements, README, environment instructions"
git push origin main

echo -e "${GREEN}✅ All done!${NC}"
echo -e "${CYAN}🎉 Repo ready at: https://github.com/prvnktech/Ultimate-Generative-AI${NC}"
