# Getting Started with Ultimate Generative AI

Welcome to the **Ultimate Generative AI** repository! This guide helps you set up your environment, run code examples from all 21 chapters, and experiment with generative models effectively.

---

## 1️⃣ Prerequisites

- **Python 3.10 or higher**
- **Git** installed
- Optional: **NVIDIA GPU + CUDA** for faster image generation
- Internet connection (for downloading pretrained models)

---

## 2️⃣ Clone the Repository

```bash
git clone https://github.com/prvnktech/Ultimate-Generative-AI.git
cd Ultimate-Generative-AI
```

---

## 3️⃣ Set Up the Environment

### **Option A: Using the Setup Script (Recommended)**

```bash
# Make the script executable (Mac/Linux)
chmod +x ultimate_repo_setup.sh

# Run the setup script
./ultimate_repo_setup.sh
```

This script will:

- Create a Python virtual environment (`venv`)
- Install all required packages from `requirements.txt`
- Update `.gitignore` and `README.md`
- Prepare your system for notebooks and CLI execution

---

### **Option B: Manual Setup**

```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate      # Mac/Linux
# venv\Scripts\activate       # Windows PowerShell

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

---

## 4️⃣ Running Jupyter Notebooks

Each chapter includes **Jupyter notebooks** for interactive learning.

```bash
# Launch Jupyter Notebook
jupyter notebook

# or Jupyter Lab
jupyter lab
```

- Navigate to the chapter folder (e.g., `Chapter_01_Introduction/notebooks`)
- Open the notebook and run cells step by step
- Visualize generated images inline

---

## 5️⃣ Running Python Scripts (CLI)

Each chapter also includes **Python scripts** for command-line execution.

Example (Chapter 1 - BigGAN Car Generator):

```bash
python Chapter_01_Introduction/scripts/chapter_01_biggan_car.py --class_name sports_car --output outputs/sports_car.png
```

- `--class_name`: Select vehicle class (e.g., sports_car, taxi, bus, race_car)  
- `--output`: Specify output file path

---

## 6️⃣ Running Streamlit Apps (Optional)

Some chapters include **interactive Streamlit apps**:

```bash
# Launch Streamlit app
streamlit run Chapter_01_Introduction/scripts/chapter_01_streamlit_app.py
```

- Use dropdowns and buttons to generate images interactively  
- Output is displayed in the browser  

---

## 7️⃣ Output Organization

- Generated images, plots, and results are stored in `outputs/` folders within each chapter  
- Keep these folders clean to avoid clutter

Example:

```
Chapter_01_Introduction/
├─ notebooks/
├─ scripts/
├─ outputs/
│  ├─ sports_car.png
│  ├─ taxi.png
```

---

## 8️⃣ Chapter Navigation

Each chapter folder is self-contained:

- **notebooks/**: Interactive examples  
- **scripts/**: CLI scripts or Streamlit apps  
- **outputs/**: Generated images or results  
- **README.md**: Chapter-specific instructions

Chapters include:

1. Introduction to Generative Models  
2. Mathematical Foundations  
3. Variational Autoencoders (VAEs)  
4. Generative Adversarial Networks (GANs)  
5. Deep Convolutional GANs (DCGANs)  
6. Conditional GANs (cGANs)  
7. CycleGANs  
8. StyleGANs  
9. Advanced VAEs (β-VAE, CVAE)  
10. Diffusion Models  
11. Data Augmentation with Generative Models  
12. Generative Models in NLP  
13. Model Evaluation and Optimization  
14. Deployment of Generative Models  
15. Ethical Considerations and Future Directions  
16. Introduction to Large Language Models (LLMs)  
17. Generative Pre-trained Transformers (GPT)  
18. Building AI-Powered Applications with LangChain  
19. Prompt Engineering, RAG, and Fine-Tuning  
20. Advanced Concepts (MCP, Agentic AI, Tools)  
21. Best Practices for Working with Generative Models  

---

## 9️⃣ Tips for Smooth Execution

- **GPU acceleration:** BigGAN, StyleGAN, and Diffusion models are faster with a GPU  
- **Experiment with latent vectors:** For GANs/VAEs, try changing the random `z` vector to generate different outputs  
- **Keep code updated:** Pull updates from the repo periodically  
- **Use separate virtual environments** for each major setup if combining multiple libraries

---

## 10️⃣ Support and Issues

- For bugs, issues, or questions, please open an issue on [GitHub Issues](https://github.com/prvnktech/Ultimate-Generative-AI/issues)  
- Include OS, Python version, and any error messages

---

## 🎯 Goal

This guide ensures that **learners and developers** can:

- Quickly set up their environment  
- Run **interactive notebooks**  
- Experiment with **scripts and Streamlit apps**  
- Explore **all 21 chapters** with minimal friction  

Happy learning and generating! 🚀