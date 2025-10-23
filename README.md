# Ultimate Generative AI

**Mastering Models from GANs to LLMs** — A practical guide to Generative AI

This repository contains all code examples, notebooks, and chapter folders from the book *Ultimate Generative AI* by Praveen Kumar. The book guides readers from **foundational concepts** to **advanced generative models**, emphasizing hands-on coding, real-world applications, and ethical AI use.

For a detailed introduction and author background, see [ABOUT_BOOK.md](./ABOUT_BOOK.md).

---

## 📂 Repository Structure

```
Ultimate-Generative-AI/
├─ Chapter_01_Introduction/
├─ Chapter_02_Math_Foundations/
├─ Chapter_03_VAEs/
├─ Chapter_04_GANs/
├─ ... (Chapters 05 to 21)
├─ ultimate_repo_setup.sh      # Cross-platform setup script
├─ .gitignore
├─ requirements.txt
├─ README.md
├─ ABOUT_BOOK.md
```

- Each `Chapter_XX_*` folder contains code examples and explanations for that chapter.  
- `ultimate_repo_setup.sh` is a **single script** for setting up your environment on **Mac, Linux, or Windows**.

---

## ⚙️ Setup Instructions

### 1. Open Terminal (Mac/Linux) or Git Bash / PowerShell (Windows)

### 2. Make the script executable (Mac/Linux)
```bash
chmod +x ultimate_repo_setup.sh
```

### 3. Run the script
```bash
./ultimate_repo_setup.sh
```

**What it does:**

- Checks Python version (>= 3.10)  
- Detects Conda or venv environment  
- Creates and activates a virtual environment if needed  
- Installs all required Python packages from `requirements.txt`  
- Updates `.gitignore` and `README.md`  
- Commits and pushes changes to GitHub  

**Windows Users Tip:**  
Use Git Bash or PowerShell. If `python` is not recognized, ensure Python is installed and added to your PATH: [Python Downloads](https://www.python.org/downloads/windows/)

---

## 📚 Quick Book Overview

The book covers:

- **Foundations & Math:** Probability, statistics, and optimization for generative modeling  
- **Core Models:** VAEs, GANs, Diffusion Models  
- **Advanced Models:** DCGANs, Conditional GANs, CycleGANs, StyleGANs, β-VAE, CVAE  
- **Applications:** NLP, data augmentation, image/text generation, real-world deployment  
- **Large Language Models & GPT:** LLM integration, LangChain, RAG, Agentic AI  
- **Best Practices:** Evaluation, optimization, ethical AI, deployment tips  

Each chapter includes **hands-on Python examples** using TensorFlow and PyTorch, helping readers implement and experiment with generative models.

---

## 📖 Chapter Navigation

Click to explore each chapter folder:

- [Chapter 01 Introduction to Generative Models](./Chapter_01_Introduction_to_Generative_Models)
- [Chapter 02 Mathematical Foundations](./Chapter_02_Mathematical_Foundations)
- [Chapter 03 Introduction to VAEs](./Chapter_03_Introduction_to_VAEs)
- [Chapter 04 Introduction to GANs](./Chapter_04_Introduction_to_GANs)
- [Chapter 05 Deep Convolutional GANs](./Chapter_05_Deep_Convolutional_GANs)
- [Chapter 06 Conditional GANs](./Chapter_06_Conditional_GANs)
- [Chapter 07 CycleGANs](./Chapter_07_CycleGANs)
- [Chapter 08 StyleGANs](./Chapter_08_StyleGANs)
- [Chapter 09 Advanced VAEs](./Chapter_09_Advanced_VAEs)
- [Chapter 10 Diffusion Models](./Chapter_10_Diffusion_Models)
- [Chapter 11 Data Augmentation](./Chapter_11_Data_Augmentation)
- [Chapter 12 Generative Models in NLP](./Chapter_12_Generative_Models_in_NLP)
- [Chapter 13 Model Evaluation and Optimization](./Chapter_13_Model_Evaluation_and_Optimization)
- [Chapter 14 Deployment](./Chapter_14_Deployment)
- [Chapter 15 Ethics and Future Directions](./Chapter_15_Ethics_and_Future_Directions)
- [Chapter 16 Introduction to LLMs](./Chapter_16_Introduction_to_LLMs)
- [Chapter 17 GPT](./Chapter_17_GPT)
- [Chapter 18 LangChain Applications](./Chapter_18_LangChain_Applications)
- [Chapter 19 Prompt Engineering RAG Fine Tuning](./Chapter_19_Prompt_Engineering_RAG_Fine_Tuning)
- [Chapter 20 Advanced Concepts](./Chapter_20_Advanced_Concepts)
- [Chapter 21 Best Practices](./Chapter_21_Best_Practices)

---

## 🤝 Contributing

Pull requests welcome. Discuss major changes before submitting.

---

## 📜 License

MIT License © 2025 Praveen Kumar