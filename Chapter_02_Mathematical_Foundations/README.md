# Information Theory Interactive Demo

An interactive demonstration of core **Information Theory** concepts using Python, Jupyter notebooks, and Streamlit. This project provides hands-on exploration of **Entropy** and **KL Divergence** through visualizations and interactive widgets.

## 📋 Overview

This project demonstrates two fundamental concepts in information theory:

1. **Shannon Entropy (H)**: Measures the uncertainty or randomness in a probability distribution
2. **Kullback-Leibler Divergence (D_KL)**: Measures how one probability distribution differs from another reference distribution

## ✨ Features

### Jupyter Notebook (`information_theory.ipynb`)
- 📚 Comprehensive explanations with mathematical formulas (LaTeX)
- 🎛️ Interactive widgets using `ipywidgets` to adjust probability distributions
- 📊 Matplotlib visualizations for static analysis
- 🧪 Multiple examples demonstrating special cases (uniform, concentrated, certain distributions)
- 🎨 Color-coded visualizations for better understanding
- 📈 Real-time entropy and KL divergence calculations

### Streamlit Web App (`information_theory_app.py`)
- 🌐 Interactive web interface accessible via browser
- 🎯 Three modes:
  - **Entropy Calculator**: Adjust probability distributions and see entropy changes
  - **KL Divergence Calculator**: Compare two distributions and visualize divergence
  - **Comparison Examples**: Explore predefined distributions with detailed analysis
- 📊 Interactive Plotly charts with zoom, pan, and hover features
- 🎲 Quick example buttons (uniform, random, concentrated distributions)
- ✅ Auto-normalization option to ensure probabilities sum to 1
- 🎨 Professional UI with color-coded interpretations
- 📱 Responsive layout that works on different screen sizes

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download the Project

```bash
cd Monday_Reqs
```

### Step 2: Install Dependencies

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

This will install:
- `numpy` - Numerical computing
- `scipy` - Scientific computing (entropy and KL divergence functions)
- `matplotlib` - Static plotting (for Jupyter)
- `plotly` - Interactive plotting (for Streamlit)
- `streamlit` - Web app framework
- `jupyter` - Jupyter notebook environment
- `ipywidgets` - Interactive widgets for Jupyter
- `pandas` - Data manipulation

### Alternative: Install Packages Individually

```bash
pip install numpy scipy matplotlib plotly streamlit jupyter ipywidgets pandas
```

## 📖 Usage

### Option 1: Jupyter Notebook (Interactive Development)

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open the notebook**:
   - Navigate to `information_theory.ipynb` in the Jupyter interface
   - Click to open the notebook

3. **Run the cells**:
   - Click "Run All" from the Cell menu, or
   - Execute cells one by one using Shift+Enter

4. **Interact with widgets**:
   - Adjust sliders to change probability distributions
   - Observe real-time updates to entropy and KL divergence values
   - Explore different examples and visualizations

### Option 2: Streamlit Web App (Production Interface)

1. **Launch the Streamlit app**:
   ```bash
   streamlit run information_theory_app.py
   ```

2. **Access the app**:
   - Your default browser will automatically open
   - If not, navigate to: `http://localhost:8501`

3. **Use the app**:
   - Select mode from sidebar (Entropy Calculator, KL Divergence Calculator, or Comparison Examples)
   - Adjust sliders to modify probability distributions
   - Explore quick examples using sidebar buttons
   - View real-time visualizations and interpretations

4. **Stop the app**:
   - Press `Ctrl+C` in the terminal

## 📊 Examples

### Example 1: Calculate Entropy

```python
import numpy as np
from scipy.stats import entropy

# Define a simple probability distribution
probability_distribution = [0.2, 0.5, 0.3]

# Calculate entropy
entropy_value = entropy(probability_distribution, base=2)
print(f"Entropy: {entropy_value}")
# Output: Entropy: 1.4854752972273344 (High Entropy)
```

**Expected Output:**
```
Entropy: 1.4854752972273344
```

### Example 2: Calculate KL Divergence

```python
import numpy as np
from scipy.special import rel_entr

# Define two probability distributions
P = [0.4, 0.6]
Q = [0.5, 0.5]

# Calculate KL Divergence = Sum of element-wise relative entropy
kl_divergence = np.sum(rel_entr(P, Q))
print(f"KL Divergence: {kl_divergence}")
# Output: KL Divergence: 0.020410407324513063 (Low divergence)
```

**Expected Output:**
```
KL Divergence: 0.020410407324513063
```

## 🎓 Concepts Explained

### Shannon Entropy

**Formula:**
```
H(X) = -∑ p(x_i) log₂ p(x_i)
```

**Properties:**
- **Range**: 0 to log₂(n) bits (for n outcomes)
- **Maximum**: Achieved with uniform distribution (all outcomes equally likely)
- **Minimum**: 0 bits (one outcome certain, probability = 1)
- **Units**: bits (base 2), nats (base e), or hartleys (base 10)

**Interpretation:**
- High entropy = high uncertainty = difficult to predict
- Low entropy = low uncertainty = easy to predict
- Used in: Data compression, decision trees, feature selection

### Kullback-Leibler Divergence

**Formula:**
```
D_KL(P || Q) = ∑ P(x_i) log(P(x_i) / Q(x_i))
```

**Properties:**
- **Non-negative**: D_KL(P || Q) ≥ 0
- **Asymmetric**: D_KL(P || Q) ≠ D_KL(Q || P)
- **Zero**: D_KL(P || Q) = 0 if and only if P = Q
- **Not a distance metric**: Doesn't satisfy triangle inequality

**Interpretation:**
- Measures "extra bits" needed to encode P using code optimized for Q
- Also called "relative entropy" or "information gain"
- Used in: Machine learning (variational inference), model comparison, hypothesis testing

## 🎯 Use Cases

1. **Data Science & Machine Learning**
   - Feature selection using information gain
   - Model evaluation and comparison
   - Variational autoencoders (VAEs)
   - Cross-entropy loss functions

2. **Natural Language Processing**
   - Language model evaluation
   - Text classification
   - Perplexity calculation

3. **Statistics**
   - Hypothesis testing
   - Distribution fitting
   - Anomaly detection

4. **Communications & Coding Theory**
   - Optimal code design (Huffman coding)
   - Channel capacity calculation
   - Data compression algorithms

## 🗂️ Project Structure

```
Monday_Reqs/
│
├── information_theory.ipynb        # Jupyter notebook with interactive demos
├── information_theory_app.py       # Streamlit web application
├── requirements.txt                # Python package dependencies
├── README.md                       # This file
└── Untitled.ipynb                 # (Other project files)
```

## 🔧 Troubleshooting

### Issue: Widgets not showing in Jupyter

**Solution:**
```bash
jupyter nbextension enable --py widgetsnbextension
```

### Issue: Streamlit app not opening

**Solution:**
- Check if port 8501 is available
- Try specifying a different port:
  ```bash
  streamlit run information_theory_app.py --server.port 8502
  ```

### Issue: Import errors

**Solution:**
- Ensure all packages are installed:
  ```bash
  pip install -r requirements.txt
  ```
- Check Python version (should be 3.8+):
  ```bash
  python --version
  ```

### Issue: Matplotlib style warning

**Solution:**
- The notebook uses `seaborn-v0_8-darkgrid` style
- If you see warnings, replace with:
  ```python
  plt.style.use('seaborn-darkgrid')  # or 'default'
  ```

## 📚 Additional Resources

### Learn More About Information Theory
- [Information Theory, Inference, and Learning Algorithms](http://www.inference.org.uk/mackay/itila/) - David MacKay
- [Elements of Information Theory](https://www.wiley.com/en-us/Elements+of+Information+Theory%2C+2nd+Edition-p-9780471241959) - Cover & Thomas
- [Wikipedia: Entropy (Information Theory)](https://en.wikipedia.org/wiki/Entropy_(information_theory))
- [Wikipedia: Kullback–Leibler Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)

### Related Projects
- [Surprisal Toolkit](https://github.com/uds-lsv/surprisal-toolkit-teaching-materials)
- [IntuitionBuilder](https://github.com/phueb/IntuitionBuilder)
- [Tutorial on Entropies](https://github.com/ileanabuhan/Tutorial-on-entropies)

## 🤝 Contributing

This is a demonstration project for educational purposes. Feel free to:
- Extend the functionality
- Add more information theory concepts (mutual information, conditional entropy, etc.)
- Improve visualizations
- Add more examples

## 📝 License

This project is for educational purposes. Feel free to use and modify as needed.

## 👤 Author

**Karan**
- Project: Requirement 2.1 - Jupyter + Streamlit
- Topic: Information Theory Demonstrations

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Uses [SciPy](https://scipy.org/) for entropy calculations
- Visualizations powered by [Plotly](https://plotly.com/) and [Matplotlib](https://matplotlib.org/)
- Interactive widgets via [ipywidgets](https://ipywidgets.readthedocs.io/)

---

**Happy Exploring! 🎉**

*For questions or issues, please refer to the troubleshooting section above.*
