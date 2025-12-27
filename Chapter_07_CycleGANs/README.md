# CycleGAN Architecture Lab (PyTorch)

🔄 **Interactive demonstration of CycleGAN architecture components and training logic for unpaired image-to-image translation.**

## Overview

This project implements a comprehensive CycleGAN architecture using PyTorch with an interactive Streamlit interface. CycleGAN enables unpaired image-to-image translation (e.g., Horse ↔ Zebra) by enforcing cycle consistency without requiring paired training data.

## Features

- **Complete CycleGAN Implementation**: Generator and Discriminator networks with ResNet-based architecture
- **Interactive Components**: 
  - Architecture visualization and component builder
  - Training logic demonstration with cycle consistency loss
  - Real-time translation cycle visualization
- **Professional UI**: Clean, modern interface with comprehensive styling
- **Educational Focus**: Step-by-step breakdown of CycleGAN concepts and implementation

## Architecture Components

### Generator Network
- **ResNet-based architecture** with residual blocks
- **Encoder-Decoder structure** with skip connections
- **Instance normalization** for stable training
- **Reflection padding** to reduce artifacts

### Discriminator Network
- **PatchGAN discriminator** for realistic texture generation
- **Leaky ReLU activations** for improved gradient flow
- **Instance normalization** layers

### Key Features
- **Cycle Consistency Loss**: Ensures A → B → A ≈ A
- **Adversarial Loss**: Generates realistic images in target domain
- **Identity Loss**: Preserves color composition when unnecessary

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd requirement_7.1_cyclegan_architecture_pytorch
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
streamlit run requirement_7.1_cyclegan_architecture_pytorch.py
```

## Usage

The application provides three interactive tabs:

### 1. Architecture Components
- Initialize PyTorch models (Generators and Discriminators)
- View network architecture details
- Explore ResNet-based generator structure
- Understand discriminator design

### 2. CycleGAN Training Logic
- Learn about cycle consistency loss
- Understand the complete training loop
- Explore loss functions and optimization strategies

### 3. Visualization
- Simulate translation cycles (A → B → A)
- Visualize the reconstruction process
- Understand model behavior with dummy data

## Technical Details

### Model Architecture
- **Generator**: ResNet with 9 residual blocks for 256x256 images
- **Discriminator**: 70x70 PatchGAN discriminator
- **Normalization**: Instance normalization throughout
- **Activation**: ReLU for generators, LeakyReLU for discriminators

### Training Strategy
```python
Total Loss = GAN Loss + λ × Cycle Consistency Loss + Identity Loss
```
- **λ = 10.0**: Standard weight for cycle consistency
- **Adam optimizer**: β1=0.5, β2=0.999
- **Learning rate**: Typically 0.0002 with linear decay

### Key Equations

**Cycle Consistency Loss**:
```
L_cycle(G, F) = E[||F(G(x)) - x||₁] + E[||G(F(y)) - y||₁]
```

**Adversarial Loss**:
```
L_GAN(G, D_Y, X, Y) = E[log D_Y(y)] + E[log(1 - D_Y(G(x)))]
```

## File Structure

```
requirement_7.1_cyclegan_architecture_pytorch/
├── requirement_7.1_cyclegan_architecture_pytorch.py  # Main application
├── README.md                                          # This file
├── requirements.txt                                   # Python dependencies
└── .ipynb_checkpoints/                               # Jupyter checkpoints
```

## Dependencies

- **Streamlit**: Interactive web application framework
- **PyTorch**: Deep learning framework
- **Matplotlib**: Visualization library
- **NumPy**: Numerical computing

## Educational Goals

This implementation serves as a comprehensive educational tool for understanding:

1. **GAN Architecture**: Generator-discriminator adversarial training
2. **Cycle Consistency**: How unpaired translation works
3. **ResNet Integration**: Using residual connections in generators
4. **Instance Normalization**: Why it's preferred over batch normalization in style transfer
5. **Loss Functions**: Combining adversarial, cycle, and identity losses

## References

- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- [Official CycleGAN Implementation](https://github.com/junyanz/CycleGAN)
- [PyTorch Documentation](https://pytorch.org/docs/)

## Contributing

This is an educational demonstration project. For improvements or bug fixes, please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is for educational purposes. Please refer to the original CycleGAN paper and implementation for research use guidelines.

---

**Built with**: Streamlit • PyTorch • Matplotlib

*Interactive CycleGAN Architecture Demonstration - Requirement 7.1 Implementation*