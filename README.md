<div align="center">
  <img src="https://github.com/user-attachments/assets/9b132fc8-bf47-4447-923c-06ada44c9985" width="200" height="200"/>
  <h1>BIALinguaNet: Transformer-based Language Translation Architecture</h1>
  <p>Implementing a language translation model using Transformer technology from scratch.</p>
</div>


---

## Table of Contents 📘
- [About The Project](#about-the-project)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Training](#training)
- [Fine-Tuning](#fine-tuning)
- [Translation Examples](#translation-examples)
- [License](#license)
- [About Me](#about-me)

---

## About The Project

<div align="center">
  <h3>Interface of our app</h3>
  <img src="https://github.com/benisalla/BIALinguaNet/assets/interface.png" width="600" height="300"/>
</div>

BIALinguaNet offers a focused implementation of language translation using advanced Transformer architectures. This project is designed to provide an intuitive, effective approach to neural network development for natural language processing, suitable for both research and practical applications.

The model can be adapted for various languages and translation types by modifying the transformer's input configurations.

---

## Features

- **Modular Design**: Clear separation of components such as data processing, model architecture, and training scripts.
- **Interactive Translation Demo**: Enables real-time testing of translation capabilities.
- **Download Capability**: Allows users to download translated texts.
- **Customizable**: Easily adapt the architecture and data pipelines for different language pairs.
- **Poetry for Dependency Management**: Utilizes Poetry for straightforward and dependable package management.

---

## Project Structure
```
BIALinguaNet
│
├── translated_texts
├── translation_architecture
│   ├── app
│   ├── core
│   ├── data
│   ├── model
│   │   ├── Transformer
│   └── src
│       ├── checkpoints
│       ├── dataset
│       └── tokenizer
├── tests
│   ├── model
│   └── tokenizer
├── tokenizing
│   └── tokenizer
│       ├── __init__.py
│       ├── AdvancedTokenizer.py
│       ├── BIALTokenizer.py
│       ├── train_advanced_tokenizer.py
│       └── train_bia_tokenizer.py
├── finetune.py
├── train.py
├── .gitignore
└── README.md
```

---

## Getting Started

Follow these simple steps to get a local copy up and running.

### Installation

1. Clone the repository
   ```sh
   git clone https://github.com/benisalla/BIALinguaNet.git
   ```
2. Install dependencies using Poetry
   ```sh
   poetry install
   ```
3. Activate the Poetry shell to set up your virtual environment
   ```sh
   poetry shell
   ```

### Running the Application

To launch the application, execute the following command:

```sh
poetry run streamlit run translation_architecture/app/main.py
```

### Training

#### How to Run Training

To train the model using the default configuration, execute the following command:

```sh
poetry run python train.py
```

### Fine-Tuning

To fine-tune a pre-trained model:

```sh
poetry run python finetune.py
```

---

### Translation Examples

Here are some examples of translations performed by the BIALinguaNet model:

[Translation Example Links]

---

## License

This project is made available under **fair use guidelines**. While there is no formal license associated with the repository, users are encouraged to credit the source if they utilize or adapt the code in their work. This approach promotes ethical practices and contributions to the open-source community. For citation purposes, please use the following:

```bibtex
@misc{bialinguanet_2024,
  title={BIALinguaNet: Transformer-based Language Translation Architecture},
  author={Ben Alla Ismail},
  year={2024},
  url={https://github.com/benisalla/BIALinguaNet}
}
```

---

## About Me

🎓 **Ismail Ben Alla** - AI and NLP Enthusiast

As a dedicated advocate for artificial intelligence and natural language processing, I am deeply committed to exploring their potential to bridge language barriers and connect cultures. My academic and professional pursuits reflect a relentless dedication to advancing knowledge in AI, deep learning, and machine learning technologies.

### Core Motivations
- **Innovation in AI and Translation**: Driven to expand the frontiers of technology and unlock novel insights across languages.
- **Lifelong Learning**: Actively engaged in mastering the latest technological developments.
- **Future-Oriented Vision**: Fueled by the transformative potential and future prospects of AI and language technology.

I am profoundly passionate about my work and optimistic about the future contributions of AI in language processing.

**Let's connect and explore the vast potential of artificial intelligence and translation together!**
<div align="center">
  <a href="https://twitter.com/ismail_ben_alla" target="_blank">
    <img src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/twitter.svg" alt="ismail_ben_alla" height="60" width="60" />
  </a>
  
  <a href="https://linkedin.com/in/ismail-ben-alla-7144b5221/" target="_blank">
    <img src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="LinkedIn Profile" height="60" width="60"/>
  </a>
  
  <a href="https://instagram.com/ismail_ben_alla" target="_blank">
    <img src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/instagram.svg" alt="Instagram Profile" height="60" width="60" />
  </a>
</div>
