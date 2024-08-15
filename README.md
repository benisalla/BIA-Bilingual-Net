<div align="center">
  <img src="https://github.com/user-attachments/assets/9b132fc8-bf47-4447-923c-06ada44c9985" width="200" height="200"/>
  <h1>BIALinguaNet: Transformer-Based Language Translation Architecture</h1>
  <p>A custom implementation of a language translation model using Transformer architecture, enhanced with some fancy modifications. 😁</p>
</div>


---

## Table of Contents
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
  <h3>What A demo Video</h3>
  <a href="https://youtu.be/tUHztFYTZ2U" target="_blank">
    <img src="https://img.youtube.com/vi/tUHztFYTZ2U/0.jpg" width="600" alt="YouTube Video Demo"/>
  </a>
  <p><em>Click the image above to watch the video on YouTube.</em></p>
</div>


BIALinguaNet uses powerful Transformer architectures to provide a reliable language translation system. Designed with simplicity and effectiveness in mind, this project makes it easier to develop neural networks for natural language processing, suitable for both academic research and practical use.

The model is versatile, allowing adjustments to meet various language and translation needs. By changing the transformer's input settings and the formats of the data being translated, users can customize the system to fit their specific requirements.


---

## Features

- **Bilingual Translation**: Supports translation between English and Darija, allowing users to switch seamlessly between the two languages.
- **Adjustable Model Settings**: Users can fine-tune the translation output by adjusting parameters such as temperature, top-k sampling, top-p sampling, beam size, and length normalization coefficient.
- **Real-time Translation**: Provides immediate translation results with an interactive interface that displays both the input and output texts.
- **Multiple Hypotheses Display**: Shows alternative translation hypotheses with associated confidence scores, giving users a range of translation options.
- **User-Friendly Interface**: The app features a visually appealing and intuitive design, making it easy for users to interact with the translation tool.
- **Customizable Translation Settings**: Allows users to control the translation process in detail, tailoring the output to their specific needs.
---

## Project Structure
```
BIA-BILINGUAL-NET
│
├── translation_engine
│   ├── app               # Contains the application-specific scripts and styles
│   │   ├── style         # Directory for styling components used in the application
│   │   │   ├── style.css     # CSS file to define the visual appearance of the app
│   │   ├── HTML5.py      # Script for handling HTML5 elements within the app
│   │   ├── main.py       # Main application script, likely responsible for launching the app
│   │   ├── utils.py      # Utility functions that support the app's operations
│   │
│   ├── core              # Core functionalities such as configurations and shared utilities
│   │   ├── config.py     # Configuration file for setting up parameters and options
│   │
│   ├── data              # Data handling scripts, data preprocessing, and resource management
│   │   ├── dataloader.py # Script responsible for loading and managing datasets
│   │
│   ├── model             # Model definition files, including neural network architectures
│   │   ├── BIALinguaNet.py   # Main architecture of the BIA LinguaNet model
│   │   ├── Decoder.py        # Script defining the decoder part of the neural network
│   │   ├── Encoder.py        # Script defining the encoder part of the neural network
│   │   ├── LSRNetwork.py     # Script for Label Smoothing Cross-Entropy, a loss function
│   │   ├── ResFFNet.py       # Script for a Residual FeedForward Network used in the model
│   │   ├── ResMHAtten.py     # Script for a Residual Multi-Head Attention mechanism
│   │   ├── utils.py          # Utility functions supporting model-related tasks
│
├── src               # Source files for the main application logic
│   ├── checkpoints   # Directory to store model checkpoints during training
│   │   ├── dr_en_chpts.pth.tar  # Checkpoint file for the Darija to English model
│   │   ├── en_dr_chpts.pth.tar  # Checkpoint file for the English to Darija model
│   │
│   ├── dataset       # Scripts and modules for dataset preparation and loading
│   │   ├── train.json  # Training data in JSON format
│   │   ├── val.json    # Validation data in JSON format
│   │
│   ├── img           # Storage for any images used within the project
│
│   ├── tests         # Test scripts for unit testing and model validation
│
├── finetune.py           # Script for fine-tuning the model on new or additional data
├── train.py              # Main training script for training the models
├── .gitignore            # Specifies files and directories for Git to ignore
├── poetry.lock           # Lock file for dependency management by Poetry
├── pyproject.toml        # Project metadata and configuration settings for Poetry
└── README.md             # Comprehensive project overview and setup instructions

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
poetry run streamlit run translation_engine/app/main.py
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

Below are demonstrations of translations performed by the BIALinguaNet model, showcasing its capabilities for both English to Darija (En to Dr) and Darija to English (Dr to En) translations.

#### Scripts Used for Generating Examples:

```
def translate_all(sentences, model, tokenizer):

    translations = []
    for sentence in sentences:
        best_hypo, _ = model.translate(
            sx=sentence,
            tokenizer=tokenizer,
            temperature=1.0,  
            beam_size=4,
            len_norm_coeff=0.6,
            top_k=50,
            top_p=0.95,
            is_ltr=False,
            max_beam_fork=128
        )
        translations.append((sentence, best_hypo))
    return translations

def display_translations(translated_pairs):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

    print(HEADER + BOLD + "Translations" + ENDC)
    for darija, english in translated_pairs:
        print(f"{OKBLUE}Darija:{ENDC} {darija.ljust(30)} {OKGREEN}English:{ENDC} {english}")

```

#### Visual Examples:
The following images provide visual confirmations of the translations facilitated by BIALinguaNet:

**From Darija to English:**
![Darija to English Translation Example](https://github.com/user-attachments/assets/d2ea08ac-e727-4694-871e-aa168d73a70e)

**From English to Darija:**
![English to Darija Translation Example](https://github.com/user-attachments/assets/ebee4b4c-d64c-4691-b151-571640b4c0ff)


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
