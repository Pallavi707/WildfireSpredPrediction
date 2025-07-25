# Wildfire Spread Prediction with Cellular Automata

This repository implements an end-to-end pipeline for simulating and predicting wildfire spread using Cellular Automata (CA), a grid-based modeling method well-suited for spatial-temporal environmental phenomena.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
- [Usage](#usage)
  - [Running Simulations](#running-simulations)
  - [Evaluation and Visualization](#evaluation-and-visualization)
- [Cellular Automata Model](#cellular-automata-model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

This project simulates and predicts wildfire spread using Cellular Automata (CA). CA models represent landscapes as grids, where each cell updates its state (e.g., unburned, burning, burned) based on local rules and the states of neighboring cells, making them ideal for modeling wildfire dynamics.

---

## Project Structure

```
Wildfire-Spread-Prediction/
│
├── data/                  # Raw and processed data
├── notebooks/             # Jupyter notebooks for EDA, visualization, experiments
├── src/                   # Source code
│   ├── ca/                # Cellular Automata model implementations
│   │   ├── __init__.py
│   │   └── wildfire_ca.py # Main CA wildfire spread logic
│   ├── data_preprocessing.py
│   ├── train.py           # For parameter fitting or calibration (if any)
│   ├── simulation.py      # Running CA-based wildfire simulations
│   ├── evaluation.py      # Metrics and visualization
│   └── utils.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Getting Started

### Data Preparation

### Data Preparation

> **Note:** The dataset is not included in this repository due to size limitations.

This project uses the [Next Day Wildfire Spread dataset](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread) available on Kaggle. Please follow the steps below to download and prepare the data:

#### 1. Download the Dataset
- Visit the dataset page:  
  https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread
- Download the `archive.zip` file and **extract it into your project directory**.

After extraction, your folder should contain:

archive/
│── next_day_wildfire_spread_train.tfrecord
│── next_day_wildfire_spread_test.tfrecord
│── next_day_wildfire_spread_eval.tfrecord

#### 2. Convert TFRecord Files to Pickle

Run the provided script to convert the `.tfrecord` files into `.pkl` format used for training:

```bash
python convert_tfrecord_to_pickle.py

This will generate the following files in the data/next-day-wildfire-spread/ folder:

data/
└── next-day-wildfire-spread/
    ├── train.data
    ├── train.labels
    ├── test.data
    ├── test.labels
    ├── validation.data
    └── validation.labels

These .pkl files are used by training, evaluation, and simulation scripts.


### Installation

```bash
git clone https://github.com/<your-username>/Wildfire-Spread-Prediction.git
cd Wildfire-Spread-Prediction
pip install -r requirements.txt
```

### Data Preparation

- Place your landscape/environmental data in `data/raw/`.
- Run preprocessing (if needed):
  ```bash
  python src/data_preprocessing.py
  ```

---

## Usage

### Running Simulations

Run a wildfire spread simulation using CA:
```bash
python src/simulation.py --config configs/simulation_config.yaml
```

### Evaluation and Visualization

Evaluate and visualize the results:
```bash
python src/evaluation.py --input results/simulation_output.npy
```

---

## Cellular Automata Model

- **Wildfire CA Rules:** Each cell updates its state based on neighboring cells, wind, fuel, moisture, and possibly stochasticity.
- **Configurable Parameters:** Grid size, ignition probability, neighborhood type (Moore/von Neumann), environmental factors, etc.
- **Extensible:** You can easily adapt rules or add new environmental drivers.

---

## Results

*Update this section with sample outputs, accuracy/fit to real data, or visualizations.*

---

## Contributing

Contributions are welcome! Please open an issue or pull request.

---

## License

MIT License

---

## Acknowledgements

- [Cellular Automata Theory](https://en.wikipedia.org/wiki/Cellular_automaton)
- [Wildfire Modeling Literature]
- Any datasets or tools you use
