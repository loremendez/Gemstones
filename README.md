# Gemstones
Deep CNN with residual blocks to classify Gemstones!


<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
-->

<!-- PROJECT LOGO -->
<br />
<p align="center">

<h3 align="center">Gemstone classification using deep CNNs with residual blocks</h3>

<!-- ![pngwing com (7)](https://user-images.githubusercontent.com/62608007/123555126-14d39e80-d784-11eb-870b-4ad3350e894a.png = 250x250)
-->

<a href="https://github.com/loremendez/Gemstones">
    <p align="center">
        <img src="https://user-images.githubusercontent.com/62608007/123556875-f8d4fa80-d78d-11eb-896c-3f4cb1f68f28.png" width="145">
        <img src="https://user-images.githubusercontent.com/62608007/123556874-f672a080-d78d-11eb-925d-7aac33aa276e.png" width="145">
        <img src="https://user-images.githubusercontent.com/62608007/123556877-fa9ebe00-d78d-11eb-857b-0329d7b809e2.png" width="145">
    </p>
  </a>
</p>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#references">References</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

Classification of gemstones using deep convolutional neural networks with residual blocks.

### Built With

* [Anaconda 4.10.1](https://www.anaconda.com/)
* [Python 3.9](https://www.python.org/downloads/release/python-380/)
* [TensorFlow 2.5](https://www.tensorflow.org/tutorials/quickstart/beginner)

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

A running installation of Anaconda. If you haven't installed Anaconda yet, you can follow the next tutorial: <br>
[Anaconda Installation](https://docs.anaconda.com/anaconda/install/)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/loremendez/Gemstones.git
   ```
2. Install the environment <br>
    You can do it either by loading the [`YML`](https://github.com/loremendez/Gemstones/blob/main/conda_environment.yml) file
    ```sh
    conda env create -f conda_environment.yml
    ```
    or step by step
    1. Create and activate the environment
        ```sh
        conda create -n arch_style python=3.9
        conda activate arch_style
        ```
    2. Install the needed packages
        ```sh
        pip install --upgrade pip
        pip list  # show packages installed within the virtual environment

        pip install tensorflow==2.5
        pip install numpy pandas matplotlib seaborn
        pip install jupyterlab
        ```

<!-- USAGE EXAMPLES -->
## Usage

Open Jupyter-lab and open the notebook [`Gemstones.ipynb`](https://github.com/loremendez/Gemstones/blob/main/Gemstones.ipynb)
```sh
jupyter-lab
```

<!-- References -->
## References
<a id="1">[1]</a>
Dataset by Chemkaeva, Daria (@LSIND) “Gemstones”.
Last updated: 2020-04-27.
Link: [https://www.kaggle.com/lsind18/gemstones-images](https://www.kaggle.com/lsind18/gemstones-images)


<!-- CONTACT -->
## Contact

Lorena Mendez - [LinkedIn](https://www.linkedin.com/in/lorena-mendezg/?originalSubdomain=de) - lorena.mendez@tum.de

Take a look into my [other](https://github.com/loremendez) projects!
