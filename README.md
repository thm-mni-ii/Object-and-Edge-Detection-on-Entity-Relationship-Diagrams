# Object and Edge Detection on Entity-Relationship Diagrams

This repository contains the code for the paper "Object and Edge Detection on Entity-Relationship Diagrams."

## Installation

Follow these steps to set up the project:

1. **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd <repository-folder>
    ```

2. **Install Poetry:**
    Follow the instructions provided in the [Poetry documentation](https://python-poetry.org/docs/#installing-with-the-official-installer).

3. **Install project dependencies:**
    ```sh
    poetry install
    ```

4. **Add the weights for YOLO, SAM and TrOCR:**

    - Create a `weights` folder in the top level of the project directory:
        ```sh
        mkdir weights
        ```

    - Download the YOLO weights from [here](https://mndthmde-my.sharepoint.com/:u:/g/personal/maximilian_huber_mnd_thm_de/ERuYDWjopadGmWYJ4NuZqbIBZT8v7sd2ideJJ9ZzmLQcsw?e=fZFiv6) and place them in the `weights` folder.

    - Download the SAM weights from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and place them in the `weights` folder.

    - Download the TrOCR weights from [here](https://drive.google.com/file/d/1t2MjHlfbkZOdNotmmQb-XeJyHOMRSKpH/view?usp=sharing) and place them in the `weights` folder.

### Automatic Installation for Linux/Unix

To set up the project automatically, including creating a virtual environment, installing `poetry`, downloading all required dependencies, and fetching the necessary weights, run the `setup.sh` script with the following commands:

1. **Make the script executable** (if needed):
    ```sh
    chmod +x setup.sh
    ```

2. **Run the script**:
    ```sh
    source setup.sh
    ```

This will handle everything, including setting up the environment and downloading required files, so the project is ready to use.

## Running the Code

Before executing the code, update the file `erd_detection/main.py` at lines 24, 31 and 41 with the appropriate paths. (This step is not required if you use setup.sh)

To execute the code, use the following command in the project directory:

```sh
poetry run python erd_detection/main.py
```

Own handwritten Entity-Relationship diagrams can be tested by adding other images to the `data` folder.
