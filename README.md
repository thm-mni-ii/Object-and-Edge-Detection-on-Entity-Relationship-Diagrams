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

4. **Add the weights for YOLO and SAM:**

    - Create a `weights` folder in the top level of the project directory:
        ```sh
        mkdir weights
        ```

    - Download the YOLO weights from [here](https://<to-announce>) and place them in the `weights` folder.

    - Download the SAM weights from [this link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and place them in the `weights` folder.

## Running the Code

Before executing the code, update the file `erd_detection/main.py` at lines 22 and 29 with the appropriate paths.

To execute the code, use the following command in the project directory:

```sh
poetry run python erd_detection/main.py
