# orion-research
research for orion

## Installation Instructions

1. Clone the repository:
   ```zsh
   git clone https://github.com/riddhimanrana/orion-research
   ```
2. If you haven't already, install [brew](https://brew.sh/):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
3. Navigate to the project directory:
    ```zsh
    cd orion-research
    ```
4. Install conda:
    ```zsh
    brew install conda
    ```
5. Setup the conda environment:
    ```zsh
    conda create --name orion python=3.10
    conda activate orion
    pip -r requirements.txt
    ```
## Other

You can use `conda deactivate` to exit the conda environment
If you install any other new python dependencies, run `pip freeze > requirements.txt` to update the requirements file
