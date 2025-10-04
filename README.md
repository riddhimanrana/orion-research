# Orion Research
Research repository for the Orion architecture.

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
    ```bash
    mkdir -p ~/miniconda3
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    source ~/miniconda3/bin/activate
    conda init --all
    ```
5. Setup the conda environment:
    ```zsh
    conda create --name orion python=3.10
    conda activate orion
    pip install -e .
    ```
## Other

### Project Structure
- `/data`: Directory for storing datasets.
- `/notebooks`: Jupyter notebooks for experiments and analysis.
- `/testing`: Put random testing scripts and files in here...you may need them later.
- `/development`: Directory for development scripts and modules.
- `/proudction`: Directory for production-ready code and models.


You can use `conda deactivate` to exit the conda environment
