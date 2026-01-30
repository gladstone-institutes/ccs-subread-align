FROM condaforge/mambaforge:24.9.0-0

# Set shell options
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install libmamba solver for faster dependency resolution
RUN conda install -n base conda-libmamba-solver && \
    conda config --set solver libmamba

# Create conda environment and install packages
RUN mamba create -n subread_env python=3.10 -y && \
    /opt/conda/envs/subread_env/bin/python -m pip install --upgrade pip && \
    mamba install -n subread_env -c conda-forge -y \
        git \
        pandas \
        notebook \
        tqdm && \
    mamba install -n subread_env -c bioconda -y pbtk && \
    /opt/conda/envs/subread_env/bin/pip install edlib git+https://github.com/PacificBiosciences/pbcore.git && \
    /opt/conda/envs/subread_env/bin/pip install "numpy==1.22.4" && \
    /opt/conda/envs/subread_env/bin/pip install "matplotlib>=3.5,<3.9" "seaborn>=0.11,<0.13" ipywidgets plotly scipy --no-deps && \
    /opt/conda/envs/subread_env/bin/pip install kiwisolver pillow pyparsing cycler fonttools "contourpy==1.0.7" --no-deps

# Make conda environment available
ENV PATH="/opt/conda/envs/subread_env/bin:$PATH"
RUN echo "conda activate subread_env" >> ~/.bashrc

# Test all package imports with proper environment activation and check numpy version
RUN /bin/bash -c "source activate subread_env && python -c 'import numpy; print(f\"NumPy version: {numpy.__version__}\"); assert numpy.__version__.startswith(\"1.22\"), f\"Wrong numpy version: {numpy.__version__}, expected 1.22.x\"'" && \
    /bin/bash -c "source activate subread_env && python -c 'import pandas; print(\"✓ pandas\")'" && \
    /bin/bash -c "source activate subread_env && python -c 'import notebook; print(\"✓ notebook\")'" && \
    /bin/bash -c "source activate subread_env && python -c 'import tqdm; print(\"✓ tqdm\")'" && \
    /bin/bash -c "source activate subread_env && python -c 'import matplotlib.pyplot as plt; print(\"✓ matplotlib\")'" && \
    /bin/bash -c "source activate subread_env && python -c 'import seaborn; print(\"✓ seaborn\")'" && \
    /bin/bash -c "source activate subread_env && python -c 'import ipywidgets; print(\"✓ ipywidgets\")'" && \
    /bin/bash -c "source activate subread_env && python -c 'import plotly; print(\"✓ plotly\")'" && \
    /bin/bash -c "source activate subread_env && python -c 'import scipy; print(\"✓ scipy\")'" && \
    /bin/bash -c "source activate subread_env && python -c 'import pbcore.io; print(\"✓ pbcore\")'" && \
    /bin/bash -c "source activate subread_env && python -c 'import edlib; print(\"✓ edlib\")'" && \
    /bin/bash -c "source activate subread_env && which bam2fasta && echo '✓ pbtk tools available'" && \
    echo "All packages imported successfully!"

CMD ["/bin/bash"]

# Copy dockerfile into the image
COPY Dockerfile /Dockerfile