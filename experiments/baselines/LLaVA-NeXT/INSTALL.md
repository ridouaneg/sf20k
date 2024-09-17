git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT

conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
