
sudo apt install -y python3-venv python3-pip build-essential

python3 -m venv .venv  
source .venv/bin/activate
pip install -U pip

python -m src.faceid.build_library_faiss
