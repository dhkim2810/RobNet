pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install requests urllib3 tqdm
python generate_trigger.py --cuda --base_dir . --data_dir ./data