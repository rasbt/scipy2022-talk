# scipy2022-talk

Usage

```bash
git clone https://github.com/rasbt/scipy2022-talk.git
cd scipy2022-talk
conda create -n coral-pytorch python=3.8
conda activate coral-pytorch
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

```python
cd src
python main_mlp.py \
--batch_size 16 \
--data_path ../datasets/ \
--learning_rate 0.005 \
--mixed_precision true \
--num_epochs 40 \
--num_workers 3 \
--output_path ./cement_strength \
--loss_mode corn
```

```python
python main_mlp.py \
...
--loss_mode crossentropy
```

## More examples

CNN 

MLP