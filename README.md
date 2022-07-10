# scipy2022-talk

Usage

git clone https://github.com/rasbt/scipy2022-talk.git
cd scipy2022-talk
conda create -n coral-pytorch python=3.8
conda activate coral-pytorch
pip install -r requirements.txt

cd src
python main_rnn.py \
--batch_size 16 \
--data_path ../datasets/ \
--learning_rate 0.005 \
--mixed_precision true \
--num_epochs 2 \
--num_workers 3 \
--output_path ./rnn_tripadvisor


## More examples

CNN 

MLP