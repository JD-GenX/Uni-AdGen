_base_ = '../base.py'

train_data=[
    dict(task_type='p2i', data_name='hico', batch_size=4),
]
test_data=dict(task_type='p2i', data_name='hico', batch_size=10)

janus_path="../ckpt/Janus-Pro-7B"

use_textual=True
use_special_tokens=True

tuning_mode='lora_control'

max_test_len=250
max_train_steps=1500 
max_seq_len=2000
validation_steps=1000

gradient_accumulation_steps=1

working_dir='project/uniadgen/personalized_dataset'

# 测试数据
grit_json='/home/process_data/PAd1M/personalized/test_dataset.json' 

# 权重加载
resume_train=None

sampling_mode='pss'

history_length=10

