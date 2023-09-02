本项目使用动态量化（Dynamic Quantization）技术对BERT模型进行量化，并实验量化后的模型在推理性能和效果上的表现。

### Author

- Jclian(jclian91@126.com or jclian91@sina.com)

### 项目结构

启动方式：`jupyter notebook --ip 0.0.0.0`

项目文件：

```bash
$ tree .
.
├── README.md
├── data
│   └── sougou
│       ├── label.json
│       ├── test.csv
│       └── train.csv
├── data_process.ipynb
├── model_eval_chn.ipynb
├── model_train_chn.ipynb
├── model_train_eng.ipynb
├── requirements.txt
└── tokenizer_test.ipynb

```

### 模型实验

量化方案：

```python
# 模型量化
cpu_device = torch.device("cpu")
torch.backends.quantized.engine = 'x86'
# 8-bit 量化
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
).to(cpu_device)
```

base model: [bert-base-chinese](https://huggingface.co/bert-base-chinese)

CPU info: x86-64, Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz

batch size: 1

thread: 1

| 实验  | 最大长度 | 量化前平均推理时间(ms) | 量化前weighted F1值 | 量化前平均推理时间(ms) | 量化前weighted F1值 |
|-----|------|---------------|-----------------|---------------|-----------------|
| 实验1 | 384  | 1066          | 0.9797          | 686           | 0.9838          |
| 实验2 | 384  | 1047.6        | 0.9899          | 738.1         | 0.9879          |
| 实验3 | 384  | 1020.9        | 0.9817          | 714.0         | 0.9838          |
| 实验1 | 256  | 668.7         | 0.9717          | 431.4         | 0.9718          |
| 实验2 | 256  | 675.1         | 0.9717          | 449.9         | 0.9718          |
| 实验3 | 256  | 656.0         | 0.9717          | 446.5         | 0.9718          |
| 实验1 | 128  | 335.8         | 0.9737          | 200.5         | 0.9737          |
| 实验2 | 128  | 336.5         | 0.9737          | 227.2         | 0.9737          |
| 实验3 | 128  | 352.4         | 0.9737          | 217.6         | 0.9737          |

结论：

- 模型推理效果：量化前后基本相同，量化后略有下降
- 模型推理时间：量化后平均提速约1.52倍

### 参考文献

1.  [Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
2.  [DYNAMIC QUANTIZATION ON BERT](https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html)
3.  [RFC-0019-Extending-PyTorch-Quantization-to-Custom-Backends](https://github.com/pytorch/rfcs/blob/master/RFC-0019-Extending-PyTorch-Quantization-to-Custom-Backends.md)