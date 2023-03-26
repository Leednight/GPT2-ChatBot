# GPT2 for Chinese chitchat

## 项目描述
- 本项目是基于GPT2的中文闲聊机器人，模型实现基于HuggingFace的[transformers](https://github.com/huggingface/transformers)。
- 在生成阶段，使用了Temperature、Top-k Sampling和Nucleus Sampling等，可参考论文[The Curious Case of Neural Text Degeneration](https://arxiv.xilesou.top/pdf/1904.09751.pdf)
- 本项目受 [GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese)启发
- 本项目受 [GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat)启发

## 运行环境
python3.8、 transformers==4.2.0、pytorch==1.7.1

## 项目结构
- config文件夹里面的config放的预训练模型的参数的配置文件（存放GPT2模型的参数的配置文件）
- data文件夹
    - train.txt:默认的原始训练集文件，存放闲聊语料 
    - train.pkl:对原始训练语料进行tokenize之后的文件,存储一个list对象，list的每条数据表示一个多轮对话，表示一条训练数据
- model文件夹:存放对话生成的模型
    - epoch40:经过40轮训练之后得到的模型
      - config.json:模型参数的配置文件
      - pytorch_model.bin:模型文件
- vocab文件夹
    - vocab.txt:字典文件。默认的字典大小为13317，若需要使用自定义字典，需要将confog.json文件中的vocab_size字段设为相应的大小。
- sample文件夹:存放人机闲聊生成的历史聊天记录
- preprocess.py:数据预处理代码
- train.py:训练代码
- interact.py:人机交互代码
- dataset.py:数据读取以及预处理
- num_workers_dataset文件夹: 存放用來測CPU最佳num_workers的数据
- choose_best_num_workers.py: 測CPU最佳num_workers的脚本
- pytorchtools.py: early stoping
- generate_dialogue_subset.py: 用于生成对话训练子集
- interact_mmi.py:人机交互代码mmi模型
- dialogue_model:存放对话生成的模型
- mmi_model:存放MMI模型(maximum mutual information scoring function)，用于预测P(Source|response)


## 模型简介
### 模型结构
![avatar](figure/model.png)


### 模型参数简介(详见模型的config.json文件)
- initializer_range: 0.02
- layer_norm_epsilon: 1e-05
- n_ctx: 1024
- n_embd: 768
- n_head: 12
- n_layer: 12
- n_positions: 1024
- vocab_size: 21128

## 训练思路
对每条训练数据进行拼接，然后将其输入到模型中，进行训练。

对于如下多轮闲聊训练数据,在训练模型时，将训练数据进行如下拼接:"[CLS]想看你的美照[SEP]亲我一口就给你看[SEP]我亲两口[SEP]讨厌人家拿小拳拳捶你胸口[SEP]"。然后将上述拼接结果作为模型的输入，让模型进行自回归训练。
```
想看你的美照
亲我一口就给你看
我亲两口
讨厌人家拿小拳拳捶你胸口
```

## 使用方法
### Quick Start
在[模型分享](#model_share)中下载模型，将模型文件夹model_epoch40_50w放到model目录下，执行如下命令，进行对话
```
python interact.py --no_cuda --model_path model_epoch40_50w (使用cpu生成，速度相对较慢)
或
python interact.py --model_path model_epoch40_50w --device 0 (指定0号GPU进行生成，速度相对较快)
```


###  数据预处理
在项目根目录下创建data文件夹，将原始训练语料命名为train.txt，存放在该目录下。train.txt的格式如下，每段闲聊之间间隔一行，格式如下：
```
真想找你一起去看电影
突然很想你
我也很想你

想看你的美照
亲我一口就给你看
我亲两口
讨厌人家拿小拳拳捶你胸口

美女约嘛
开好房等你了
我来啦
```
运行preprocess.py，对data/train.txt对话语料进行tokenize，然后进行序列化保存到data/train.pkl。train.pkl中序列化的对象的类型为List[List],记录对话列表中,每个对话包含的token。
```
python preprocess.py --train_path data/train.txt --save_path data/train.pkl
```

### 训练模型
运行train.py,使用预处理后的数据，对模型进行自回归训练，模型保存在根目录下的model文件夹中。

在训练时，可以通过指定patience参数进行early stop。当patience=n时，若连续n个epoch，模型在验证集上的loss均没有下降，则进行early stop，停止训练。当patience=0时，不进行early stop。

代码中默认关闭了early stop，因为在实践中，early stop得到的模型的生成效果不一定会更好。
```
python train.py --epochs 40 --batch_size 8 --device 0,1 --train_path data/train.pkl
```
更多的训练参数介绍，可直接看train.py中的set_args()函数中的参数说明

### 人机交互
运行interact.py，使用训练好的模型，进行人机交互，输入Ctrl+Z结束对话之后，聊天记录将保存到sample目录下的sample.txt文件中。
```
python interact.py --no_cuda --model_path path_to_your_model --max_history_len 3(由于闲聊对话生成的内容长度不是很长，因此生成部分在CPU上跑速度也挺快的)
```
执行interact.py时，可以尝试通过调整topk、topp、repetition_penalty、max_history_len等参数，调整生成的效果。更多的参数介绍，可直接看interact.py的set_args()函数中的参数说明
如果要使用GPU进行生成，则不要调用--no_cuda参数，并且通过--device gpu_id来指定使用哪块GPU。


## 闲聊语料分享
|中文闲聊语料 | 数据集地址 |语料描述|
|---------|--------|--------|
|常见中文闲聊|[chinese_chatbot_corpus](https://github.com/codemayq/chinese_chatbot_corpus)|包含小黄鸡语料、豆瓣语料、电视剧对白语料、贴吧论坛回帖语料、微博语料、PTT八卦语料、青云语料等|
|50w中文闲聊语料 | [百度网盘【提取码:4g5e】](https://pan.baidu.com/s/1M87Zf9e8iBqqmfTkKBWBWA) 或 [GoogleDrive](https://drive.google.com/drive/folders/1QFRsftLNTR_D3T55mS_FocPEZI7khdST?usp=sharing) |包含50w个多轮对话的原始语料、预处理数据|
|100w中文闲聊语料 | [百度网盘【提取码:s908】](https://pan.baidu.com/s/1TvCQgJWuOoK2f5D95nH3xg) 或 [GoogleDrive](https://drive.google.com/drive/folders/1NU4KLDRxdOGINwxoHGWfVOfP0wL05gyj?usp=sharing)|包含100w个多轮对话的原始语料、预处理数据|


中文闲聊语料的内容样例如下:
```
谢谢你所做的一切
你开心就好
开心
嗯因为你的心里只有学习
某某某，还有你
这个某某某用的好

你们宿舍都是这么厉害的人吗
眼睛特别搞笑这土也不好捏但就是觉得挺可爱
特别可爱啊

今天好点了吗？
一天比一天严重
吃药不管用，去打一针。别拖着
```

<h2 id="model_share">模型分享</h2>

|模型 | 共享地址 |模型描述|
|---------|--------|--------|
|model_epoch40_50w | [百度网盘【提取码:ju6m】](https://pan.baidu.com/s/1iEu_-Avy-JTRsO4aJNiRiA) 或 [GoogleDrive](https://drive.google.com/drive/folders/1fJ6VuBp4wA1LSMpZgpe7Hgm9dbZT5bHS?usp=sharing) |使用50w多轮对话语料训练了40个epoch，loss降到2.0左右。|


## 生成样例
#### Sample 1:
``` 
  user :你好啊
chatbot:你好，请问您是？
  user :我是你的主人啊 
chatbot:你是狗吗？
  user :你才是狗
chatbot:你不是狗我是你的小主人
  user :你这个大笨蛋
chatbot:你大笨蛋，你是想挨打了
  user :你全家都是大笨蛋
chatbot:我是你爸爸
  user :算了，不跟你吵了，你在干嘛呢
chatbot:我在上班啊，你还没下班呢啊你
  user :我周末要做作业啊
chatbot:不是才交了吗，早晚做完啊
  user :好多作业啊，还没做完呢
chatbot:我在抄，我妈让我周五交，还有事
  user :你就不怕被你妈妈骂吗
chatbot:不怕
  user :你周末有去哪里玩吗
chatbot:没时间啊，忙得很啊
  user :忙着干嘛呢
chatbot:不知道啊，我周末没有作业，我想睡觉，然后还得找人
```

#### Dialogue Model
Dialogue Model是基于GPT2模型的生成模型，对每条训练数据进行"顺序"拼接，然后将其输入到网络中，进行训练(此处的"顺序"是相对于MMI Model的"逆序")
例如存在如下多轮闲聊训练数据,在训练Dialogue Model时，将上述训练数据进行如下拼接:"[CLS]想看你的美照[SEP]亲我一口就给你看[SEP]我亲两口[SEP]讨厌人家拿小拳拳捶你胸口[SEP]"。然后将上述拼接结果作为Dialogue Model的输入，对模型进行训练

#### MMI Model
MMI Model的思想基于微软的论文DialoGPT:Large-Scale Generative Pre-training for Conversational Response Generation
MMI Model也是一个基于GPT2的生成模型，将每条训练数据进行"逆序"拼接,然后输入到网络中。该模型主要用于计算Dialogue Model生成的所有候选response相对于dialogue history的loss。
训练时，将一条训练语料进行逆序拼接，如 “[CLS]讨厌人家拿小拳拳捶你胸口[SEP]我亲两口[SEP]亲我一口就给你看[SEP]想看你的美照[SEP]”，并作为MMI Model的输入进行训练
#### response生成步骤
假设当前dialogue history=[“你好”,“你好呀”,“你在干嘛呢”]
首先使用Dialogue Model根据dialogue history生成n个候选response:[“在看电视”,“我在上课啊”,“人家在想你啊”,“我不知道”]
使用MMI Model将每个候选response分别与dialogue history进行逆序拼接，如 "[CLS]在看电视[SEP]你在干嘛呢[SEP]你好呀[SEP]你好[SEP]"
将上述拼接结果作为MMI Model的输入，计算每个response的loss
选择loss最小的response作为最终的结果进行回复
#### 模型分享
闲聊语料大小为67M，包含50w个多轮对话。使用该语料训练了两个模型dialogue_model与mmi_model
#### dialogue_model
使用闲聊语料训练了40个epoch，最终loss在2.0左右，继续训练的话，loss应该还能继续下降。
【提取码:osi6】
https://pan.baidu.com/s/1qDZ24VKLBU9GKARX9Ev65g
#### mmi_model
以dialogue_model作为预训练模型，使用上述闲聊语料，训练了40个epoch，最终loss在1.8-2.2之间，继续训练的话，loss也能继续下降。
【提取码:1j88】
https://pan.baidu.com/s/1ubXGuEvY8KmwEjIVTJVLww


## Reference
- [The Curious Case of Neural Text Degeneration](https://arxiv.xilesou.top/pdf/1904.09751.pdf)
- [transformers](https://github.com/huggingface/transformers)
- [GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese)
- [GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat)




