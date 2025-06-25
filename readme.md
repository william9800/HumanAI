

环境配置要求：Win11

pip install -r requirements.txt





# 代码使用说明：

train.py :主要训练代码，注释完全，直接跑就行

Test.py 测试代码，用于对测试集的数据进行直接判断，在命令行输出结果

assess.py 评价测试代码，使用测试集，绘制混淆矩阵图片，输出TPR等数据矩阵（在命令行）

Incremental.py 增量训练代码，能够实现在冻结和非冻结全连接层以外两种情况的增量训练

test1.py 同Test，使用Test就可以

model.pth :放置第一次训练结果，使用5000FFHQ+5000stylegan3，epoch=1 batch=32

model_final: 第二次训练结果，综合使用所有数据集，随机函数选取对应数量，epoch=10，每个epoch检查点都有保存。

model_stylegan2: 第三次训练结果，500stylegan2+FFHQ500，epoch=5，进行增量训练测试

model_stylegan2stablediff.pth:第三次训练结果，500stable +500FFHQ，epoch=5，测试发现仅调整全连接层精准度仅60%，这里选择不冻结。

## 文件夹saved_model

保存各实验结果备份，目前存储上述三次实验结果

## 文件夹Random_selection_catagories

存储随机选取代码，主要用于快速调整选取训练集和测试集的数据：

assess_selection.py 存储随机选取代码，能够选取目前文件夹可见图片进行选取

assess_selection_multi.py 遍历当前文件夹，进行选取

test_torch.py 测试torch软件包等版本时使用，可以删除

## 文件夹Image Processing

图片处理和简单清洗攻击使用

imag_compression.py 图片压缩

Interference_Attack.py 实现模糊噪音等不同清洗攻击干扰

## 文件夹dataset

dataset.py 存储数据集的相关代码，在train.py中调用

## 文件夹Data Visualizaion

数据可视化处理使用

## 文件夹data

存储训练数据和测试数据，文件夹内容根据实验内容变化

## 文件夹crawl_ai2

爬虫程序文件夹，爬取了lexica网站，unsplashed网站，生成了stable diffusion的图片

针对doubao和openAI由于网站设置，个人能力不够，目前没有办法自动爬取

## 文件夹Image Processing

图片干扰攻击设置使用的文件夹

## 文件夹T_train

训练用于对比的模型的文件夹

额外声明：由于部分模型训练结果的标准化和分类标准选择不同，一开始是选择多分类训练的，因为考虑到非人脸目标的检测，后续改为二分类，采用判断概率【0，1】之间判断，因此在使用access代码的时候需要考虑到该部分影响，可能需要修改。如果自己训练就没有问题，使用训练结果可能会出错。

# 数据集下载指南



训练和实验结果：

通过网盘分享的文件：HumanAIResult
链接: https://pan.baidu.com/s/1VfhQXu_2NOLHnKRvTj62mg?pwd=yemn 提取码: yemn 



全测试数据集（全Fake）：包括豆包生成图片，SDXl，flux，Dall-3生成的图片，以及从https://generated.photos/faces/asian-race爬取的数据集

通过网盘分享的文件：TestDataset
链接: https://pan.baidu.com/s/1YX1r9SENfLZt0T1Hmbrx1Q?pwd=w9dt 提取码: w9dt 



全训练数据集：

stylegan2（Fake）+stylegan（Fake）+FFHQ（Real)：seeprettyface.com 下载(http://www.seeprettyface.com/mydataset.html)

Star(Real)，CelebA(Real)为公开数据集,图片数目较多，互联网中也有，因此不作分享

FaceV5（Real）：公开数据集，考虑到空间，放置在自建数据集文件夹中。



自建数据集，包括Lexica爬取（Fake），Unsplash爬取（Real），StableDiffusion利用8个不同检查点生成（Fake），Thispersondonotexist网站爬取（Fake）：

通过网盘分享的文件：SelfBuiltDatasets.zip
链接: https://pan.baidu.com/s/13JLYdWEdDiHblSy3zO2mRg?pwd=4gbv 提取码: 4gbv 