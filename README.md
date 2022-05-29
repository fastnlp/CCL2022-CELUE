# CCL 2022 中文高效自然语言理解模型评测

**最新消息**
- `2022.5.23` 训练集和验证集将于2022年6月1日在该网站发布，报名入口后续会在[智源指数网站](http://cuge.baai.ac.cn/#/)上开放，参赛者通过注册并完善参赛队伍信息即可完成报名。
---

- 组织者
  - 刘向阳（复旦大学）
  - 孙天祥（复旦大学）
  - 姚远  （清华大学）
  - 何俊亮（复旦大学）
  - 吴嘉文（复旦大学）
  - 吴玲玲（复旦大学）
  - 刘知远（清华大学）
  - 邱锡鹏（复旦大学）
- 联系人
  - 刘向阳（xiangyangliu20@fudan.edu.cn）

## 1.任务内容

超大规模的预训练模型已经将大多数自然语言处理任务推向了新的SOTA（State-of-the-art），伴随而来的则是越来越昂贵的计算代价。现如今，越来越多的研究者开始将注意力放在了模型效率和易用性上，而不是一再追求遥不可及的SOTA性能。这些工作的目标从追求SOTA转移到了“帕累托SOTA”。帕累托SOTA模型意味着目前没有其他模型能够在所有感兴趣的维度上都优于它。相比于性能的评估，不同研究对效率的评估不尽相同，这使得很难对其进行全面公平地对比，很难判断一种方法是否以及在多大程度上实现了帕累托改进，尤其是在对比不同加速方法的情况下。

复旦大学针对目前自然语言处理领域两个高效方法分支：静态方法（知识蒸馏、模型剪枝等）以及动态方法（早退技术等）提出了一个可以用于标准评估模型推断效率的基准——ELUE[^1]（Efficient Language Understanding Evaluation）。为了使评估结果更加全面且公平，ELUE采用了FLOPs（浮点运算次数）作为模型效率的指标，同时采用了多维度的评测，结合效率以及性能为模型计算出一个综合的ELUE得分，得分越高表明在相同的FLOPs该模型能达到越高的性能，或者说是在相同的性能下该模型的FLOPs更少。


## 2.评测数据

本次评测中我们提供五个中文自然语言理解任务数据集：CNHC, WordSeg-Weibo, PKU-SEGPOS, CCPM, CMeEE. 所有任务和数据集的统计如表1所示。

<p align='center'>表1：任务和数据集的描述与统计</p>
<table align='center'>
<tr align='center'>
<td> 数据集  </td>
<td> |Train|  </td>
<td> |Dev|  </td>
<td> |Test|  </td>
<td> 任务类型  </td>
<td> 指标  </td>
<td> 数据域  </td>
</tr>

<tr align='center'>
<td> CNHC  </td>
<td> 156k  </td>
<td> 36k  </td>
<td> 36k  </td>
<td> 文本分类  </td>
<td> 准确率  </td>
<td> 新闻  </td>
</tr>

<tr align='center'>
<td> WordSeg-Weibo  </td>
<td> 20.1k  </td>
<td> 2.1k  </td>
<td> 8.6k  </td>
<td> 中文分词  </td>
<td> F1分数  </td>
<td> 博客  </td>
</tr>

<tr align='center'>
<td> PKU-SEGPOS  </td>
<td> 31.7k  </td>
<td> 5.2k  </td>
<td> 4.8k  </td>
<td> 词性标注  </td>
<td> F1分数  </td>
<td> 新闻  </td>
</tr>

<tr align='center'>
<td> CCPM  </td>
<td> 21.8k  </td>
<td> 2.7k  </td>
<td> 2.7k  </td>
<td> 诗歌匹配  </td>
<td> 准确率  </td>
<td> 中文诗歌  </td>
</tr>

<tr align='center'>
<td> CMeEE  </td>
<td> 15.0k  </td>
<td> 5.0k  </td>
<td> 3.0k  </td>
<td> 命名实体识别  </td>
<td> F1分数  </td>
<td> 医疗  </td>
</tr>
</table>

- CNHC  
CNHC（Chinese News Headline Categorization）数据集是在NLPCC（The CCF Conference on Natural Language Processing & Chinese Computing）2017发布的新闻标题分类数据集[^2]，数据集是通过在头条、新浪等新闻媒体网页上搜集相关新闻标题并进行标注得到的，总共有十八个类别：entertainment, sports, car, society, tech, world, finance, game, travel, military, history, baby, fashion, food, discovery, story, regimen, essay，其中discovery, story, regimen, essay这四个类别的样本数量为8000，其余类别的样本数量均为14000。该数据集由复旦大学提供。
- WordSeg-Weibo  
该数据集是在NLPCC 2016发布的中文分词评测数据集[^3]，数据集是在新浪微博收集到相关博客并进行标注得到的。与传统单一的分词评价方法不同，本任务引入了一种新的多粒度分词评价准则。数据集中的总字数和总次数分别为1077854和652740，字和词的类型数分别为4838和56155。该数据集由复旦大学提供。
- PKU-SEGPOS  
该数据是由人民日报语料进行标注的词性标注数据集。数据中总共包含39个词性标记，除了使用《现代汉语语法信息词典》中的26个词性标记（名词n、时间词t、处所词s、方位词f、数词 m、量词q、区别词b、代词r、动词v、形容词a、状态词z、副词d、介词p、连词c、助词u、语气词y、叹词e、拟声词o、成语i、习用语l、简称j、前接成分h、后接成分k、语素g、非语素字x、标点符号w）外，增加了以下3类标记：  
①专有名词的分类标记，即人名nr，地名ns，团体机关单位名称nt，其他专有名词nz；  
②语素的子类标记，即名语素Ng，动语素Vg，形容语素Ag，时语素Tg，副语素Dg；   
③动词和形容词的子类标记，即名动词vn（具有名词特性的动词），名形词an（具有名词特性的形容词），副动词vd（具有副词特性的动词），副形词ad（具有副词特性的形容词）。  
数据分割时，将2000年1月和2000年12月1日-15日的语料作为训练集，2000年12月16日-12月23日的语料作为开发集，2000年12月24日-12月31日的语料作为测试集，该数据集由北京大学提供。
- CCPM  
中国古典诗歌翻译数据集CCPM[^4] (Chinese Classical Poetry Retrieval Dataset-Multiple Choice)，可用于诗歌的匹配、理解和翻译。给定中国古典诗歌的现代文描述，要求从候选的四句诗中挑选出与给定的现代文描述意思符合的那一句诗歌。数据来自网站上提供的中国古典诗歌和其相应的现代汉语翻译，由清华大学收集整理。
- CMeEE  
CMeEE数据集[^5]主要用于医学实体识别任务。该任务共标注了938个文件，47,194个句子，包含了504种常见的儿科疾病、7,085种身体部位、12,907种临床表现、4,354种医疗程序等9大类医学实体，将医学文本命名实体划分为九大类，包括：疾病（dis），临床表现（sym），药物（dru），医疗设备（equ），医疗程序（pro），身体（bod），医学检验项目（ite），微生物类（mic），科室（dep）。标注之前对文章进行自动分词处理，所有的医学实体均已正确切分。数据集来源于CHIP2020学术评测比赛，由北京大学、郑州大学和鹏城实验室联合提供。


## 3.评价标准

评测性能时，对于CCPM数据集我们采用准确率作为评测指标，另外三个数据集采用F1分数作为评测指标，其计算方式分别如下：
$$\text{准确率} = \frac{\text{预测正确的题数}}{\text{总题数}},$$
$$\text{F1} = \frac{\text{2 * P * R}}{\text{P + R}},$$
其中P和R分别为精确率（Precision）和召回率（Recall）。对于模型推断效率，我们采用FLOPs（浮点运算次数）作为评测指标。
获得了模型在每个数据集上的性能和FLOPs后，我们需要结合这两个指标计算模型在每个数据集上的ELUE分数，然后取其平均分数，平均ELUE分数越高的模型排名越高。我们将每个提交都视为二维坐标中的一个点 $(p,f)$，其中 $p$ 是性能，$f$ 为FLOPs. 为了衡量每个点的好坏，我们需要一个基线。因此我们会提供一个基线模型——ElasticBERT-Chinese，该模型每一层（总层数为 $L$）都有一个预训练任务，在预训练阶段我们对这 $L$ 个预训练任务的损失进行联合优化。这种预训练方式能够获得 $L$ 个不同规模（效率）的模型，因为我们在任意层截断ElasticBERT-Chinese后，截取后的小模型都使用预训练任务在大量的语料上预训练过，且其在下游任务上的性能也比较优异[^1]。利用截取得到的 $L$ 个模型，可以在每个数据集上获得 $L$ 个坐标点 $(p^{EB}_i, f^{EB}_i)^L_i$, 利用这些坐标点来插值出曲线 $p^{EB}(f)$。我们将这条曲线视为基准曲线，并通过将参赛者提交的数据点 $(p,f)$ 与该基准曲线进行对比来获得最终的ELUE分数，图1为计算ELUE分数的示意图，且具体的计算公式如下：

$$\text{ELUE}_{\text{score}} = \frac{1}{n}\sum_i^n{\Delta_i} = \frac{1}{n}\sum_i^n{[p_i - p^{EB}(f_i)]},$$

其中 $p_i$ 和 $f_i$ 为所提交模型的性能和对应的FLOPs，$n$ 是所提交模型的数据点数，考虑到一些动态方法（如早退方法）可以灵活调整模型的推断效率，因此允许提交时包含多个数据点（一个数据点包含一个FLOPs以及对应的性能）。

<div align=center><img width="468" height="236" src="https://github.com/fastnlp/CCL2022-CELUE/blob/main/img/elue_score.png"/></div>
<p align='center'>图1：计算ELUE分数的示意图</p>

## 4.提交方式

最终的结果提交与评测均在智源指数网站[^6]（[CUGE](http://cuge.baai.ac.cn/#/)）的上进行，届时智源指数网站将会开通相应的提交与评测系统，参赛者可以在网站上注册账号并提交相应的测试文件。对于每个数据集，需要提交两种文件：（1）包含预测结果的测试文件，（2）定义模型的Python文件。每个数据集一个单独文件夹且以数据名称命名。测试文件可以有多个，每个都表示在一定效率下的预测结果，其格式如图2所示。

<div align=center><img width="389" height="78" src="https://github.com/fastnlp/CCL2022-CELUE/blob/main/img/test.png"/></div>
<p align='center'>图2：测试文件示例</p>

除了测试样本序号和预测的结果外，测试文件中还需要有一列“modules”，这一列表示在对相应的样本进行预测时通过了模型的哪些模块。每个模块前的数字代表该模块的输入形状，例如“emb”前的“(10)”表示输入的“emb”是一个长度为10的序列。除了测试文件之外，还需要提交一个定义模型的Python文件，仅支持使用Pytorch实现模型。图3是一个使用PyTorch和Transformers代码库实现的Python文件示例。通过提交定义模型的Python文件，ELUE能够计算平均浮点运算次数，以及模型的参数数量。

由于本次评测会重点关注模型的推断效率，在提交的时候会限制所提交模型的规模大小，允许提交的模型参数数量需控制在0-140M内（评测组织方所提供的基线模型ElasticBERT-Chinese约有120M的参数）。

<div align=center><img width="392" height="451" src="https://github.com/fastnlp/CCL2022-CELUE/blob/main/img/python.png"/></div>
<p align='center'>图3：定义模型的Python文件示例</p>

## 5.评测赛程

- 开放报名并公布训练集和验证集：2022.6.1
- 公布无答案的测试集：2022.8.15
- 开放提交系统：2022.8.25 – 2022.9.20
- 报名结束：2022.9.1
- 开放提交技术报告及比赛代码：2022.9.10 – 2022.9.20
- 公布结果：2022.9.31

## 6.奖项设置

本届评测由中国中文信息学会为获奖队伍颁发荣誉证书，具体奖项设置如下：

- 一等奖：0-1名
- 二等奖：0-2名
- 三等奖：0-3名

**参考文献**

[^1]: Liu, X., Sun, T., He, J., Wu, L., Zhang, X., Jiang, H., ... & Qiu, X. (2021). Towards Efficient NLP: A Standard Evaluation and A Strong Baseline. In Proceedings of NAACL 2022.
[^2]: Qiu, X., Gong, J., & Huang, X. (2017, November). Overview of the nlpcc 2017 shared task: Chinese news headline categorization. In National CCF Conference on Natural Language Processing and Chinese Computing (pp. 948-953). Springer, Cham.
[^3]: Qiu, X., Qian, P., & Shi, Z. (2016). Overview of the NLPCC-ICCPOL 2016 shared task: Chinese word segmentation for micro-blog texts. In Natural Language Understanding and Intelligent Applications (pp. 901-906). Springer, Cham.
[^4]: Li, W., Qi, F., Sun, M., Yi, X., & Zhang, J. (2021). CCPM: A Chinese Classical Poetry Matching Dataset. arXiv preprint arXiv:2106.01979.
[^5]: Hongying, Z., Wenxin, L., Kunli, Z., Yajuan, Y., Baobao, C., & Zhifang, S. (2020, May). Building a pediatric medical corpus: Word segmentation and named entity annotation. In Workshop on Chinese Lexical Semantics (pp. 652-664). Springer, Cham.
[^6]: Yao, Y., Dong, Q., Guan, J., Cao, B., Zhang, Z., Xiao, C., ... & Sun, M. (2021). CUGE: A Chinese Language Understanding and Generation Evaluation Benchmark. arXiv preprint arXiv:2112.13610.
