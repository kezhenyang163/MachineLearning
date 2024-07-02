### 1、KNN(k nearest neighbors) 临近算法

#### 1.1、什么是KNN

KNN是一个极其简单的算法，中文叫K近邻算法；k近邻算法既可以用于分类，也可以用于回归，这里我们只介绍其分类算法。

* k表示个数 你有几个好朋友？ 你有几个邻居？

* 邻居是什么样性质，类别；根据这个特征，来对事物进行分类



#### 1.2、KNN算法的核心原理

* 给定一个预测目标，然后计算预测目标和所有样本之间的距离或者相似度，选择距离最近的前K个样本，然后通过这些样本来投票决策。K值是人为定义的，可以通过技巧选择合适的K值；一般对于二分类问题来说，把K设置为奇数是容易防止平局的现象

  <img src="./img/001-KNN算法.png" alt="image.png" style="zoom:80%;margin-left:0px" />

#### 1.3、距离度量

KNN算法的理论基础就是欧式距离(欧几里得距离公式)；

点A(1,2)，点B(4,6)，请问A和B之间的距离怎么计算

欧式距离

$$
distance_{AB} = \sqrt{(4 - 1)^2 + (6-2)^2} = \sqrt{3^2 + 4^2} = 5
$$
点A(2,3,4)，点B(5,8,9):
$$
  distance_{AB} = \sqrt{(5-2)^2 + (8-3)^2 + (9-4)^2}
$$
点A(x1,x2,x3,x4,……xn)，B(y1,y2,y3,y4,……yn)：
$$
distance_{AB} = \sqrt{\sum_{i=1}^n(x_i - y_i)^2}
$$
#### 1.4、KNN超参数

```
KNN超参数解释:
KNeighborsClassifier(
    n_neighbors=5,
    *,
    weights='uniform',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski',
    metric_params=None,
    n_jobs=None,
)
```

`KNeighborsClassifier` 是 Python 的 `scikit-learn` 库中实现的 K-最近邻分类器。以下是该分类器中各个超参数的解释：

1. **n_neighbors** (`int`): 
   - 指定了用于决策的最近邻居的数量。默认值为5。这个参数是最重要的超参数之一，因为它直接影响分类决策。

2. **weights** (`str` or `callable`):
   - 指定在进行决策时如何为邻居分配权重。可以是以下两种：
     - `'uniform'`：所有邻居的权重相同。
     - `'distance'`：权重与距离成反比，即距离越近的邻居对决策的影响越大。
     - 也可以是一个函数，该函数接受距离数组和返回权重数组。

3. **algorithm** (`str` or `callable`):
   - 指定用于搜索邻居的算法。可以是：
     - `'auto'`：根据数据集的特征数量自动选择最合适的算法。
     - `'ball_tree'`：使用球树数据结构。
     - `'kd_tree'`：使用KD树数据结构。
     - `'brute'`：使用暴力搜索，即直接计算每个点之间的距离。
     - 也可以是一个自定义的函数。

4. **leaf_size** (`int`):
   - 用于确定树结构（如KD树或球树）中叶节点的最大大小。在构建树时，这个参数可以影响算法的效率。默认值为30。

5. **p** (`int` or `float`):
   - 指定闵可夫斯基距离的度量参数。当`p=1`时，它是曼哈顿距离；当`p=2`时，它是欧氏距离；当`p`趋向于无穷大时，它是切比雪夫距离。默认值为2。

6. **metric** (`str` or `callable`):
   - 指定用于计算距离的度量。可以是：
     - `'minkowski'`：闵可夫斯基距离。
     - `'euclidean'`：欧氏距离。
     - `'manhattan'`：曼哈顿距离。
     - `'chebyshev'`：切比雪夫距离。
     - `'seuclidean'`：标准化欧氏距离。
     - 也可以是一个自定义的距离度量函数。

7. **metric_params** (`dict`):
   - 用于传递给度量函数的额外参数。例如，如果使用`'minkowski'`度量，可以在这里指定`p`值。

8. **n_jobs** (`int` or `None`):
   - 指定在执行算法时并行运行的任务数量。如果设置为`None`，则使用CPU的可用核心数量。如果设置为负数，则使用所有可用的核心（1-表示使用除主核心外的所有核心）。

这些超参数允许用户根据具体的数据集和问题调整KNN分类器，以获得最佳的性能。

* p = 1 曼哈顿距离 p= 2 表示欧式距离

  红色的线就是曼哈顿距离；蓝色和黄色等价曼哈顿距离；绿色线就是欧式距离

  <img src="img\004-KNN-Length-Compute.jpg" alt="image.png" style="zoom:110%;margin-left:0" />

#### 1.4、KNN算法缺陷

1. **计算复杂度高**：KNN算法在进行预测时需要计算待分类样本与所有训练样本的距离，当数据集较大时，这会导致计算量非常大，从而影响算法的效率。
2. **对参数K的选择敏感**：K值的选择对分类结果有显著影响，不同的K值可能会导致不同的分类效果，因此需要通过交叉验证等技术来确定最优的K值。
3. **存储空间需求大**：KNN算法需要存储所有的训练数据，这在大数据集的情况下会占用大量的内存 。
4. **对噪声数据敏感**：由于KNN算法是基于邻近数据点的，因此它对噪声数据比较敏感，噪声数据可能会影响分类的准确性。
5. **样本不平衡问题**：在样本类别分布不均衡的情况下，KNN算法可能会偏向于多数类，导致对少数类的预测准确率较低。
6. **特征维度的诅咒**：当特征维度很高时，KNN算法的性能可能会下降，因为高维空间中的距离度量可能会变得不那么有效。
7. **预测速度慢**：由于KNN算法在预测时需要进行大量的距离计算，这可能导致预测速度较慢，尤其是在样本数量较大的情况下。
8. **可解释性不强**：与决策树等算法相比，KNN模型的可解释性不强，这可能会影响模型的透明度和解释能力。
9. **对异常值敏感**：尽管KNN算法对异常点不敏感，但在某些情况下，异常值的存在可能会影响最近邻的计算，从而影响分类结果。
10. **需要优化方法**：为了解决KNN算法的缺点，可能需要采用近似最近邻搜索或层次化KNN分类器等优化方法来提高分类效率。

<img src="img\003-KNN-Short.png" alt="image.png" style="zoom:120%;margin-left:0" />

#### 1.5、KNN算法优点

1. **简单易懂**：KNN算法的原理简单直观，容易理解和实现。
2. **无需训练**：作为一种惰性学习算法，KNN不需要在训练阶段构建模型，所有的计算都是在预测阶段进行。
3. **易于实现**：KNN算法的实现相对简单，不需要复杂的数学模型或优化过程。
4. **适用于多种问题**：KNN算法既可以用于分类问题，也可以用于回归问题，具有很好的通用性。
5. **快速计算**：在数据集较小或经过优化的情况下，KNN算法可以快速给出预测结果。

### 2、KNN的案例

鸢尾花分类 生长的环境不同，所以类别3类 ；类别不同，性质不同：花萼长宽不一样，花瓣长宽不同了。

植物学家，根据形状不同，进行分类

分类算法使用流程：

1. 加载数据
2. 数据预处理，拆分训练集和测试集数据
3. 声明算法，给定超参数
4. 训练算法，算法学习数据，归纳规律
5. 算法，通过数学，找到，数据和目标值之间的规律
6. 算法找到规律，应用

```python
# 导入算法和 sklearn自带数据集
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import datasets 

# Load 鸢尾花数据集，分三类，鸢尾花花萼和花瓣长宽  X大写，约定
X,y = datasets.load_iris(return_X_y=True)
display(X,y)

# 数据进行拆分
from sklearn.model_selection import train_test_split 

# train训练数据，将训练数据，交给算法，进行建模，总结规律
# test测试，应用规律
# train_test_split拆分数据【随机拆分】 X_train和X_test每次，结果会不同！
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=11) # 30个测试数据
display(X_train.shape,X_test.shape)

# 初始化Knn算法给定 K值
knn = KNeighborsClassifier(n_neighbors=5)

# 120个鸢尾花的特征数据和目标值 对应关系 ;这里的fit，就是向里面填充内容进行训练; 没有fit这一步，下面都不能执行
knn.fit(X_train,y_train)
# 获取准确率
knn.score(X_test,y_test)
```



### 4、手写数字识别【实战案例】

```python
# 导入相关工具包
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data = np.load('./digit.npy')
# 随机选择一个索引，用于从数据集中选取一个图像进行展示。
index = np.random.randint(0,5000,size = 1)[0]
# 创建一个新的图形，并设置图形的大小为2x2英寸
plt.figure(figsize=(2,2))
plt.imshow(data[index])

# 新建一个包含5000个元素的数组，每个数字0到9重复500次，代表图像的标签
y = np.array([0,1,2,3,4,5,6,7,8,9]*500)
y = np.sort(y)
# 将图像数据从形状(5000, 28, 28)展平为(5000, 784)，每个图像变为一个784维的向量。
data = data.reshape(5000,-1) 
# 使用train_test_split函数将数据集分割为训练集和测试集，测试集占10%。
X_train,X_test,y_train,y_test = train_test_split(data,y,test_size=0.1) 
display(X_train.shape,X_test.shape)

# 使用训练集数据拟合KNN模型。
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)

# 使用拟合后的模型对测试集进行预测。
y_ = knn.predict(X_test)
display(y_test[:20],y_[:20])

# 计算模型在测试集上的准确率。
knn.score(X_test,y_test)

# 创建一个新的图形，大小设置为5列2行，总共10个子图。
plt.figure(figsize=(2*5,3*10))

# 循环用于显示测试集中的前50个图像，每个图像旁边显示其真实标签和预测标签
for i in range(50):
    plt.subplot(10,5,i+1)
    # 将测试集中的图像数据重新转换为28x28的图像，并显示。
    plt.imshow(X_test[i].reshape(28,28))
    plt.axis('off')
    plt.title('True:%d\nPredict:%d' % (y_test[i],y_[i]))
```



### 5、癌症诊断【实战案例】

* 有一些微观数据，都是人体细胞内的数据
* 中医：望闻问切
* 西医：各种设备，检查一通
* 无论中医还是西医：
  * 获取指标
  * 获取数据
  * 获取特征
* 看病，通过，指标诊断
* 设备越来越先进，获取更多微观数据、指标
* 有一些病，先微观，变成宏观（感觉不舒服）
* 量变到质变
* 可以尝试找到围观数据和疾病之间的关系！
* 使用算法寻找数据内部的规律
* KNN调整超参数，准确率提升
* 数据归一化、标准化，提升更加明显！

### 6、薪资预测【实战案例】

* 属性清理，将没用属性删除
* 一些属性是str类型的
* pandas中map、agg、apply、transform这些都可以转变！
* 建模
  * knn
  * knn.fit()
  * knn.predict()预测
  * knn.score()准确率，分类
* 模型优秀，模型准确率更高
  * 超参数调整
  * 归一化、标准化
  * pandas.cut()，分箱操作，面元化操作，其实就是分类
  * 把相近的数值，归到一类中
  * 大学时候，体育成绩：优（90～100）、良（80～90）、中等（70～80）、及格（60～70）、不及格（&#x3c;60）
  * 大学成绩，就是分箱操作。
  * 简明扼要。
