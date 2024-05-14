

<div align=center>

# Report of Data Science for DC Crime

</div>

## 0. Group Member

| Name   | Student ID | T2 (1) | T2 (2) | T2 (3) | T3   | T4   |
| ------ | ---------- | ------ | ------ | ------ | ---- | ---- |
| 赵钊   | 12110120   |        |        |        |      |      |
| 杨皓翔 | 12112523   |        |        |        |      |      |



## 1. Introduction

在现代社会，犯罪问题是一个普遍存在的社会现象，它不仅关系到经济、文化、政治和技术的发展，更直接影响到人民的幸福感和社会的稳定。华盛顿特区（DC）作为美国首都，其犯罪数据的分析对于理解犯罪模式、预防犯罪行为以及提升公共安全具有重要意义。本项目旨在通过深入分析2008年至2017年DC地区的犯罪数据，运用数据挖掘技术揭示其中的潜在信息，并探索犯罪与住房价格之间的关系，以期为政策制定者和社会规划者提供数据支持和决策参考。

随着大数据技术的发展，我们现在有能力处理和分析大规模的犯罪数据集，从而更准确地理解犯罪分布的地理和时间特征。本项目将使用 `DC_Crime.csv` 数据集，该数据集包含了犯罪类型、地理位置、时间等多个维度的信息。此外，为了深入分析犯罪与经济行为之间的关系，我们还将结合DC地区的住房数据集 `DC_Properties.csv`，通过地理信息和时间信息的关联，探索犯罪率与房价之间的潜在联系。

本项目的研究内容包括但不限于以下几个方面：

1. 数据预处理：对原始数据进行清洗，处理缺失值和异常值，为后续分析打下坚实基础。
2. 数据探索与可视化：通过统计和可视化手段，探索数据中的模式和趋势，如犯罪类型、时间分布、地理分布等。
3. 属性相关性分析：分析不同犯罪特征之间的相关性，如犯罪时间与犯罪类型之间的关系。
4. 地理区域与犯罪数量的关联：通过地理空间分析，研究犯罪事件在不同地区的分布情况。
5. 房价预测：结合犯罪数据和住房数据，构建房价预测模型，分析犯罪率对房价的潜在影响。

通过本项目，我们期望能够为理解DC地区的犯罪现象提供新的视角，并为相关领域的研究者和决策者提供有价值的信息和见解。



## 2. Data Preprocessing

### 2.1 属性分类

`DC_Crime.csv` 中共有 29 个属性，每个属性分别带有一些方面的信息，现在，我们把这 29 个属性分成 3 类，分别为地理相关、时间相关、犯罪相关的属性，分别用 G、T、C 标记。分类的结果如下表：

| Attribute            | Category | Attribute       | Category | Attribute      | Category |
| -------------------- | -------- | --------------- | -------- | -------------- | -------- |
| NEIGHBORHOOD_CLUSTER | G        | YEAR            | T        | START_DATE     | T        |
| CENSUS_TRACT         | G        | offensekey      | C        | CCN            | C        |
| offensegroup         | C        | BID             | G        | OFFENSE        | C        |
| LONGITUDE            | G        | sector          | G        | OCTO_RECORD_ID | C        |
| END_DATE             | T        | PSA             | G        | ANC            | G        |
| offense-text         | C        | ucr-rank        | C        | REPORT_DATE    | T        |
| SHIFT                | T        | BLOCK_GROUP     | G        | METHOD         | C        |
| YBLOCK               | G        | VOTING_PRECINCT | G        | location       | G        |
| DISTRICT             | G        | XBLOCK          | G        | LATITUDE       | G        |
| WARD                 | G        | BLOCK           | G        |                |          |

由于原数据表 `DC_Crime.csv` 中的变量排布较为混杂，因此考虑先按照属性的分类对变量进行分组。使用如下代码将变量划分为 3 组并输出到 3 个文件中，再进行下一步处理。

```python
df = pd.read_csv('DC_Crime.csv')

new_order = ['NEIGHBORHOOD_CLUSTER', 'CENSUS_TRACT', 'LONGITUDE', 'YBLOCK', 'DISTRICT', 'WARD', 'BID', 'sector', 'PSA', 'BLOCK_GROUP', 'VOTING_PRECINCT', 'XBLOCK', 'BLOCK', 'ANC', 'location', 'LATITUDE']
df1 = df.reindex(columns=new_order)
df1.to_csv('DC_Crime_G.csv', index=True)

new_order = ['END_DATE', 'SHIFT', 'YEAR', 'START_DATE', 'REPORT_DAT']
df2 = df.reindex(columns=new_order)
df2.to_csv('DC_Crime_T.csv', index=True)

new_order = ['offensegroup', 'offense-text', 'offensekey', 'ucr-rank', 'CCN', 'OFFENSE', 'OCTO_RECORD_ID', 'METHOD']
df3 = df.reindex(columns=new_order)
df3.to_csv('DC_Crime_C.csv', index=True)
```



### 2.2 数据筛选

注意到数据的维数很大，因此需要对数据进行筛选。考虑先对数据间的相关性进行分析，相关性强的数据包含的重复信息较多，对该部分进行筛选。

#### 地理信息筛选

首先，删去以下 3 个变量：

- `BID`：该变量的空缺率高达 **83.3%**，并且提供的信息不重要，因此舍弃该变量
- `BLOCK`：该变量是一个字符串，根据理解，大致是街区的具体位置。但由于各个街区的表示方式不同不能形成统一，且字符串过长，难以数字化解析，因此舍弃该变量
- `location`：该变量是一个包含经度和纬度的二维元组，所包含的信息完全与变量 `LONGITUDE` 和 `LATITUDE` 重合，属于冗余变量，因此舍弃该变量

对剩下的变量两两之间计算皮尔逊相关系数。但考虑到部分变量为字符串类型，现对其进行转化，将字符串类型成整数：

- `NEIGHBORHOOD_CLUSTER`：该变量表示 what neighborhood cluster the case belongs to，格式为 cluster + number，例如 'cluster 21'，对该变量的处理是直接提取出 cluster 的编号

- `VOTING_PRECINCT`：该变量的格式与 `NEIGHBORHOOD_CLUSTER` 类似，例如 'precinct 75'，处理方式相同，提取出 precinct 对应的编号即可

- `ANC`：该变量是一种地理信息，格式为一位数字加一个大写字母，如 '5E'。由于 ASCII 码最大值为 256，因此采用将字符串看作 256 进制，然后根据对应的 ASCII 码转通过计算换成数字。函数的代码如下：

  ```python
  def ANC_convert(s):
      try:
          ascii_val = [ord(char) for char in s]
          return 256 * ascii_val[0] + ascii_val[1]
      except ValueError:
          return 0
  ```

- `sector`：该变量的处理方式和 `ANC` 同理，函数代码如下：

  ```python
  def sector_convert(s):
      try:
          ascii_val = [ord(char) for char in s]
          return 256 * 256 * ascii_val[0] + 256 * ascii_val[1] + ascii_val[2]
      except ValueError:
          return 0
  ```

- `BLOCK_GROUP`：该变量的形式为 'XXXXXX XX'，每个 X 代表一位数字，其中前 6 位数字与 `CENSUS_TRACT` 属性完全一致，因此只取出最后两位转化为数字，即完成了字符串到整数类型的转化

**注意，以上转化时，会将缺失值直接转化成 0，但由于缺失值均较少（最多不超过 2%），因此不会对下面计算相关性产生太大的影响。**

对 13 个变量进行两两之间的相关性分析，计算两两之间的皮尔逊相关系数，核心代码如下：

```python
df = pd.read_csv(path)
correlation_matrix = df.corr(method='pearson')
```

得到的结果如下图，颜色越深代表相关性越强：

<img src=".\\pic\\task2\\2_Corr_Matrix_Heatmap.png" width=500>



我们认为相关系数大于 0.95 为相关性极强，为冗余变量，因此有以下 4 组冗余变量

- `LONGITUDE` 和 `XBLOCK` 
- `LATITUDE` 和 `YBLOCK` 
- `ANC` 和 `WARD` 
- `PSA`，`sector` 和 `DSITRICT` 

综合变量类型，变量含义，包含信息量与分析的难易度，分别选择 `LONGITUDE`, `LATITUDE`, `ANC`, `PSA` 作为代表变量，其余作为冗余变量舍弃。

#### 时间信息筛选

观察到时间信息有以下特点：

- `END_DATE`, `START_DATE`, `YEAR`, `REPORT_DATE` 存在较大的信息重复
- 每个犯罪记录的 `END_DATE` 和 `START_DATE` 相差很小，也与 `REPORT_DATE` 几乎重合
- `END_DATE` 存在数据缺失，而 `START_DATE` 和 `REPORT_DATE` 没有数据缺失

过于具体的时间没有太大的意义，因此选择 `START_DATE` 和 `SHIFT` 代表时间信息，其余变量冗余舍弃。

#### 犯罪信息筛选

从 8 个有关犯罪信息的变量中，可以发现：

- `offense-text` 与 `OFFENSE` 的内容是完全一致的
- `offensekey` 仅为 `offensegroup` 和 `OFFENSE` 的直接拼接，为冗余变量
- `OCTO_RECORD_ID` 为 `CCN` 后面拼接字符串 '-01'，因此为冗余变量

根据以上，选择 `offensegroup` , `OFFENSE`, `ucr-rank`, `CCN`, `METHOD` 作为描述犯罪信息的变量

 

### 2.3 预处理

#### 缺失值处理

统计筛选过后选出的变量对应的缺失率（如下表），发现缺失率均较低，且缺失数据由于数据类型及数据含义等原因不易于填补，因此采用将含有缺失列的数据删去的方式。

| Attribute            | Missing Rate | Attribute       | Missing Rate | Attribute | Missing Rate |
| -------------------- | ------------ | --------------- | ------------ | --------- | ------------ |
| NEIGHBORHOOD_CLUSTER | 1.22%        | VOTING_PRECINCT | 0.02%        | OFFENSE   | 0            |
| CENSUS_TRACT         | 0.27%        | ANC             | 0            | METHOD    | 0            |
| LONGITUDE            | 0            | START_DATE      | ≈ 0          | ucr-rank  | 0            |
| LATITUDE             | 0            | SHIFT           | 0            | CCN       | 0            |
| PSA                  | 0.05%        | offensegroup    | 0            |           |              |

#### 原子性拆分

删去缺失值的同时，为了使数据耦合度更低，进行以下处理：

- 考虑到 `ANC` 包含一个数字和一个字母，可能是某种编号，但为了使数据更符合原子性，我们将这一变量解耦合，拆分成 `ANC1` 和 `ANC2` ，`ANC1` 为第一位数字，`ANC2` 为第二位字母映射到 0 到 25 的值。
- `START_DATE` 精确到秒，然而只需要年月日信息，将变量解耦合拆分成 `YEAR`, `MONTH`, `DAY` 并转化为整数。

#### 哑变量设置

由于 `SHIFT`, `OFFENSE`, `OFFENSE_GROUP`, `METHOD` 均为类别变量，将它们强制编码成数字并没有实际意义，因此采用哑变量的方式来处理这 4 个变量。

进行数据预处理的核心代码如下：

```python
df = df.dropna()

df['NEIGHBORHOOD_CLUSTER'] = df['NEIGHBORHOOD_CLUSTER'].astype(str)
df['NEIGHBORHOOD_CLUSTER'] = df['NEIGHBORHOOD_CLUSTER'].apply(NEIGHBORHOOD_CLUSTER_convert)
df['VOTING_PRECINCT'] = df['VOTING_PRECINCT'].astype(str)
df['VOTING_PRECINCT'] = df['VOTING_PRECINCT'].apply(VOTING_PRECINCT_convert)

df['ANC'] = df['ANC'].astype(str)
df['ANC1'] = df['ANC'].apply(anc1)
df['ANC2'] = df['ANC'].apply(anc2)
df['START_DATE'] = df['START_DATE'].astype(str)
df['YEAR'] = df['START_DATE'].apply(year_cal)
df['MONTH'] = df['START_DATE'].apply(month_cal)
df['DAY'] = df['START_DATE'].apply(day_cal)

df = pd.get_dummies(df, columns=['SHIFT', 'OFFENSE_GROUP', 'OFFENSE', 'METHOD'])
```



## 3. Correlation between Different Features

在对`SHIFT`, `OFFENSE`, `OFFENSE_GROUP`, `METHOD`这四个类别进行热编码处理之后，我们即可对这四个属性进行相关性分析。首先我们直接计算这些属性的相关性矩阵，并绘制热图。

计算与绘制的核心代码如下：

```python
index = df.columns.get_loc('SHIFT_day')
    CA_df = df.iloc[:, index :]

    correlation_matrix = CA_df.corr()
    correlation_matrix=correlation_matrix.abs()
    correlation_matrix=correlation_matrix.round(2)

    Heatmap=sns.heatmap(correlation_matrix, annot=True, cmap='Reds', vmin=0, vmax=1,annot_kws={'fontsize': 6})
    plt.title('Crime Info Correlation Matrix Heatmap  shift-offense-method')
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.show()
```

<img src=".\\pic\\task2\\2.2_Corr_crime_Matrix_Heatmap.png" width=600>

得到的结果如上图，由于我们是通过热编码对四类数据进行预处理，四类数据中的每一种可能的取值都会形成新的一列，故我们在观察热图时只针对从不同属性中独立编码得到的数据之间的关系。由于热编码的特点，同一属性中热编码得到的不同分类之间一定是呈负相关的，这与我们热图表现的结果一致。观察热图得到的部分信息如下：

- 暴力犯罪团体的活动集中在夜晚，而财产犯罪团体的活动集中在白天
- 杀人案件在夜晚更为频繁，其中枪杀的占比要高于刀，而这两者最为常见
- 暴力犯罪往往采用枪械，刀等攻击性较强的武器，而财产类犯罪则更偏好其他武器
- 抢劫相较于其他我们通常认识的财产类犯罪拥有更强的暴力属性，甚至超过了谋杀、纵火等我们通常认为的暴力犯罪
- 盗窃案件与时间的关系并不大

注意到纵火与盗窃相较于时间的相关性系数都很小，我们针对这两种犯罪进行更细致的分析

<div align="center">
    <img src=".\\pic\\task2\\2.2_arson_bar.png" alt="" width="270">
    <img src=".\\pic\\task2\\2.2_Burglary_bar.png" alt="" width="270">
</div>

上图为纵火案件与盗窃案件发生时段的柱状图，可以看到在 `day` 和 `evening` 盗窃事件的数量相差不多，而在 `midnight` 发生的盗窃事件明显少于 `day` 和 `evening` 。然而在 `midnight` ，总犯罪事件的发生也少于`day` 和 `evening`，故我们得到的相关系数是具有合理性的。而对于纵火案件，我们注意到纵火案件在三种犯罪时间中发生的次数几乎一致，但结合总发生次数，我们可以认为在`midnight`发生纵火案件的占比相较于`day` 和 `evening`更大 。热图中反映的情况可能是因为这类犯罪整体占比较低，导致对应的所有系数均偏小。

接下来我们分析不同属性之间的相关系数是否会随着年份不同发生变化，我们挑选出不同年份的犯罪信息并计算相关系数矩阵、绘制热图。以下为 2008 到 2021 年共 14 年，每年的相关系数矩阵对应的图像

<div align="center">
    <img src=".\\pic\\task2\\2.2_Crime Info Correlation Matrix Heatmap  shift-offense-method in 2008.png" alt="" width="190">
    <img src=".\\pic\\task2\\2.2_Crime Info Correlation Matrix Heatmap  shift-offense-method in 2009.png" alt="" width="190">
    <img src=".\\pic\\task2\\2.2_Crime Info Correlation Matrix Heatmap  shift-offense-method in 2010.png" alt="" width="190">
</div>

<div align="center">
    <img src=".\\pic\\task2\\2.2_Crime Info Correlation Matrix Heatmap  shift-offense-method in 2011.png" alt="" width="190">
    <img src=".\\pic\\task2\\2.2_Crime Info Correlation Matrix Heatmap  shift-offense-method in 2012.png" alt="" width="190">
    <img src=".\\pic\\task2\\2.2_Crime Info Correlation Matrix Heatmap  shift-offense-method in 2013.png" alt="" width="190">
</div>

<div align="center">
    <img src=".\\pic\\task2\\2.2_Crime Info Correlation Matrix Heatmap  shift-offense-method in 2014.png" alt="" width="190">
    <img src=".\\pic\\task2\\2.2_Crime Info Correlation Matrix Heatmap  shift-offense-method in 2015.png" alt="" width="190">
    <img src=".\\pic\\task2\\2.2_Crime Info Correlation Matrix Heatmap  shift-offense-method in 2016.png" alt="" width="190">
</div>

<div align="center">
    <img src=".\\pic\\task2\\2.2_Crime Info Correlation Matrix Heatmap  shift-offense-method in 2017.png" alt="" width="190">
    <img src=".\\pic\\task2\\2.2_Crime Info Correlation Matrix Heatmap  shift-offense-method in 2018.png" alt="" width="190">
    <img src=".\\pic\\task2\\2.2_Crime Info Correlation Matrix Heatmap  shift-offense-method in 2019.png" alt="" width="190">
</div>

<div align="center">
    <img src=".\\pic\\task2\\2.2_Crime Info Correlation Matrix Heatmap  shift-offense-method in 2020.png" alt="" width="190">
    <img src=".\\pic\\task2\\2.2_Crime Info Correlation Matrix Heatmap  shift-offense-method in 2021.png" alt="" width="190">
    <img src=".\\pic\\task2\\.png" alt="" width="190">
</div>


2008-2021年，大部分属性相关性并无明显变化，但仍然可以观察到

- 总体而言暴力犯罪集团的活动始终在夜晚更为常见，但在2016-2020年其活动的与夜晚的相关性有所下降
- 2008-2020年，抢劫事件与夜晚的相关性几乎逐年增长，可以认为犯罪团体变得更加热衷于在夜晚实施抢劫活动
- 2008-2021年，暴力犯罪团体与枪械的相关性在升高，同时与刀等锐器的相关性呈波动态势，可以认为暴力犯罪团体的武装力量有所上升。

## 4. Correlation between Crime Events and Space and Time

犯罪的时间有 3 种，分别为 `day`, `evening`, `midnight`；犯罪的事件种类共 9 种，分别为 `arson`, `assault w/dangerous weapon`, `burglary`, `homicide`, `motor vehicle theft`, `robbery`, `sex abuse`, `theft f/auto`, `theft/other` 

下表为 3 个时间的犯罪记录数量，可以看到在 `day` 和 `evening` 犯罪事件的数量相差不多，而在 `midnight` 发生的犯罪事件明显少于 `day` 和 `evening` 

| Shift           | day    | evening | midnight |
| --------------- | ------ | ------- | -------- |
| Number of Crime | 167077 | 189015  | 86101    |

对于 `midnight` 时段，统计每个 cluster 发生最多的犯罪类型，计算部分的核心代码如下

```python
df = pd.read_csv('DC_Crime_Preprocessed.csv')
crime_counts = df.groupby('NEIGHBORHOOD_CLUSTER')[crime_columns].sum()
most_common_crime = crime_counts.apply(lambda x: x.idxmax(), axis=1)
```

<img src=".\\pic\\task2\\3_20.png" width=400>

得到的结果如上图，观察可得

- 偷窃事件发生的最频繁
- 在华盛顿的中北部抢劫事件也发生较多
- 在华盛顿的东部和南部危险武器袭击事件较多

<div align="center">
    <img src=".\\pic\\task2\\3_18.png" alt="" width="270">
    <img src=".\\pic\\task2\\3_19.png" alt="" width="270">
</div>

观察 `day` 和 `evening` 时段，容易发现，整个华盛顿 DC 每个 cluster 最常见的犯罪类型都是偷窃，但在晚上偷窃载具较为普遍。

对比 `midnight` 时段，总结并解释现象

- 结合犯罪记录数量，白天的犯罪并没有夜晚多，而白天和夜晚均为偷窃事件最常见
- 白天的载具很多都投入了使用，而夜晚处于闲置，因此夜晚更易发生载具偷窃
- 在午夜，人员活动稀少，因此抢劫和袭击更易发生，与统计数据相吻合



## 5. Number of Crime & Geographical Districts

### 5.1 Total Number of Crime

<img src=".\\pic\\task2\\3_01.png" width=400>

上图为华盛顿特区每年所记录的犯罪数量随年份变化的柱状图，有如下几点规律

- 2021 年数量远少于其他年份，原因是数据集只包含 2021 年的前几个月
- 2008 到 2019 年犯罪记录数量有波动，但整体变化不大，基本维持在平均值上下
- 2020 年的犯罪记录数量有较为明显的下降，可能由于治安等方面的提升导致



### 5.2 Number of Crime & Cluster

<img src=".\\pic\\task2\\3_02.png" width=800>

华盛顿特区划分为 46 个 cluster，编号分别为 1 到 46. 上面的柱状图统计了每个 cluster 在 2008 到 2021 年记录的犯罪数量。

- 总体来看，cluster 之间的记录数量方差很大
- cluster 40 到 46 的记录数量可以近似于没有

但由于犯罪数量和 cluster 面积大小也有一定关联，因此我们分析每个 cluster 的犯罪记录密度。以下为 2008 到 2021 年共 14 年，每年各个 cluster 的犯罪密度示意图

<div align="center">
    <img src=".\\pic\\task2\\3_03.png" alt="" width="190">
    <img src=".\\pic\\task2\\3_04.png" alt="" width="190">
    <img src=".\\pic\\task2\\3_05.png" alt="" width="190">
</div>

<div align="center">
    <img src=".\\pic\\task2\\3_06.png" alt="" width="190">
    <img src=".\\pic\\task2\\3_07.png" alt="" width="190">
    <img src=".\\pic\\task2\\3_08.png" alt="" width="190">
</div>

<div align="center">
    <img src=".\\pic\\task2\\3_09.png" alt="" width="190">
    <img src=".\\pic\\task2\\3_10.png" alt="" width="190">
    <img src=".\\pic\\task2\\3_11.png" alt="" width="190">
</div>

<div align="center">
    <img src=".\\pic\\task2\\3_12.png" alt="" width="190">
    <img src=".\\pic\\task2\\3_13.png" alt="" width="190">
    <img src=".\\pic\\task2\\3_14.png" alt="" width="190">
</div>

<div align="center">
    <img src=".\\pic\\task2\\3_15.png" alt="" width="190">
    <img src=".\\pic\\task2\\3_16.png" alt="" width="190">
    <img src=".\\pic\\task2\\.png" alt="" width="190">
</div>

可以观察到

- 犯罪记录多集中在城市中心，犯罪记录密度明显大于四周
- 城市边缘几乎没有记录，而该区域对应的 cluster 编号为 40 到 46

针对编号为 40 到 46 的 cluster 进行深入研究，加入华盛顿 DC 的建筑分布及水域分布，并标记出记录的犯罪事件，得到的结果如下图

<img src=".\\pic\\task2\\3_17.png" width=500>

- 犯罪多分布在大型建筑或水域周围
- 大型建筑更可能是一些公共设施，如音乐厅，博物馆等，而非居民住宅区
- 发生在这类区域的犯罪数量少于居民区，结果较为符合常理








## 7. Housing Price Data Preprocessing

### 7.1 属性分类

`DC_Properties.csv` 中共有 48 个属性，每个属性分别带有一些方面的信息，现在，我们仿照在犯罪属性对数据进行的分类，将这 48 个属性分成 3 类，分别为地理位置相关、时间相关、房屋相关的属性，分别用 G、T、F 标记。分类的结果如下表：

| Attribute | Category | Attribute | Category | Attribute         | Category | Attribute       | Category | Attribute          | Category |
| --------- | -------- | --------- | -------- | ----------------- | -------- | --------------- | -------- | ------------------ | -------- |
| BATHRM    | F        | STORIES   | F        | CNDTN             | F        | CMPLX_NUM       | F        | ASSESSMENT_SUBNBHD | G        |
| HF_BATHRM | F        | SALEDATE  | T        | EXTWALL           | F        | LIVING_GBA      | F        | CENSUS_TRACT       | G        |
| HEAT      | F        | PRICE     | F        | ROOF              | F        | FULLADDRESS     | G        | CENSUS_BLOCK       | G        |
| AC        | F        | QUALIFIED | F        | INTWALL           | F        | CITY            | G        | WARD               | G        |
| NUM_UNITS | F        | SALE_NUM  | F        | KITCHENS          | F        | STATE           | G        | SQUARE             | G        |
| ROOMS     | F        | GBA       | F        | FIREPLACES        | F        | ZIPCODE         | G        | X                  | G        |
| BEDRM     | F        | BLDG_NUM  | F        | USECODE           | F        | NATIONALGRID    | G        | Y                  | G        |
| AYB       | T        | STYLE     | F        | LANDAREA          | F        | LATITUDE        | G        | QUADRANT           | G        |
| YR_RMDL   | T        | STRUCT    | F        | GIS_LAST_MOD_DTTM | T        | LONGITUDE       | G        |                    |          |
| EYB       | T        | GRADE     | F        | SOURCE            | F        | ASSESSMENT_NBHD | G        |                    |          |

特别的，由于我们需要根据房屋属性以及地区所存在的犯罪属性对房价进行预测，在数据处理时，我们直接删去所有缺失`PRICE` 这一属性的行。

另外，我们注意到一个关键的属性：`SOURCE` 。该属性表示房屋类型，有两个不同的取值：

- `Residential`：该取值表示这一房屋为住宅，不存在`CMPLX_NUM`与`LIVING_GBA`等属性。
- `Condominium`：该取值表示这一房屋为公寓，不存在`STYLE`等属性。

故我们根据`SOURCE` 将房屋分两类分别进行属性分类，关键代码如下：

```python
df = pd.read_csv('..\\..\\origin_material\\DC_Properties.csv')
df = df.dropna(subset=['PRICE'])

df_residential = df[df['SOURCE'] == 'Residential']

df_condominium = df[df['SOURCE'] == 'Condominium']

new_order = ['FULLADDRESS','CITY','STATE','ZIPCODE', 'NATIONALGRID', 'LONGITUDE', 'ASSESSMENT_NBHD', 'ASSESSMENT_SUBNBHD', 'CENSUS_TRACT', 'CENSUS_BLOCK', 'WARD','SQUARE','X','Y','QUADRANT']
df11 = df_condominium.reindex(columns=new_order)
df11.to_csv('..\\..\\data\\task4\\DC_Properties_condominium_G.csv', index=True)
df12=df_residential.reindex(columns=new_order)
df12.to_csv('..\\..\\data\\task4\\DC_Properties_residential_G.csv', index=True)

new_order = ['AYB', 'YR_RMDL', 'EYB', 'SALEDATE', 'GIS_LAST_MOD_DTTM']
df21 = df_condominium.reindex(columns=new_order)
df21.to_csv('..\\..\\data\\task4\\DC_Properties_condominium_T.csv', index=True)
df22=df_residential.reindex(columns=new_order)
df22.to_csv('..\\..\\data\\task4\\DC_Properties_residential_T.csv', index=True)

new_order = ['BATHRM', 'HF_BATHRM', 'HEAT', 'AC', 'NUM_UNITS', 'ROOMS', 'BEDRM', 'STORIES','PRICE','QUALIFIED','SALE_NUM',
             'GBA','BLDG_NUM','STYLE','STRUCT','GRADE','CNDTN','EXTWALL','ROOF','INTWALL','KITCHENS',' FIREPLACES','USECODE','LANDAREA','SOURCE','CMPLX_NUM','LIVING_GBA']
df31 = df_condominium.reindex(columns=new_order)
df31.to_csv('..\\..\\data\\task4\\DC_Properties_condominium_F.csv', index=True)
df32=df_residential.reindex(columns=new_order)
df32.to_csv('..\\..\\data\\task4\\DC_Properties_residential_F.csv', index=True)
```



### 7.2 数据预处理

注意到数据的维数很大，且其中包含大量string类型的数据，因此需要对数据进行预处理。首先，为了减小数据的维度，考虑先对数据的含义以及数据间的相关性进行分析，对该部分进行筛选。

#### 时间信息筛选

首先，删去以下 1 个变量：

- `GIS_LAST_MOD_DTTM`：该变量表示数据的最后修改日期，统一为 ‘2018-07-22 18:01:43’，可以认为是无关变量。

对剩下的变量进行转化：

- `SALEDATE`：该变量表示对应房屋最近一次售卖发生的时间，变量的形式为‘XXXX/XX/X 0:00’，为了简化问题，只保留其中的`YEAR`对应的属性。



#### 空间信息筛选

从与空间有关的16个变量中，可以发现：
- `X` 与 `LONGITUDE` ，`Y` 与 `LATITUDE`虽然取值不同，但是现实意义是完全一样的，都表示房屋的经纬度信息。为减小模型的复杂程度，我们选择删去前两者。
- 对于住宅，所有数据的`STATE` 与 `CITY` 对应的取值都是一致的，均为`WASHINGTON`，`DC` ，这是因为我们取样的区域限制决定的，故这部分属性对结果不具有影响。而公寓没这两项变量没有取值。综合考虑，我们选择删去这两个属性。
- 住宅的`CENSUS_BLOCK` 的前半部分一定为 `CENSUS_TRACT` ，因此后者为冗余变量。而公寓的`CENSUS_BLOCK`属性空缺，仅存在`CENSUS_TRACT`属性。

结合数据处理的难易程度与前述对犯罪地理位置的分析，最终选取 `LONGITUDE` ， `LATITUDE`，`WARD`与`QUADRANT`作为需要保留的数据，其中`WARD`的取值只需要直接保留数字即可，而`QUADRANT`直接选择独热编码取得八个独立的子属性。



#### 房屋属性信息筛选

分别考虑住宅与公寓。首先统计缺失值大于50%的属性，由于这部分数据缺失值较多，难以对预测价格时提供有效的帮助，故选择舍去，最终在住宅中得到23个有效属性，公寓中得到14个有效属性：
- `AC`表示有无制冷，取值为`Y`或`N`，可以直接映射为1或0。

- `QUALIFIED`表示是否具备资格，取值为`Q`或`U`，可以直接映射为1或0。

- `STYLE`表示楼层，楼层数若为半层则可能还具有属性`Fin`。我们选择提取楼层数据，并通过热编码统计`Fin`的信息。不符合该数据组成规律的数据数量小于千分之一，在此忽略不计。

- `HEAT`表示有无供暖，`EXTWALL`，`ROOF`，`INTWALL`分别表示外墙，天花板以及内墙材料，`STRUCT`表示结构，这部分信息种类过于繁多，且难以直接映射处理。为了避免最终的数据维数过大，选择直接舍去。

- `GRADE`与`CNDTN`均为房屋评价，分别有13种以及7种不同的评价，可以直接映射到数字，数字越大表示评价越好。对于住宅等级评价，在我们保留的数据中，有共计1.2%的评价为`Exceptional-D`，`Exceptional-C`，`Exceptional-A`以及`Exceptional-B`，这部分评价含义理解困难，故将其均映射为0，避免其对结果拟合的影响。

这样，便得到了初步处理完毕的数据。再通过计算相关性矩阵制作热图大部分数据处理的方法与第二题中一致，对于变量`Fin`的处理关键代码如下：

```python
df32 = pd.read_csv('..\\..\\data\\task4\\DC_Properties_residential_F.csv')
df32['FLOOR'] = df32['STYLE'].str.extract('(\d+)') 
df32['STYLE_FIN'] = df32['STYLE'].str.replace('Fin', '1', regex=True)
df32['STYLE_FIN'] = df32['STYLE'].str.replace('Unfin', '0.5', regex=True)
df32['STYLE_FIN'] = df32['STYLE'].str.replace(r'[^0]', '0', regex=True)
df32 = df32.drop('STYLE', axis=1)
```



<img src=".\\pic\\task4\\4.1_Corr_Matrix_Heatmap_Condominium.png" width=600>

<img src=".\\pic\\task4\\4.1_Corr_Matrix_Heatmap_Residential.png" width=600>

得到的结果如上图。该图像中并不存在相关系数大于95%的数据对，因此我们认为并无冗余变量。注意到在分析公寓的属性的相关性时变量`BLDG_NUM`列的数据出现了空缺，这是因为该列的取值在公寓中均为常数1，与其他变量均无关。最后，将处理结束的几个变量拼接在一起并储存，用以接下来的回归。