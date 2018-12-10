
# # 投掷硬币 vs 正态分布与假设检验

0.说在前面 1.投掷硬币 vs 正态分布与假设检验  2.作者的话


# 0.说在前面

基于大数定律和中心极限定律，使用正态分布拟合二项分布，并且计算某一个变量位于特定区间的概率以及相关假设检验的置信区间。

# 1.投掷硬币 vs 正态分布与假设检验
### 正态分布的概率累积分布函数
以下函数 normal_distribution_cdf 是正态分布的概率累积分布函数。


```python
import math
def normal_distribution_cdf(x, mu = 0, sigma = 1):
    '''
    正态分布概率累积分布函数
    
    参数：
        x: 数值型，正态分布中自变量的某个取值
        mu: 数值型，正态分布的均值，默认值为0
        sigma: 数值型，正态分布的标准差，默认值为1
        
    返回值：
        数值型，正态分布中落于小于x区间的概率
    '''
    return (1 + math.erf((x-mu) / math.sqrt(2) / sigma)) / 2
```

### 定义函数计算变量位于特定区间的概率
基于正态分布概率累积分布函数 normal_distribution_cdf, 定义以下四个函数，分别计算变量位于：<br>
1. 小于阈值的概率
2. 大于等于阈值的概率
3. 大于某个较小阈值并且小于某个较大阈值的概率
4. 小于某个较小阈值或者大于某个较大阈值的概率


```python

# normal_pro_below 函数用以计算变量小于阈值的概率
# 请基于 normal_distribution_cdf 定义 normal_pro_below

def normal_pro_below(bound, mu = 0, sigma =1):
    return normal_distribution_cdf(bound, mu, sigma)

# normal_pro_above 函数用以计算变量大于阈值的概率
def normal_pro_above(bound, mu = 0, sigma =1):
    '''
    计算变量大于阈值的概率
    
    参数：
        bound: 数值型，阈值
        mu: 数值型，正态分布的均值，默认值为0
        sigma: 数值型，正态分布的标准差，默认值为1
        
    返回值：
        数值型，正态分布中落于大于阈值的概率
        
    '''
    # 请基于 normal_distribution_cdf 定义 normal_pro_above
    
    
    return 1- normal_distribution_cdf(bound, mu, sigma)


# normal_pro_between 函数用以计算变量大于某个较小阈值并且小于某个较大阈值的概率
def normal_pro_between(lower_bound, upper_bound, mu = 0, sigma = 1):
    '''
    计算变量大于某个较小阈值并且小于某个较大阈值的概率
    
    参数：
        lower_bound: 数值型，较小阈值
        upper_bound: 数值型，较大阈值
        mu: 数值型，正态分布的均值，默认值为0
        sigma: 数值型，正态分布的标准差，默认值为1
        
    返回值：
        数值型，正态分布中落于大于某个较小阈值小于某个较大阈值的概率
    '''
    # 请基于 normal_distribution_cdf 定义 normal_pro_between
    return normal_distribution_cdf(upper_bound, mu, sigma) - normal_distribution_cdf(lower_bound, mu, sigma)


# normal_pro_between 函数用以计算小于某个较小阈值或大于某个较大阈值的概率
def normal_pro_outside(lower_bound, upper_bound, mu = 0, sigma = 1):
    '''
    计算变量小于某个较小阈值或大于某个较大阈值的概率
    
    参数：
        lower_bound: 数值型，较小阈值
        upper_bound: 数值型，较大阈值
        mu: 数值型，正态分布的均值，默认值为0
        sigma: 数值型，正态分布的标准差，默认值为1
        
    返回值：
        数值型，正态分布中落于小于某个较小阈值或大于某个较大阈值的概率
    '''
    # 请基于 normal_distribution_cdf 定义 normal_pro_outside
    return  normal_distribution_cdf(lower_bound, mu, sigma) + 1 - normal_distribution_cdf(upper_bound, mu, sigma)
```

### 正态分布计算某一个特定累计概率的相应自变量值函数
使用以下函数 inverse_normal_cdf 可以计算正态分布中某一个特定的累计概率所对应的自变量取值。<br>
normal_distribution_cdf 严格递增，以下函数使用二分查找法计算。


```python
def inverse_normal_distribution_cdf(p, mu=0, sigma=1, tolerance = 0.00001):
    '''
    计算正态分布中某一个特定的累计概率所对应的自变量取值
    
    参数：
        p: 数值型，特定概率值
        mu: 数值型，正态分布的均值，默认值为0
        sigma: 数值型，正态分布的标准差，默认值为1
        tolerance: 数值型，容错率
        
    返回值：
        数值型，与特定概率值p对应的自变量取值
    '''
    if mu != 0 or sigma != 1:
        return mu + sigma*inverse_normal_distribution_cdf(p,tolerance = tolerance)
    low_z, low_p = -10.0, 0
    up_z, up_p = 10.0, 1
    while up_z - low_z > tolerance:
        mid_z = (low_z + up_z) / 2
        mid_p = normal_distribution_cdf(mid_z)
        if mid_p < p:
            low_z, low_p = mid_z, mid_p
        elif mid_p > p:
            up_z, up_p = mid_z, mid_p
        else:
            break
            
    return mid_z

```

### 计算特定累积概率对应阈值
基于 inverse_normal_distribution_cdf 函数, 定义以下三个函数，分别计算正态分布中：<br>
1. 小于某概率即左尾阈值
2. 大于某概率即右尾阈值
3. 双尾阈值，即上下限


```python
def normal_upper_bound(pro, mu = 0, sigma = 1):
    '''
    计算正态分布中小于某概率的阈值
    
    参数：
        pro: 数值型，特定概率值
        mu: 数值型，正态分布的均值，默认值为0
        sigma: 数值型，正态分布的标准差，默认值为1
        
    返回值：
        数值型，小于特定概率值p对应的自变量取值
    '''
    # 请基于 inverse_normal_distribution_cdf 定义 normal_upper_bound
    return inverse_normal_distribution_cdf(pro, mu, sigma, tolerance = 0.00001)


def normal_lower_bound(pro, mu = 0, sigma = 1):
    '''
    计算正态分布中大于某概率的阈值
    
    参数：
        pro: 数值型，特定概率值
        mu: 数值型，正态分布的均值，默认值为0
        sigma: 数值型，正态分布的标准差，默认值为1
        
    返回值：
        数值型，大于特定概率值p对应的自变量取值
    '''
    # 请基于 inverse_normal_distribution_cdf 定义 normal_lower_bound
    # 等同于概率小于 1-pro 时的阈值
    return inverse_normal_distribution_cdf(1 - pro, mu, sigma, tolerance = 0.00001)


def normal_twosided_bounds(pro, mu = 0, sigma = 1):
    '''
    计算正态分布中双尾阈值
    
    参数：
        pro: 数值型，特定概率值
        mu: 数值型，正态分布的均值，默认值为0
        sigma: 数值型，正态分布的标准差，默认值为1
        
    返回值：
        lower_bound: 数值型，下限
        upper_bound: 数值型，上限
    '''
    # 请基于 inverse_normal_distribution_cdf 定义 normal_twosided_bounds
    # 或者调用 normal_lower_bound 及 normal_upper_bound 函数定义 normal_twosided_bounds
    # 注意：双尾部分的概率和为 1-pro，则下限左侧及上限右侧的概率分别为（1-pro) 的 1/2
    lower_bound = inverse_normal_distribution_cdf((1-pro)/2, mu, sigma, tolerance = 0.00001)
    upper_bound = inverse_normal_distribution_cdf((1+pro)/2, mu, sigma, tolerance = 0.00001)
    return lower_bound, upper_bound

```

### 投掷硬币试验
现有一枚均匀的硬币，投掷n次，正面朝上的次数记为X，每次试验都可以看做一次伯努利试验，那么X满足二项分布 B(n,p)。<br>
当进行了大量的试验，试验结果近似服从正态分布，可以使用正态分布来拟合。


```python
# 计算正态分布拟合的二项分布的均值和标准差的函数
def normal_binomial(n,p):
    '''
    计算正态分布拟合的二项分布的均值和标准差
    
    参数：
        n: 数值型，试验次数
        p: 数值型，每次伯努利试验目标事件出现的概率
        
    返回值：
        mu: 数值型，均值
        sigma: 数值型，标准差
    '''
    mu = n*p
    sigma = (n*p*(1-p))**0.5
    
    return mu, sigma
```

#### 计算变量位于某一特定区间的概率
现投掷了2000次硬币，请使用以上函数计算：

1) 出现正面朝上次数小于995次的概率


```python

mu, sigma= normal_binomial(2000,0.5)

normal_pro_below(995, mu =mu, sigma =sigma)


```




    0.41153163687906075



2) 出现正面朝上次数大于990次的概率


```python

normal_pro_above(990, mu = mu, sigma =sigma)
```




    0.6726395769907114



3) 出现正面朝上次数大于998次且小于1010次的概率


```python

normal_pro_between(998, 1010, mu = mu, sigma =sigma)

```




    0.2082745397083101



4) 出现正面朝上次数小于990或者大于1030次的概率


```python

normal_pro_outside(990, 1030, mu = mu, sigma = sigma)

```




    0.4172166704487885



### 假设检验
现有一枚硬币，投掷2000次，根据中心极限定理，伯努利随机变量的均值近似服从正态分布，<br>

1) 投掷的结果发现，有1100次正面朝上，那么我们可以估计这枚硬币是均匀的吗？


```python

# 请计算在显著性水平为0.05（传入的参数pro=0.95）的情况下，是否可以认为硬币是均匀的
# 硬币均匀的条件是: 正面朝上的概率P为0.5 
# 原假设 H0: P = 0.5, 如果0.5落于双尾上下限之间，那么不能拒绝原假设，则认为硬币是均匀的，反之拒绝原假设，认为硬币是不均匀的
# 调用函数计算阈值时使用的均值和标准差参数为本实验结果的均值和标准差
print('原假设H0: P = 0.5，硬币是均匀的')
mu = 0.5
pro=0.95
mu_1 = 1100/2000
sigma_1 = (mu_1*(1-mu_1)/2000)**0.5
lower_bound1, upper_bound1 = normal_twosided_bounds(pro, mu_1, sigma_1)

if (mu < lower_bound1 )| (mu > upper_bound1):
    print ('拒绝原假设H0: P = 0.5，硬币是不均匀的')
else: 
    print ('接受原假设H0: P =0.5，硬币是均匀的')
    

```

    原假设H0: P = 0.5，硬币是均匀的
    拒绝原假设H0: P = 0.5，硬币是不均匀的
    

1) 投掷的结果发现，有1050次正面朝上，那么我们可以估计这枚硬币是均匀的吗？


```python

# 请计算在显著性水平为0.05的情况下，是否可以认为硬币是均匀的
# 其余需要注意的点同 1）中的注释
print('原假设H0: P = 0.5，硬币是均匀的')
mu = 0.5
pro=0.95
mu_2 = 1050/2000
sigma_2 = (mu_2*(1-mu_2)/2000)**0.5
lower_bound2, upper_bound2 = normal_twosided_bounds(pro, mu_2, sigma_2)

if (mu < lower_bound2 )| (mu > upper_bound2):
    print ('拒绝原假设H0: P = 0.5，硬币是不均匀的')
else: 
    print ('接受原假设H0: P =0.5，硬币是均匀的')

```

    原假设H0: P = 0.5，硬币是均匀的
    拒绝原假设H0: P = 0.5，硬币是不均匀的
    
