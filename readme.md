## 一、numpy的数组对象
numpy的n维数组对象用ndarray创建
1. 数组对象可以去掉元素间所需的循环，使得一维向量更像单个数据
2. 设置专门的数组对象，经过优化，可以提升此类运算的速度
3. 因为在科学计算中，一个维度所有的数据类型往往相同，所以数组对象区别于列表对象的一点就是数组对象采用相同的数据类型，有助于节省运算和存储的空间

```
np.array() 用于生成一个ndarray数组，ndarray在程序中的别名是array,ndarray数组print的时候输出成[]形式，元素由空格分割

import numpy as np
np.array([1,2,3])
```
两个概念：
1. 轴(axis)：数据的维度
2. 秩(rank)：轴的数量

ndarray对象的属性
1. .ndim: 秩
2. .shape: ndarray对象的尺度，对于矩阵，m行n列，结果是一个元组，即使是一维数组也返回一个元组
3. .size: m * n的值，ndarray对象的元素个数
4. .dtype: ndarray对象中元素的数据类型
5. .itemsize: ndarray对象中每个元素的大小,以字节为单位


numpy中的数据类型划分的比较精细正是为了应对科学计算。
```
ndarray中的数据类型
1. bool
2. intc: 与c语言的int类型一致，一般是int32或者int64
3. intp
4. int8: 2的8次方， [-128, 127]
5. int16: 2的16次方
6. int32
7. int64
8. uint8: [0, 255]
9. uint16
10. unit32
11. uint64
12. float16: 1位符号位,5位指数,10位尾数
13. float32： 1位符号位,8位指数,23位尾数
14. float64： 1位符号位,11位指数,52位尾数
15. complex64: 实部和虚部都是32位浮点数
16. complex128: 实部和虚部都是64位浮点数
```
特殊：
ndarray数组可以由非同质对象构成，这时候ndarray数组中的元素就看做是一个对象，非同质ndarray无法发挥numpy的优势，应该避免使用

## ndarray数组的创建与变换

### 创建方法
1. 从list和tuple等类型创建(可以是单个，也可以是两者的混合)x=np.array([[1,2],[9,8]])
print(x)
[[1 2]
 [9 8]]
x=np.array([[1,2],[9,8], [0.1, 0.2]])
print(x)
[[ 1.   2. ]
 [ 9.   8. ]
 [ 0.1  0.2]]

2. 使用numpy中的几个函数，如arange, ones, zeros等，这些函数默认元素类型采用浮点型
* .arange(n)  类似于range()
* .ones(shape)
* .ones_like(a)   根据数组a的形状生成全1的数组
* .zeros(shape)
* .zeros_like(a)   根据数组a的形状生成全0的数组
* full(shape, val)
* full_like(a, val)  根据数组a的形状生成全val的数组
* eye(n)  创建一个正方的n*n的单位矩阵
* linespace() 根据起止数据等间距地填充数据，形成数组
* concatenate()  将两个或多个数组合并成一个新数组,里面的参数是一个元组
```
np.ones((2,3))
Out[18]: 
array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.]])
默认用的是float作为数据类型,可以用dtype参数指定
np.ones((2,3), dtype=np.int64)
Out[23]: 
array([[1, 1, 1],
       [1, 1, 1]], dtype=int64)
np.zeros((2,3,4))
Out[24]: 
array([[[ 0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.]],
       [[ 0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.]]])
理解(2,3,4) 每个数字对应的含义
a = np.linspace(1,10,4)
a
Out[30]: array([  1.,   4.,   7.,  10.])
a = np.linspace(1,10,4, endpoint=False)
a
Out[32]: array([ 1.  ,  3.25,  5.5 ,  7.75])
endpoint=False表示最后一个数字不包含，相当于生成5个数字，最后一个数字剔除掉即可
```
3. 从字节流(raw bytes)中创建
4. 从文件中读取特定格式创建

### ndarray数组的变换
对于创建后的ndarray数组，可以对其进行维度变换和元素类型变换
* reshape(shape)  不修改原数组
* resize()   修改原数组
* swrapaxes(ax1, ax2)
```
a = np.array([[1,2],[3,4]])
a
Out[34]: 
array([[1, 2],
       [3, 4]])
a.swapaxes(0,1)
Out[35]: 
array([[1, 3],
       [2, 4]])
```
* flatten()  对数组进行降维，返回折叠后的一维数组，原数组不变
* astype(new_type)  改变数据类型，会复制一遍原数组然后修改数据类型，即使要修改的数据的类型和原来的数据类型一致，可以用这个来拷贝数据
* tolist()

## 数组的操作之切片和索引
索引： 获取数组中特定位置元素的过程
切片： 获取数组元素子集的过程

### 一维数组的索引和切片
```
a = np.array([1,2,3])
a[0]
Out[43]: 1
a[1:3]
Out[44]: array([2, 3])
a[1:3:2]
Out[45]: array([2])
```
### 多维数组的索引
```
a = np.arange(24).reshape((2,3,4))
a
Out[47]: 
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],
       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
a[1,2,3]
Out[48]: 23
a[-1,-2,-3]
Out[49]: 17
```
切片是为了获取某个元素，元素有几维，那么索引取值里面需要有几个值
### 多维数组的切片
```
a = np.arange(24).reshape((2,3,4))
a
Out[55]: 
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],
       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
a[:,1,-3]    # 第一维不管，意思是两个维度都要取值，第二个维度取第2个元素，第三个维度取倒数第三个元素，这样取得的多个值组合在一起返回给我们

Out[56]: array([ 5, 17])
a[:,1:3,:]   # 每一个维度的切片方法和一维相同，这里可以看出把向量级别当做单个元素处理
Out[57]: 
array([[[ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],
       [[16, 17, 18, 19],
        [20, 21, 22, 23]]])

a[:,:,::2]
Out[58]: 
array([[[ 0,  2],
        [ 4,  6],
        [ 8, 10]],
       [[12, 14],
        [16, 18],
        [20, 22]]])
```

## ndarray数组的运算
###数组和标量的运算  
作用于数组的每一个元素
### 一元函数
对ndarray中的数据执行元素级别的运算，不会改变原数组
* np.abs(x)
* np.sqrt(x)
* np.square(x)
* np.log(x)  np.log10(x)  np.log2(x)
* np.ceil(x)  np.floor(x)
* np.rint(x)
* np.modf(x) -- 把数组各元素的小数和整数部分以两个独立数组形式返回
* np.exp(x)
* np.sign(x)
* np.cos(x)  np.sin(x) ...

### 二元函数
操作两个数组对象
1. `+ - * / **`
2. np.fmax()  np.fmin() np.maximum(x,y)  np.minimun(x,y)
3. np.mod(x,y)
4. np.copysign(x,y)
5. `> < >= <= == !=  --` 产生布尔型数组

## 数据的csv文件存取
### 把array存入csv文件
np.savetxt(frame, array, fmt="%.18e",delimiter=None)

### 从csv文件还原成ndarray对象
np.loadtxt(frame, dtype=np.float, delimiter=None, unpack=False)
unpack: 如果是True,读入属性将分别写入不同变量，体现在二维数组的情况下就是原二维数组维度的数据颠倒了。

### numpy中的csv文件的局限性
只能适用于一维或者二维数组

## 多维数据的存取
csv可以存取一维或者二维数组数据，那么多维数据呢？多维数据存入文件只能是二进制
### 多维数据的存入
a.tofile(frame, sep='', format='%s')     a是ndarray对象
a.tofile() 之后是一系列从前到后排列的一维数据
### 多维数据的取出
np.fromfile(frame, dtype=float,count=-1,sep='')
count=-1表示取出所有文件内容

np.fromfile()得到的也是一维，要想得到原数组对象，需要reshape(),而reshape()需要知道原数组的形状。 
所以，a.tofile()和np.fromfile() 常常是配合使用

## numpy中的便捷文件存取
np.save(frame, array)    得到的文件以npy为文件拓展名
np.load(frame)

## numpy中的随机数函数
使用方式： np.random.*

```
1. rand(d0, d1, d2)   根据d0-dn创建随机数数组，浮点数，[0,1),均匀分布  rand(2,3) 表示2行3列
2. randn(d0, d1, d2)  根据d0-dn创建随机数数组，标准正态分布
3. randint(low[,high,shape])  根据shape创建随机数数组，范围是[low, high)   randint(10,20,(3,4))
4. seed(s)   随机数种子，s是给定的种子值
   用法：先 np.random.seed(10) 然后 np.random.randint(10,20,(3,4))
```

```
1. shuffle(a)  根据a的第一轴打乱后的顺序重排a(a只有第0轴则打乱第0轴)，会改变原数组
2. permutation(a) 效果基本同上，只不过不会改变原数组
3. choice(a,[,size,replace,p]) 从一维数组a中以p概率抽取元素，形成size形状新数组，replace表示是否可以重用元素，默认False(不能重用元素)
```

```
uniform(low,high,size)    产生具有均匀分布的数组，low起始值，high结束值，size是尺寸
normal(loc, scale, size)  产生具有正态分布的数组，loc是均值，scale是标准差
poisson(lam, size)		  产生具有泊松分布的数组，lam是随机事件发生的概率
```

## numpy的统计函数
```
1. sum(a, axis=None) 
2. mean(a, axis=None)
3. average(a, axis=None, weights=None)   加权平均值
4. std(a, axis=None)    标准差
5. var(a, axis=None)	方差

axis=None 是统计函数的标配
```

```
1. min(a)  max(a)
2. argmin(a)  argmax(a)  最大值和最小值降成一维后的下标 
3. unravel_index(index,shape)  根据shape把一维下标index转换成多维下标
4. ptp(a)  a最大值和最小值的差
5. median(a)  中位数
```

```
np.gradient(f)  计算数组f中元素的梯度，当f是多维时，返回多个维度的梯度
```