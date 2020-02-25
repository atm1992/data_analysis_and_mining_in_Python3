#-*- coding: UTF-8 -*-
import numpy as np

def main():
	lst = [[1,3,5],[2,4,6]]
	print(type(lst))
	np_lst = np.array(lst)
	print(type(np_lst))
	np_lst = np.array(lst,dtype=np.float)
# dtype 的可选值为：bool,int,int8,int16,int32,int64,int128,
# uint8,uint16,uint32,uint64,uint128,float,float16/32/64,complex64/128
	print(np_lst.shape)	#几行几列
	print(np_lst.ndim)	#维度
	print(np_lst.dtype)	#数据类型
	print(np_lst.itemsize)	#每个元素所占字节的大小，float64是64位，因此是8个字节
	print(np_lst.size)	#元素的个数。元素个数乘以元素所占字节的大小（6*8=48），就是np_lst所占的字节大小
	# Some Arrays
	print(np.zeros([2,4]))	# 2行4列都是0，可以用来进行数值的初始化
	print(np.ones([3,5]))	# 3行5列都是1
	print(np.random.rand(2,4))	# 生成2行4列的随机浮点数，0到1之间
	print(np.random.rand())		# 生成一个0到1之间的随机浮点数
	print("RandInt:",np.random.randint(1,10,3))		# 随机生成3个1到10之间的整数
	print("Randn:",np.random.randn(2,4))	# 随机生成2行4列的服从标准正态分布的浮点数
	print("Choice:",np.random.choice([10,20,30,5,6,7],2))	# 从给定的list随机抽取两个数值
	print("β-Distribute:",np.random.beta(1,10,100))	# 随机生成100个1到10之间服从β分布的浮点数组成的list
	# Array Operations
	print(np.arange(1,11))	# 生成一个从1到11（不包含11）的等差数列
	print(np.arange(1,11).reshape([2,5]))	# 把上面的那个等差数列转换成2行5列的形式，最后面的那个5可以写成-1（缺省）
	lst = np.arange(1,11).reshape([2,-1])
	print(np.exp(lst))	# 以自然数e为底的指数，对每个元素进行操作
	print(np.exp2(lst))	# 以2为底的指数
	print(np.sqrt(lst))	# 开方
	print(np.sin(lst))	# 三角函数
	print(np.log(lst))	# 自然对数，以自然数e为底




if __name__ == '__main__':
    main()