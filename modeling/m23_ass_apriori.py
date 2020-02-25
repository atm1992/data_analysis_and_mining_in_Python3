#-*- coding: UTF-8 -*-

from itertools import combinations


def comb(lst):
    ret = []
    for i in range(1, len(lst) + 1):
        # 返回长度为 i 的没有重复元素的有序组合，返回结果是一个个元组。即 1项集、2项集、3项集……
        ret += list(combinations(lst, i))
    return ret


class AprLayer(object):
    d = dict()

    def __init__(self):
        self.d = dict()


class AprNode(object):
    # node是个项集（元组形式），s是由元组内的元素组成的set，size是set的长度，num为node的出现次数
    # lnk_nodes为由该node组成的高一阶项集node
    def __init__(self, node):
        self.s = set(node)
        self.size = len(self.s)
        self.lnk_nodes = dict()
        self.num = 0

    def __hash__(self):
        return hash("__".join(sorted([str(itm) for itm in list(self.s)])))

    def __eq__(self, other):
        # 比较两个node是否相同
        if "__".join(sorted([str(itm) for itm in list(self.s)])) == "__".join(sorted([str(itm) for itm in list(other.s)])):
            return True
        return False

    def isSubnode(self, node):
        return self.s.issubset(node.s)

    def incNum(self, num=1):
        self.num += num

    def addLnk(self, node):
        # dict的key值为调用__hash__方法变成排好序的string，value值为由元组内的元素组成的set
        self.lnk_nodes[node] = node.s


class AprBlk():
    def __init__(self, data):
        cnt = 0
        self.apr_layers = dict()
        # data数组长度
        self.data_num = len(data)
        for datum in data:
            cnt += 1
            # 得到所有的候选项集。[(,),(,),……]形式
            datum = comb(datum)
            # 将所有候选项集存储到自定义的数据结构AprNode中
            nodes = [AprNode(da) for da in datum]
            for node in nodes:
                # apr_layers[1].d 存储所有的1项集
                if not node.size in self.apr_layers:
                    self.apr_layers[node.size] = AprLayer()
                if not node in self.apr_layers[node.size].d:
                    self.apr_layers[node.size].d[node] = node
                # node的出现次数加1
                self.apr_layers[node.size].d[node].incNum()
            for node in nodes:
                if node.size == 1:
                    continue
                # node.s 是一个set
                for sn in node.s:
                    # 由高阶项集node生成低一阶项集node
                    sub_n = AprNode(node.s - set([sn]))
                    # 低阶项集node指向高阶项集node
                    self.apr_layers[node.size - 1].d[sub_n].addLnk(node)

    # 获取频繁项集。thd为支持度阈值
    def getFreqItems(self, thd=1, hd=1):
        freq_items = []
        # 遍历AprBlk，首先判断第一层apr_layers中的1项集是否频繁
        for layer in self.apr_layers:
            for node in self.apr_layers[layer].d:
                # 若node的出现次数小于支持度阈值thd，则不进行操作
                if self.apr_layers[layer].d[node].num < thd:
                    continue
                # 将频繁项集加入到freq_items数组，每个元素为 (项集set,项集出现次数)
                freq_items.append((self.apr_layers[layer].d[node].s, self.apr_layers[layer].d[node].num))
        # 根据频繁项集出现次数降序
        freq_items.sort(key=lambda x: x[1], reverse=True)
        # 返回最前面hd个频繁项集
        return freq_items[:hd]

    # 获取置信度。
    def getConf(self, low=True, h_thd=10, l_thd=1, hd=1):
        confidence = []
        # 遍历AprBlk，首先判断第一层apr_layers中的1项集是否频繁
        for layer in self.apr_layers:
            for node in self.apr_layers[layer].d:
                # 若node的出现次数小于h_thd，则不进行操作
                if self.apr_layers[layer].d[node].num < h_thd:
                    continue
                # 然后遍历符合要求的node所组成的高一阶node项集
                for lnk_node in node.lnk_nodes:
                    # 若高一阶node项集的出现次数小于l_thd，则不进行操作
                    if lnk_node.num < l_thd:
                        continue
                    # 计算置信度
                    conf = float(lnk_node.num) / float(node.num)
                    confidence.append([node.s, node.num, lnk_node.s, lnk_node.num, conf])

        # 根据置信度升序
        confidence.sort(key=lambda x: x[4])
        if low:
            # 返回低置信度的项集组合，即 confidence数组中的前hd个
            return confidence[:hd]
        else:
            # 返回高置信度的项集组合
            return confidence[-hd::-1]

# 关联规则类
class AssctAnaClass():
    def fit(self, data):
        self.apr_blk = AprBlk(data)
        return self

    # 获取频繁项集
    def get_freq(self, thd=1, hd=1):
        return self.apr_blk.getFreqItems(thd=thd, hd=hd)

    # 获取高置信度的项集组合
    def get_conf_high(self, thd, h_thd=10):
        return self.apr_blk.getConf(low=False, h_thd=h_thd, l_thd=thd)

    # 获取低置信度的项集组合
    def get_conf_low(self, thd, hd, l_thd=1):
        return self.apr_blk.getConf(h_thd=thd, l_thd=l_thd, hd=hd)


if __name__ == "__main__":
    data = [["牛奶", "啤酒", "尿布"], ["牛奶", "啤酒", "咖啡", "尿布"], ["香肠", "牛奶", "饼干"], ["尿布", "果汁", "啤酒"], ["钉子", "啤酒"],
            ["尿布", "毛巾", "香肠"],
            ["啤酒", "毛巾", "尿布", "饼干"]]
    print("Freq", AssctAnaClass().fit(data).get_freq(thd=3, hd=10))
    print("Conf_high", AssctAnaClass().fit(data).get_conf_high(thd=3, h_thd=3))
    print("Conf_low", AssctAnaClass().fit(data).get_conf_low(thd=3, hd=10, l_thd=2))