# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

#生成两个集群: 集群 a 有100个数据点, 集群 b 有50个数据点:
np.random.seed(1029)
a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[100,])
b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[50,])
X = np.concatenate((a, b),)
# print X.shape
# plt.figure(figsize=(25, 10))
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()

# 建立集群关系数组
Z = linkage(X, 'ward')

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

#c, coph_dists = cophenet(Z, pdist(X))
#print Z[1]

# plt.figure(figsize=(50, 10))
# plt.title('Hierarchical Clustering Dendrogram')
# plt.xlabel('sample index')
# plt.ylabel('distance')
# dendrogram(Z, leaf_rotation=90., leaf_font_size=8.
# )
# plt.show()


def fancy_dendrogram(*args, **kwargs):
	max_d = kwargs.pop('max_d', None)
	if max_d and 'color_threshold' not in kwargs:
		kwargs['color_threshold'] = max_d
	annotate_above = kwargs.pop('annotate_above', 0)

	ddata = dendrogram(*args, **kwargs)

	if not kwargs.get('no_plot', False):
		plt.title('Hierarchical Clustering Dendrogram (truncated)')
		plt.xlabel('sample index or (cluster size)')
		plt.ylabel('distance')
		for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
			x = 0.5 * sum(i[1:3])
			y = d[1]
			if y > annotate_above:
				plt.plot(x, y, 'o', c=c)
				plt.annotate("%.3g" % y, (x, y), xytext=(0, -5), textcoords='offset points', va='top', ha='center')
		if max_d:
			plt.axhline(y=max_d, c='k')
	return ddata

max_d = 16
fancy_dendrogram(Z, truncate_mode='lastp', p=20, leaf_rotation=90., leaf_font_size=12., show_contracted=True, annotate_above=10,max_d=max_d)
plt.show()

#如果我们已经通过树状图知道了最大临界值，我们可以通过以下代码获得每个实验样本所对应的集群下标：
from scipy.cluster.hierarchy import fcluster
max_d = 50
clusters = fcluster(Z, max_d, criterion='distance')
print clusters

#如果我们已经知道最终会有2个集群，我们可以这样获取集群下标：
k = 2
clusters = fcluster(Z, k, criterion='maxclust')
print clusters

#可视化集群
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='prism')
plt.show()