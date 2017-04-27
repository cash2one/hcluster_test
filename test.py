# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.cluster.hierarchy import fcluster
import numpy as np
import pandas

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

def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		pass
	try:
		import unicodedata
		unicodedata.numeric(s)
		return True
	except (TypeError, ValueError):
		pass
	return False

def paint(Z):
	plt.figure(figsize=(50, 10))
	plt.title('Hierarchical Clustering Dendrogram')
	plt.xlabel('sample index')
	plt.ylabel('distance')
	dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
	plt.show()

titles = [line.strip().split('\t')[1] for line in open('sogou_experiment.txt')]
corpus = [line.strip() for line in open('sogou_experiment_segs.txt')]
corpus_without_number = []
for line in corpus[:100]:
	temp = []
	for l in line.split():
		if is_number(l):continue
		temp.append(l)
	corpus_without_number.append(' '.join(temp))
# 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
vectorizer = CountVectorizer(max_df=0.8,min_df=0.1)#, min_df=0.1
#该类会统计每个词语的tf-idf权值
transformer = TfidfTransformer()
# 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus_without_number))
#获取词袋模型中的所有词语
word = vectorizer.get_feature_names()
#print len(word)
#将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
weight = tfidf.toarray()
#np.savetxt('input.csv', weight, delimiter=",")
df = pandas.DataFrame(data= weight, columns=word)
print weight.shape
#df.to_csv('baidu_hotdata_input0.csv')

Z = linkage(weight, method='ward')
paint(Z) #画层次聚类过程图

# max_d = 2.5
# fancy_dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=90., leaf_font_size=12., show_contracted=True, annotate_above=10,max_d=max_d)
# plt.show()
dict_rst = {}
max_d = 2.5
clusters = fcluster(Z, max_d, criterion='distance')
#print clusters
print '距离为' + str(max_d) + '的分类结果------------------'
for i,v in enumerate(clusters):
	if v in dict_rst:
		dict_rst[v].append(i + 1)
	else:
		dict_rst[v] = []
#print clusters
count = 1
for it in dict_rst.iteritems():
	print '话题' + str(count) + '------'
	for i in it[1]:
		print titles[i-1]
	count += 1
k = 8
clusters = fcluster(Z, k, criterion='maxclust')

