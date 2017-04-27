# coding:utf-8
import sys

import jieba

reload(sys)
sys.setdefaultencoding("utf-8")

import urllib2
import json
import re

#remove 标点和特殊字符
etlregex = re.compile(ur"[^\u4e00-\u9f5a0-9]")
def etl(content):
    content = etlregex.sub('', content)
    return content

'''过滤HTML中的标签
#将HTML中标签等信息去掉
#@param htmlstr HTML字符串.'''
def filter_tag(htmlstr):
    re_cdata = re.compile('<!DOCTYPE HTML PUBLIC[^>]*>', re.I)
    re_script = re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>', re.I) #过滤脚本
    re_style = re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>', re.I) #过滤style
    re_br = re.compile('<br\s*?/?>')
    re_h = re.compile('</?\w+[^>]*>')
    re_comment = re.compile('<!--[\s\S]*-->')
    s = re_cdata.sub('', htmlstr)
    s = re_script.sub('', s)
    s=re_style.sub('',s)
    s=re_br.sub('\n',s)
    s=re_h.sub(' ',s)
    s=re_comment.sub('',s)
    blank_line=re.compile('\n+')
    s=blank_line.sub('\n',s)
    s=re.sub('\s+',' ',s)
    s=replaceCharEntity(s)
    return s

'''##替换常用HTML字符实体.
#使用正常的字符替换HTML中特殊的字符实体.
#你可以添加新的实体字符到CHAR_ENTITIES中,处理更多HTML字符实体.
#@param htmlstr HTML字符串.'''
def replaceCharEntity(htmlstr):
    CHAR_ENTITIES={'nbsp':'','160':'',
                    'lt':'<','60':'<',
                    'gt':'>','62':'>',
                    'amp':'&','38':'&',
                    'quot':'"''"','34':'"'}
    re_charEntity=re.compile(r'&#?(?P<name>\w+);') #命名组,把 匹配字段中\w+的部分命名为name,可以用group函数获取
    sz=re_charEntity.search(htmlstr)
    while sz:
        #entity=sz.group()
        key=sz.group('name') #命名组的获取
        try:
            htmlstr=re_charEntity.sub(CHAR_ENTITIES[key],htmlstr,1) #1表示替换第一个匹配
            sz=re_charEntity.search(htmlstr)
        except KeyError:
            htmlstr=re_charEntity.sub('',htmlstr,1)
            sz=re_charEntity.search(htmlstr)
    return htmlstr


def get_content(urls):
    #urls = [line.rstrip() for line in open('./urls.txt')]
    output = open('sogou_experiment.txt', 'w')
    for url in urls:
        try:
            #print url
            trans_url = 'http://tc.chinaso.com/forecommon/?rf=1&url=' + url + '&format=json'
            req = urllib2.Request(trans_url)
            res_data = urllib2.urlopen(req).read()
            res_json = json.loads(res_data)
            title = filter_tag(str(res_json['page']['meta']['title']).strip()).decode("utf8")
            content = filter_tag(str(res_json['page']['contentTransHtml']).strip()).decode("utf8")
            #print title, content
            output.write(url + '\t' + title + '\t' + content + '\n')
        except: continue
    output.close()

def del_stopwords(seg_sent):
    stopwords = [line.rstrip().decode('utf-8') for line in open('./stopwords.txt')]
    new_sent = []
    for word in seg_sent:
        if word.rstrip() in stopwords or not word.rstrip():
            continue
        else:
            new_sent.append(word)
    return new_sent


if __name__ == '__main__':
    #urls = [line.strip().split('\t')[0] for line in open('SogouTDTE.txt')]
    #get_content(urls)
    o = open("sogou_experiment_segs.txt", 'w')
    lines = [''.join(line.strip().split('\t')[1:]) for line in open('sogou_experiment.txt')]
    for line in lines:
        word_list = filter(lambda x: len(x) > 0, map(etl, jieba.cut(line, cut_all=False)))
        word_list = del_stopwords(word_list)
        o.write(' '.join(word_list) + '\n')
    o.close()