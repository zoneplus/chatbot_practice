# encoding=utf-8
"""
This file is used to pre-process raw files

@author:wangqiqi

"""
import gensim
import codecs
import word2vec
import sys
import jieba
from jieba import analyse
def main1():  
    path_to_model = 'sub_test.bin'  
    output_file = 'vector_test_sub_test.txt'  
    export_to_file(path_to_model, output_file)  
  
  
def export_to_file(path_to_model, output_file):  
    output = codecs.open(output_file, 'w' , 'utf-8')  
    #model = gensim.models.KeyedVectors.load_word2vec_format(path_to_model, binary=True)  
    model = word2vec.load(path_to_model)
    print('done loading Word2Vec')  
    vocab = model.vocab  
    for mid in vocab:   
        vector = list()  
        for dimension in model[mid]:  
            vector.append(str(dimension))  
         vector_str = ",".join(vector)  
        line = mid + "\t"  + vector_str  
        output.write(line + "\n")  
    output.close()
def w2v(input):
     word2vec.word2vec(input, 'sub_test.bin', size=300,verbose=True)

def segment(input, output):
    input_file = open(input, "r")
    lis = ['M','E','?','!','...','e']
    output_file = open(output, "w")
    while True:
        line = input_file.readline()
        if line:
            line = line.strip()
            seg_list = jieba.cut(line)
            segments = ""
            for str in seg_list:
              if str not in lis:
                segments = segments + " " + str
            segments = segments + "\n"
            output_file.write(segments)
        else:
            break
    input_file.close()
    output_file.close()

if __name__ == '__main__':
    segment('fk24_sub.txt', 'vector_seg_sub.txt');
    w2v('vector_seg_sub.txt')
    main1()
