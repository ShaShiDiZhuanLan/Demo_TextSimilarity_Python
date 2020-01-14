# -*- coding: utf-8 -*-
"""
Author: 沙振宇
CreateTime: 2019-1-14
Info: 文本相似度算法（利用CountVectorizer计算文本相似度）
"""
from sklearn.feature_extraction.text import CountVectorizer
import math
import jieba


class TextSimilar(object):
    """
    文本相似度算法
    """
    @classmethod
    def sparse_convert(cls, matrix):
        """
        稀疏转换
        :param matrix:
        :return:
        """
        output = []
        mat = matrix.toarray()
        for v in mat:
            out = []
            for i in range(len(v)):
                if v[i] > 0:
                    out.append((i,v[i]))
            output.append(out)
        return output

    @classmethod
    def preprocess(cls, info):
        """
        预处理文本
        :param info:
        :return:
        """
        if isinstance(info, str):
            text_with_spaces = ""
            text_cut = jieba.cut(info.strip())
            stopwords = ""
            for word in text_cut:
                if word not in stopwords:
                    if word != "\t":
                        text_with_spaces += word + " "
        else:
            raise TypeError("text should be str")
        return text_with_spaces

    @classmethod
    def norm_vector_nonzero(cls, ori_vec):
        """
        计算非零的范数向量
        :param ori_vec:
        :return:
        """
        ori_sum = math.sqrt(sum([math.pow(float(value),2) for (idx,value) in ori_vec]))
        if ori_sum < 1e-6:
            return ori_vec
        result_vec = []
        for idx, ori_value in ori_vec:
            result_vec.append((idx, float(ori_value)/ori_sum))
        return result_vec

    @classmethod
    def sort_score(cls, t_arr):
        """
        按照分值降序排序
        :param t_arr:
        :return:
        """
        t_arr = sorted(t_arr, reverse=True, key=lambda s: s[0])
        return t_arr

    def cosine_distance_nonzero(self, feat_vec1, feat_vec2, norm=True):
        """
        计算非零的余弦距离
        :param feat_vec1:
        :param feat_vec2:
        :param norm:
        :return:
        """
        if norm:
            feat_vec1 = self.norm_vector_nonzero(feat_vec1)
            feat_vec2 = self.norm_vector_nonzero(feat_vec2)
        dist = 0
        idx1 = 0
        idx2 = 0
        while idx1 < len(feat_vec1) and idx2 < len(feat_vec2):
            if feat_vec1[idx1][0] == feat_vec2[idx2][0]:
                dist += float(feat_vec1[idx1][1])*float(feat_vec2[idx2][1])
                idx1 += 1
                idx2 += 1
            elif feat_vec1[idx1][0] > feat_vec2[idx2][0]:
                idx2 += 1
            else:
                idx1 += 1
        return dist

    def run(self, t_train, t_test, score_minimum=0.8):
        """
        计算文本相似度
        :param t_train: 训练数据
        :param t_test: 测试数据
        :param score_minimum: 最低分值
        :return:
        """
        texts_test = []
        texts_train = []

        for tex in t_test:
            texts_test.append(self.preprocess(tex))
        for tex in t_train:
            texts_train.append(self.preprocess(tex))

        # 计算训练数据词向量
        all_texts = texts_train + texts_test
        count_vector = CountVectorizer(analyzer="word", token_pattern=u"(?u)\\b\\w+\\b")
        vector_matrix = count_vector.fit_transform(all_texts)

        # 计算测试数据词向量
        all_matrix = self.sparse_convert(vector_matrix)
        matrix_train = all_matrix[:len(texts_train)]
        matrix_test = all_matrix[len(texts_train):]
        result_map = {}
        array_map = []
        for i in range(len(matrix_train)):
            for j in range(len(matrix_test)):
                dis = self.cosine_distance_nonzero(matrix_train[i], matrix_test[j])
                if dis >= score_minimum:
                    result_map[i] = "1"
                    tmp_map = (dis, str(t_train[i]))
                    array_map.append(tmp_map)
                else:
                    result_map[i] = "0"

        if "1" in result_map.values():
            array_map = text_similar.sort_score(array_map)
            return True, array_map
        else:
            return False, array_map


if __name__ == "__main__":
    text_similar = TextSimilar()
    test = ["温馨的一笑"]
    train = ["笑的很温馨", "温馨的花朵", "温馨的一个人笑了"]
    match, arr = text_similar.run(train, test, score_minimum=0.1)
    print("test:", test)
    print("train:", train)
    print("match:", match)
    print("arr:", arr)
