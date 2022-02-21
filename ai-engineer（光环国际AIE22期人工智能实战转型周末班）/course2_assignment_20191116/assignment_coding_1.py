
###  病例数据
# 症状    职业     疾病 #
#1 打喷嚏  护士      感冒
#2 打喷嚏  农夫      过敏
#3 头痛    建筑工人  脑震荡
#4 头痛    建筑工人  感冒
#5 打喷嚏  教师      感冒
#6 头痛    教师      脑震荡

arr_list = [['打喷嚏','护士','感冒'],
        ['打喷嚏' ,'农夫','过敏'],
        ['头痛','建筑工人','脑震荡'],
        ['头痛', '建筑工人', '感冒'],
        ['打喷嚏','教师','感冒'],
        ['头痛','教师','脑震荡']]

import numpy as np
arr_data = np.matrix(arr_list)
# print(arr_data)

# 各维度 独立发生概率
def p_fun(arr,dime):
    arr_len = len(arr)
    list_temp = []
    for str in arr:
        # print(str[0,0])
        list_temp.append(str[0,0])

    for w in list_temp:
        if w in dime:
            dime[w] = float(dime[w]) + float(1/arr_len)
        else:
            dime[w] = float(1/arr_len)

    return dime


# 症状概率
p_symptom = {}
p_symptom = p_fun(arr_data[:,0],p_symptom)
# print(p_symptom)

# 职业概率
p_occupation = {}
p_occupation = p_fun(arr_data[:,1],p_occupation)
# print(p_occupation)

# 疾病概率
p_disease = {}
p_disease = p_fun(arr_data[:,2],p_disease)
#print(p_disease)

def v_disease_fun(v1,disease,v1_arr):
    len_d = 0
    occ_dis_p = 0.0
    print(v1,disease)
    for d in v1_arr:
        if disease == d[0,1]:
            len_d += 1
    for o in v1_arr:
        if v1 == o[0,0] and o[0,1] == disease:
            occ_dis_p += float(1 / len_d)
    return occ_dis_p


# p = v_disease_fun('建筑工人','感冒',arr_data[:,1:3])
# p = v_disease_fun('打喷嚏','感冒',arr_data[:,c])

c = [0,2]
## 问题公式  P(疾病|症状 * 职业) = P(症状|疾病) * P(职业|疾病) * P(疾病) / （P(症状) * P(职业)）
def forecast_fun(occ,sym,dis):
    p_sym_dis = v_disease_fun(sym,dis,arr_data[:,c])
    print(p_sym_dis)

    p_occ_dis = v_disease_fun(occ,dis,arr_data[:,1:3])
    print(p_occ_dis)

    p_dis     = p_disease[dis]
    print(p_dis)

    p_sym     = p_symptom[sym]
    print(p_sym)

    p_occ     = p_occupation[occ]
    print(p_occ)

    return p_sym_dis * p_occ_dis * p_dis / (p_sym * p_occ)

# 打喷嚏的建筑工人患感冒的概率?
print(forecast_fun('建筑工人','打喷嚏','感冒'))

# 打喷嚏的护士患感冒的概率？
#print(forecast_fun('农夫','打喷嚏','感冒'))

## 打喷嚏的建筑工人患感冒的概率？






