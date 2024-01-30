from model import txt2json as tj
import ast
import json
import os
import re
#from dateutil.parser import parse

def mutilevel2singlelevel(json1, json2):
    fp = open(json1, encoding = 'utf-8', errors = 'ignore')
    f2 = open(json2, 'w', encoding = 'utf-8', errors = 'ignore')
    #f2.write('[')
    data = tj.get_data(fp)
    print('phase1:converting')
    while data:
        d = eval(data)
        new_dic = {}
        count = 0
        data = tj.get_data(fp)
        if data:
            for item in d.items():
         
                if count == 0:
                    count = 1
                    item = item[1]
                    for item2 in item.items():
                        if item2[0] == 'EventDescriptor':
                            item2 = item2[1]
                            for item3 in item2.items():
                                new_dic[item3[0]] = item3[1]
                        else:
                            new_dic[item2[0]] = item2[1]
                else:

                    new_dic[item[0]] = item[1]
            new_str = json.dumps(new_dic)
            f2.write(new_str + "\n")
        else:
            for item in d.items():
                if count == 0:
                    count = 1
                    item = item[1]
                    for item2 in item.items():
                        if item2[0] == 'EventDescriptor':
                            item2 = item2[1]
                            for item3 in item2.items():
                                new_dic[item3[0]] = item3[1]
                        else:
                            new_dic[item2[0]] = item2[1]
                else:
                    new_dic[item[0]] = item[1]
            new_str = json.dumps(new_dic)
            f2.write(new_str)
    fp.close()
    f2.close()
    print('phase1:finished')


if __name__ == "__main__":
    main()
