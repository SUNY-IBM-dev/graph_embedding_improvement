from model import txt2json as tj
import ast
import json
import os
import re
#from dateutil.parser import parse
def selection(fp1,fp2):

    f1 = open(fp1, 'r', encoding = 'utf-8', errors = 'ignore')
    f2 = open(fp2, 'w', encoding = 'utf-8', errors = 'ignore')
    #f2.write('[')
    data = tj.get_data(f1)
    f2.write('[')
    while data:
        d = json.loads(data)

        new_dic = {}
        count = 0
        data = tj.get_data(f1)
        if data:
            for item in d.items():
                if item[0] not in tj.no_attributes:
                    if item[0] == 'ThreadId' :
                        if isinstance(item[1], str):
                            new_dic[item[0]]= int(item[1], 16)
                        else:
                            new_dic[item[0]] = item[1]
                    else:
                        new_dic[item[0]] = item[1]
            new_str = json.dumps(new_dic)
            f2.write(new_str + ",\n")
        else: # LAST LOG-ENTRY
            # JY @ 2023-1-15 : DONT KNOW WHY TREATING THE LAST LOG-ENTRY DIFFERENTLY IS NEEDED. THIS CAUSES PROBLEMS
            #                  SO IM JUST GOING TO HAVE IT AS ABOVE EXCEPT FOR THE LAST TIME (f2.write(new_str + "]"))
            for item in d.items():
                if item[0] not in tj.no_attributes:
                    if item[0] == 'ThreadId' :
                        if isinstance(item[1], str):
                            new_dic[item[0]]= int(item[1], 16)
                        else:
                            new_dic[item[0]] = item[1]
                    else:
                        new_dic[item[0]] = item[1]
            new_str = json.dumps(new_dic)
            f2.write(new_str + "]")

            '''
            for item in d.items():
                if item[0] in tj.attributes:
                    if item[0] not in tj.no_attributes:
                        if item[0] == 'ThreadId':
                            if isinstance(item[1], str):
                                new_dic[item[0]] = int(item[1], 16)
                            else:
                                new_dic[item[0]] = item[1]
                        else:
                            new_dic[item[0]] = item[1]
            new_str = json.dumps(new_dic)
            f2.write(new_str + "]")
            '''
    f1.close()
    f2.close()



if __name__ == "__main__":
    main()
