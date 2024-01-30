import shutil
import socket
import csv
import igraph as g
from igraph import *
from igraph.clustering import *
import hashlib
import uuid
import networkx as nx
#import graph_tools
import json
from matplotlib import pyplot as plt
from itertools import product
import numpy as np
import ast
from elasticsearch import Elasticsearch
import sys
import os
import pickle
from datetime import datetime
from multiprocessing import Process, process

create_time_dict ={}
def from_elastic(idx):
    if idx == 'small_log2':
        dtype = '_doc'
    else:
        dtype = 'some_type'
    es = Elasticsearch(['http://panther.cs.binghamton.edu:9200'],timeout = 300)
    es.indices.put_settings(index=idx,
                        body={'index':{
                            'max_result_window':99999999}})
    query_body = {
        "query": {
            "match_all": {}
        },
        "sort" : [
        {
            "TimeStamp" : {
                "order" : "asc"
            }
        }
          
        ]
    }
    #es.search(index = idx)
#    result = es.search(index = "125", doc_type = "some_type", body = query_body, size = 9999999999)
    result = es.search(index = idx, doc_type = dtype, body = query_body, size = 99999999)
    all_hits = result['hits']['hits']
    return all_hits

def elastic(idx):
    my_set = []
    if idx == 'small_log':
        dtype = '_doc'
    else:
        dtype = 'some_type'
    es=Elasticsearch(['http://panther.cs.binghamton.edu:9200'],timeout = 300)
    es_index = idx

    #user = "Francesco Totti"

    body2 = {
            "query": {
                "match_all": {}
            }
    }

    res = es.count(index=es_index, doc_type=dtype, body= body2)
    size = res['count']
    body = { "size": 10000000, ##100000,
                "query": {
                    "match_all": {}
                },
                "sort" : [
                    {"TimeStamp": "asc"},
                    {"_id": "desc"}
                ]
            }

    result = es.search(index=es_index, doc_type=dtype, body= body)
    result1 = es.search(index=es_index, doc_type=dtype, body= body)
    print(len(result1['hits']['hits']))
    my_set.append(result1['hits']['hits'])
    bookmark = [result['hits']['hits'][-1]['sort'][0], str(result['hits']['hits'][-1]['sort'][1]) ]

    body1 = {"size": 10000000, #100000,
                "query": {
                    "match_all": {}
                },
                "search_after": bookmark,
                "sort" : [
                    {"TimeStamp": "asc"},
                    {"_id": "desc"}
                ]
            }
    


    #print(size)
    while len(result['hits']['hits']) < size:
        res =es.search(index=es_index, doc_type=dtype, body= body1)
        for el in res['hits']['hits']:
            result['hits']['hits'].append( el )
        bookmark = [res['hits']['hits'][-1]['sort'][0], str(result['hits']['hits'][-1]['sort'][1]) ]
        body1 = {"size": 100000,
                "query": {
                    "match_all": {}
                },
                "search_after": bookmark,
                "sort": [
                    {"TimeStamp": "asc"},
                    {"_id": "desc"}
                ]
            }
        my_set.append(res['hits']['hits'])
    print("my_set",len(my_set))
    return (my_set)

def get_host():
    hostname = socket.gethostname()
    ## getting the IP address using socket.gethostbyname() method
    ip_address = socket.gethostbyname(hostname)
    ## printing the hostname and ip_address
    print(f"Hostname: {hostname}")
    print(f"IP Address: {ip_address}")
    return hostname

def get_uid(hash,hostname):
    hash1 = hashlib.md5(hash.encode()).hexdigest()
    hash2 = hashlib.md5(hostname.encode()).hexdigest()
    h1 = str(bin(int(hash1, 16)).zfill(8))
    h2 = str(bin(int(hash2, 16)).zfill(8))
    out = h1[:92] + h2[92:]
    decimal_representation = int(out, 2)
    hexadecimal_string = hex(decimal_representation)
    hexadecimal_string = hexadecimal_string[2:]
    out = str(uuid.UUID(hexadecimal_string))
    return out

def get_create_time(d):
    provider = d.get('_source', {}).get('ProviderId')
    task_name = d.get('_source', {}).get('Task Name')

    if provider == '{22FB2CD6-0E7B-422B-A0C7-2FAD1FD0E716}':
        if task_name == 'PROCESSSTART':
            gpid = d.get('_source', {}).get('ProcessId')
            gct = d.get('_source', {}).get('CreateTime')
            create_time_dict[gpid] = gct # 1234 - xyz 
    return create_time_dict       

#######################################################################################################################
# Node File----------------------------------------------------------------
def get_file_info(d,host,file_uid, file_thread_uid, file_edge_uid,
mapping,file_node_dict,file_thread_dict,file_edge_dict):
           
    file_event2 = d.get('_source', {}).get('Task Name')
            
    if (file_event2 != 'OPERATIONEND' and file_event2 != "NAMEDELETE"): 
        #print(file_event1)
        
        #infoclass = d.get('_source', {}).get('InfoClass')

        filename = d.get('_source', {}).get('FileName')                 # feature

        fileobj = d.get('_source', {}).get('FileObject')

        #filekey = d.get('_source', {}).get('FileKey')

        #filepath = d.get('_source', {}).get('FilePath')                     # pool

        #byteoffset = d.get('_source', {}).get('ByteOffset')

        file_event1 = d.get('_source', {}).get('Task Name')                 # feature
        #file_event.append(file_event1)
        file_op = d.get('_source', {}).get('Opcode')                         #feature

        #ioflags = d.get('_source', {}).get('IOFlags')

        #iosize = d.get('_source', {}).get('IOSize')

        irp = d.get('_source', {}).get('Irp') #pool

        file_tid = d.get('_source', {}).get('ThreadId')

        file_pid = d.get('_source', {}).get('ProcessId')

        file_PID = d.get('_source', {}).get('ProcessID')
        
        file_ts1 = d.get('_source', {}).get('TimeStamp')
        
        file_TID = d.get('_source', {}).get('ThreadID') 

        #shareacc = d.get('_source', {}).get('ShareAccess')

        #usl = d.get('_source', {}).get('UserStackLimit')
        #sl = d.get('_source', {}).get('StackLimit')
        #wsa = d.get('_source', {}).get('Win32StartAddr')        
        #tb = d.get('_source', {}).get('TebBase')
        #op = d.get('_source', {}).get('OldPriority')
        #np = d.get('_source', {}).get('NewPriority')
       # sb = d.get('_source', {}).get('StackBase')
        spt = d.get('_source', {}).get('SubProcessTag')                         # feature
       # usb = d.get('_source', {}).get('UserStackBase')
        #sa = d.get('_source', {}).get('StartAddr')
        #jobid = d.get('_source', {}).get('Job ID')
       # contid = d.get('_source', {}).get('Container ID')
        
        
        #File UID (Filepath)
        if (file_event1 == 'CREATE' or file_event1 == "CREATENEWFILE"):
            if filename is not None:
                file_hash = str(fileobj)+ str(filename.upper()) + str(file_pid) + str(file_tid)
            else:
                file_hash = str(fileobj)+ str(filename) + str(file_pid) + str(file_tid)
            file_out=get_uid(file_hash,host)
            file_uid.append(file_out)
            mapping[(fileobj,file_pid,file_tid)]=file_out
            file_node_dict[file_out]= {'FileName': filename, 'FileObject': fileobj}
            
        elif file_event1 == 'CLOSE':
            if (fileobj,file_pid,file_tid) in mapping:
                key = mapping[(fileobj,file_pid,file_tid)]
                file_uid.append(key)
                mapping.pop((fileobj,file_pid,file_tid))
            else:
                file_hash = str(fileobj) + str(file_pid) + str(file_tid)
                file_out=get_uid(file_hash,host)
                file_uid.append(file_out)
                file_node_dict[file_out]= {'FileName': filename, 'FileObject': fileobj}
        else:  
            file_hash = str(fileobj) + str(file_pid) + str(file_tid)
            file_out=get_uid(file_hash,host)
            file_uid.append(file_out)
            file_node_dict[file_out]= {'FileName': filename, 'FileObject': fileobj}

            
        #Edge_uid
        hashout = str(file_pid) + str(file_tid) + str(file_ts1)
        fedge_out = get_uid(hashout,host)
        file_edge_uid.append(fedge_out)
    
        #Thread_uid
        create_time_dict=get_create_time(d)
        if file_pid in create_time_dict:
            CreationTime = create_time_dict[file_pid]   # JY @ 2022-11-01 : Changed from "if str(file_pid) in create_time_dict:
                                                             #                             CreationTime = create_time_dict[str(file_pid)]"
                                                             #                             to 
                                                             #                            "if file_pid in create_time_dict:
                                                             #                                 CreationTime = create_time_dict[file_pid]"
            thread_hash = str(file_tid) + str(file_pid) + str(CreationTime)
            out = get_uid(thread_hash,host)
            file_thread_uid.append(out)
        else:
            thread_hash = str(file_tid) + str(file_pid)
            out = get_uid(thread_hash,host)
            file_thread_uid.append(out)


        #print(f"file-thread {out} 'wsa': {wsa}") # added by JY @ 2022-11-01


        file_edge_dict[fedge_out]={'ProcessId': file_pid, 
                                    'Task Name':file_event1,
                                    'Opcode': file_op, 'Irp':irp, 'TimeStamp':file_ts1,
                                    'SubProcessTag':spt}
        file_thread_dict[out]= {'ThreadID' : file_TID, 'ProcessID' :file_PID, 
                                 'ThreadId':file_tid, 'ProcessId':file_pid}
        
## Node Network -------------------------------------------------------------     
def get_net_info(d,host,net_uid, net_thread_uid, net_edge_uid,net_node_dict,net_thread_dict,net_edge_dict):   

    destaddr = d.get('_source', {}).get('daddr') # feature
    
    #srcaddr = d.get('_source', {}).get('saddr') 
    
    #srcport = d.get('_source', {}).get('sport')
            
    #destport = d.get('_source', {}).get('dport')
            
    #conn_id = d.get('_source', {}).get('connid')
            
    net_event1 = d.get('_source', {}).get('Task Name') # feature
    #net_event.append(net_event1)   
    net_op = d.get('_source', {}).get('Opcode') # feature

    net_tid = d.get('_source', {}).get('ThreadId')
    
    net_pid = d.get('_source', {}).get('ProcessId')

    net_PID = d.get('_source', {}).get('ProcessID')

    net_ts1 = d.get('_source', {}).get('TimeStamp')
    
    size = d.get('_source', {}).get('size')             # feature

    # usl = d.get('_source', {}).get('UserStackLimit')
    # sl = d.get('_source', {}).get('StackLimit')
    # wsa = d.get('_source', {}).get('Win32StartAddr')
    # tb = d.get('_source', {}).get('TebBase')
    # op = d.get('_source', {}).get('OldPriority')
    # np = d.get('_source', {}).get('NewPriority')
    # sb = d.get('_source', {}).get('StackBase')
    spt = d.get('_source', {}).get('SubProcessTag')         #feature
    # usb = d.get('_source', {}).get('UserStackBase')
    # sa = d.get('_source', {}).get('StartAddr')
    # jobid = d.get('_source', {}).get('Job ID')
    # contid = d.get('_source', {}).get('Container ID')

        
    # Network UID (daddr)
    net_hash = str(destaddr)
    net_out = get_uid(net_hash,host)
    net_uid.append(net_out)
    
    # Edge_uid
    hashout = str(net_pid) + str(net_tid) + str(net_ts1)
    nedge_out = get_uid(hashout,host)
    net_edge_uid.append(nedge_out)

    #Thread_uid
    create_time_dict=get_create_time(d)
    #if str(net_pid) in create_time_dict:                                   # JY @ 2022-11-01 : Replaced 'str(net_pid)' with 'net_pid'
    #    CreationTime = create_time_dict[str(net_pid)]
    if net_pid in create_time_dict:
        CreationTime = create_time_dict[net_pid]
        thread_hash = str(net_tid) + str(net_pid) + str(CreationTime)
        out = get_uid(thread_hash,host)
        net_thread_uid.append(out)
    else:
        thread_hash = str(net_tid) + str(net_pid)
        out = get_uid(thread_hash,host)
        net_thread_uid.append(out)

    #print(f"net-thread {out} 'wsa': {wsa}") # added by JY @ 2022-11-01

    
    net_node_dict[net_out]= {'daddr':destaddr}
    net_edge_dict[nedge_out]= {'ProcessId': net_pid, 'Task Name':net_event1, 'Opcode':net_op,
                                'TimeStamp':net_ts1, 'size':size,'SubProcessTag':spt}
    net_thread_dict[out]= {'ThreadID' : net_tid, 'ProcessID': net_PID, 
                           'ThreadId':net_tid, 'ProcessId':net_pid}
   
## Node Process -------------------------------------------------------------
def get_proc_info(d, host, proc_uid, proc_thread_uid, proc_edge_uid, proc_node_dict, proc_thread_dict, proc_edge_dict):
    # if d.get('_id', {}) == 'HGVxpIMBmvl4jI2n8zhX':
        # print("found HGVxpIMBmvl4jI2n8zhX")
    proc_event1 = d.get('_source', {}).get('Task Name')             # feature

   # ppid = d.get('_source', {}).get('ParentProcessID')
            
    #sa = d.get('_source', {}).get('StartAddr')
            
    #fpid = d.get('_source', {}).get('FrozenProcessID')
            
    imgs = d.get('_source', {}).get('ImageSize')                    #feature
            
   # ics = d.get('_source', {}).get('ImageCheckSum')
            
    hc = d.get('_source', {}).get('HandleCount')                    #feature
    
   # imgb = d.get('_source', {}).get('ImageBase')
    
    imgname = d.get('_source', {}).get('ImageName')
    
    tds = d.get('_source', {}).get('TimeDateStamp')
    
    proc_PID = d.get('_source', {}).get('ProcessID')

    #ec = d.get('_source', {}).get('ExitCode')
            
    proc_event1 = d.get('_source', {}).get('Task Name')
    #proc_event.append(proc_event1)

    proc_op = d.get('_source', {}).get('Opcode')
            
   # sid = d.get('_source', {}).get('SessionId')
    
    ct = d.get('_source', {}).get('CreateTime')
    
    proc_tid = d.get('_source', {}).get('ThreadId')
    
    proc_pid = d.get('_source', {}).get('ProcessId')
    
    proc_ts1 = d.get('_source', {}).get('TimeStamp')
   
    #et = d.get('_source', {}).get('ExitTime')


    sc = d.get('_source', {}).get('StatusCode')
    ptet = d.get('_source', {}).get('ProcessTokenElevationType') # feature
    pte = d.get('_source', {}).get('ProcessTokenIsElevated')
    ml = d.get('_source',{}).get('MandatoryLabel') #feature
    pfn = d.get('_source',{}).get('PackageFullName') # feature
    pappid = d.get('_source',{}).get('PackageRelativeAppId')
    tet = d.get('_source',{}).get('TokenElevationType') # feature
    readc = d.get('_source',{}).get('ReadOperationCount') # feature
    writec = d.get('_source',{}).get('WriteOperationCount') # feature
    readt = d.get('_source',{}).get('ReadTransferKiloBytes') # feature
    writet = d.get('_source',{}).get('WriteTransferKiloBytes') # feature



    # usl = d.get('_source', {}).get('UserStackLimit')
    # sl = d.get('_source', {}).get('StackLimit')
    # wsa = d.get('_source', {}).get('Win32StartAddr')
    # tb = d.get('_source', {}).get('TebBase')
    # op = d.get('_source', {}).get('OldPriority')
    # np = d.get('_source', {}).get('NewPriority')
    # sb = d.get('_source', {}).get('StackBase')
    spt = d.get('_source', {}).get('SubProcessTag')             #feature
    # usb = d.get('_source', {}).get('UserStackBase')
    # sa = d.get('_source', {}).get('StartAddr')
    # jobid = d.get('_source', {}).get('Job ID')
    # contid = d.get('_source', {}).get('Container ID')
        
    
    # Process UID (PID + Creation time)
    create_time_dict=get_create_time(d)
    #if str(proc_pid) in create_time_dict:                      # JY @ 2022-11-01 : Replaced 'str(proc_pid)' with proc_pid
    #    CreationTime = create_time_dict[str(proc_pid)]
    if proc_pid in create_time_dict:
        CreationTime = create_time_dict[proc_pid]    
        proc_hash = str(proc_pid) + str(CreationTime) 
        proc_out = get_uid(proc_hash,host)
        proc_uid.append(proc_out)
    else:
        proc_hash = str(proc_pid) 
        proc_out = get_uid(proc_hash,host)
        proc_uid.append(proc_out)

        
    # Thread UID (TID + Creation time)
    #if str(proc_pid) in create_time_dict:
    #    CreationTime = create_time_dict[str(proc_pid)]
    if proc_pid in create_time_dict:                            # JY @ 2022-11-01 : Replaced 'str(proc_pid)' with proc_pid
        CreationTime = create_time_dict[proc_pid]    
        thread_hash = str(proc_tid) + str(proc_pid)+ str(CreationTime) 
        out = get_uid(thread_hash,host)
        proc_thread_uid.append(out)
    else:
        thread_hash = str(proc_tid) + str(proc_pid) 
        out = get_uid(thread_hash,host)
        proc_thread_uid.append(out)

    #print(f"proc-thread {out} 'wsa': {wsa}") # added by JY @ 2022-11-01


    # Edge_uid
    hashout = str(proc_pid) + str(proc_tid) + str(proc_ts1)
    eproc_out = get_uid(hashout,host)
    proc_edge_uid.append(eproc_out)
    
    # if proc_out == '21f0a53e-c1fb-216d-031e-8468a47faa78':
        # print("found 21f0a53e-c1fb-216d-031e-8468a47faa78")

    proc_node_dict[proc_out]={'ProcessId': proc_pid}  
    proc_edge_dict[eproc_out] = {'ImageSize':imgs,'HandleCount':hc,'ImageName':imgname,
                                'ProcessID':proc_PID,'Task Name':proc_event1,'Opcode':proc_op,
                                'TimeDateStamp':tds,'CreateTime':ct,'TimeStamp':proc_ts1, 'StatusCode': sc,
                                'ReadOperationCount':readc,'WriteOperationCount':writec, 
                                'ReadTransferKiloBytes': readt, 'WriteTransferKiloBytes':writet,
                                'ProcessTokenElevationType': ptet, 'ProcessTokenIsElevated': pte, 'MandatoryLabel': ml,
                               'PackageRelativeAppId': pappid ,'SubProcessTag':spt,}
    proc_thread_dict[out]= {'ThreadID' : proc_tid, 'ProcessID':proc_PID, 
                            'ThreadId':proc_tid, 'ProcessId':proc_pid}


def get_reg_info(d,host,reg_uid, reg_thread_uid, reg_edge_uid,mapp,reg_node_dict,reg_thread_dict,reg_edge_dict):
        
    #valname = d.get('_source', {}).get('ValueName')
    
    keyobject = d.get('_source', {}).get('KeyObject')
    
    relname = d.get('_source', {}).get('RelativeName') # feature
    
    reg_ts1 = d.get('_source', {}).get('TimeStamp')
    
    #index = d.get('_source', {}).get('Index')
    
    reg_event1 = d.get('_source', {}).get('Task Name')
    #reg_event.append(reg_event1)
    reg_op = d.get('_source', {}).get('Opcode')

    reg_pid = d.get('_source', {}).get('ProcessId')
    
    reg_PID = d.get('_source', {}).get('ProcessID')

    reg_tid = d.get('_source', {}).get('ThreadId')

    status = d.get('_source', {}).get('Status') 

    disp = d.get('_source', {}).get('Disposition')

    # usl = d.get('_source', {}).get('UserStackLimit')
    # sl = d.get('_source', {}).get('StackLimit')
    # wsa = d.get('_source', {}).get('Win32StartAddr')
    # tb = d.get('_source', {}).get('TebBase')
    # op = d.get('_source', {}).get('OldPriority')
    # np = d.get('_source', {}).get('NewPriority')
    # sb = d.get('_source', {}).get('StackBase')
    spt = d.get('_source', {}).get('SubProcessTag') #feature
    # usb = d.get('_source', {}).get('UserStackBase')
    # sa = d.get('_source', {}).get('StartAddr')
    # jobid = d.get('_source', {}).get('Job ID')
    # contid = d.get('_source', {}).get('Container ID')

    # 2. Regisrty UID (KeyName + KeyObject)

    if (reg_op == 32 or reg_op == 33):
        if relname is not None:
            reg_hash = str(keyobject) + str(relname.upper()) + str(reg_pid) + str(reg_tid)
        else:
            reg_hash = str(keyobject) + str(relname) + str(reg_pid) + str(reg_tid)
        reg_out = get_uid(reg_hash,host)
        reg_uid.append(reg_out)
        mapp[(keyobject,reg_pid,reg_tid)] = reg_out
        reg_node_dict[reg_out] = {'KeyObject':keyobject, 'RelativeName':relname
                             }
            #print(mapp)
    elif reg_op == 44:
        if (keyobject,reg_pid,reg_tid) in mapp:
            key = mapp[(keyobject,reg_pid,reg_tid)]
            reg_uid.append(key)
            mapp.pop((keyobject,reg_pid,reg_tid))
        else:
            reg_hash = str(keyobject) + str(reg_pid) + str(reg_tid)
            reg_out = get_uid(reg_hash,host)
            reg_uid.append(reg_out)
            reg_node_dict[reg_out] = {'KeyObject':keyobject, 'RelativeName':relname}
    else:
        reg_hash = str(keyobject) + str(reg_pid) + str(reg_tid)
        reg_out = get_uid(reg_hash,host)
        reg_uid.append(reg_out)
        reg_node_dict[reg_out] = {'KeyObject':keyobject, 'RelativeName':relname}
        

    #Edge_uid
    hashout = str(reg_pid) + str(reg_tid) + str(reg_ts1)
    ereg_out = get_uid(hashout,host)
    reg_edge_uid.append(ereg_out)

    #Thread_UID
    create_time_dict=get_create_time(d)
    #if str(reg_pid) in create_time_dict:
    #    CreationTime = create_time_dict[str(reg_pid)]
    if reg_pid in create_time_dict:                                     # JY @ 2022-11-01 : replaced 'str(reg_pid)' with 'reg_pid'
        CreationTime = create_time_dict[reg_pid]    
        thread_hash = str(reg_tid) + str(reg_pid)+ str(CreationTime) 
        out = get_uid(thread_hash,host)
        reg_thread_uid.append(out)
    else:
        thread_hash = str(reg_tid) + str(reg_pid) 
        out = get_uid(thread_hash,host)
        reg_thread_uid.append(out)
    
    #print(f"reg-thread {out} 'wsa': {wsa}") # added by JY @ 2022-11-01


    reg_edge_dict[ereg_out] = {'ProcessId':reg_pid, 'Task Name':reg_event1,
                                'Opcode':reg_op, 'TimeStamp':reg_ts1,'SubProcessTag':spt,
                                'Status': status, 'Disposition':disp}   
    reg_thread_dict[out]= {'ThreadID' : reg_tid, 'ProcessID': reg_PID,
                            'ThreadId':reg_tid, 'ProcessId':reg_pid}

#######################################################################################################################

def csv_file(csv_root,file_thread_uid, net_thread_uid, proc_thread_uid, reg_thread_uid,file_edge_uid, net_edge_uid, proc_edge_uid, reg_edge_uid, file_uid, net_uid, proc_uid, reg_uid):
    f1= open(os.path.join(csv_root,'proc.csv'), 'w', encoding='UTF8')
    f2= open(os.path.join(csv_root,'net.csv'), 'w', encoding='UTF8')
    f3= open(os.path.join(csv_root,'file.csv'), 'w', encoding='UTF8')
    f4= open(os.path.join(csv_root,'reg.csv'), 'w', encoding='UTF8')
    writer1 = csv.writer(f1, delimiter=' ', lineterminator='\n')
    writer2= csv.writer(f2, delimiter=' ', lineterminator='\n')
    writer3 = csv.writer(f3, delimiter=' ', lineterminator='\n')
    writer4= csv.writer(f4, delimiter=' ', lineterminator='\n')
    print(len(proc_edge_uid), len(net_edge_uid), len(file_edge_uid), len(reg_edge_uid))
    print(len(proc_thread_uid), len(net_thread_uid), len(file_thread_uid), len(reg_thread_uid))
    print(len(proc_uid), len(net_uid), len(file_uid), len(reg_uid))
    for i in range(len(proc_edge_uid)):
        data1 = [proc_thread_uid[i], proc_uid[i], proc_edge_uid[i]]
        writer1.writerow(data1)
    for i in range(len(net_edge_uid)):
        data2 = [net_thread_uid[i], net_uid[i], net_edge_uid[i]]
        writer2.writerow(data2)
    for i in range(len(file_edge_uid)):
        data3 = [file_thread_uid[i], file_uid[i], file_edge_uid[i]]
        writer3.writerow(data3)
    for i in range(len(reg_edge_uid)):
        data4 = [reg_thread_uid[i], reg_uid[i], reg_edge_uid[i]]
        writer4.writerow(data4)
        
    f1.close()
    f2.close()
    f3.close()
    f4.close()

def get_graph(csv_root,graph_root,file_edge_uid, net_edge_uid, proc_edge_uid, reg_edge_uid):
    # csv root to store csv graph root to store graph 
    g1 = Graph.Read_Ncol(os.path.join(csv_root,"proc.csv"), directed=True, names = True, weights =False)
    g1.es["name"] = proc_edge_uid
   # g1.es["Task"] = proc_event
    #Netwrok
    g2 = Graph.Read_Ncol(os.path.join(csv_root,"net.csv"), directed=True, names = True, weights =False)
    g2.es["name"] = net_edge_uid
    #g2.es["Task"] = net_event
    

    #File
    g3 = Graph.Read_Ncol(os.path.join(csv_root,"file.csv"), directed=True, names = True, weights =False)
    g3.es["name"] = file_edge_uid
    #g3.es["Task"] = file_event
    

    #Registry
    g4 = Graph.Read_Ncol(os.path.join(csv_root,"reg.csv"), directed=True, names = True, weights =False)
    g4.es["name"] = reg_edge_uid
    #g4.es["Task"] = reg_event

    # Union
    #final_union = Graph.union(g1,[g2,g3,g4], byname = 'auto')
    
    final_union = Graph.union(g1,[g2,g3,g4], byname = 'auto')
    
    #final_union.vs["color"] = g1.vs["color"]+g2.vs["color"]+g3.vs["color"]+g4.vs["color"]
    #final_union.vs["label"] = g1.vs["label"]+g2.vs["label"]+g3.vs["label"]+g4.vs["label"]
    try:
        final = final_union.simplify( multiple = True, combine_edges = ','.join)
    except:
        # Exception has occurred: TypeError
        # sequence item 0: expected str instance, NoneType found
        
        # Debugger [e for e in final_union.es] returns following
        # igraph.Edge(<igraph.Graph object at 0x7fb2401dfc70>, 235, {'name_1': None, 'name_2': None, 'name_3': '28bb59a8-2b3f-f98b-b046-a778a47faa78', 'name_4': None})
        # igraph.Edge(<igraph.Graph object at 0x7fb2401dfc70>, 235, {'name_1': None, 'name_2': None, 'name_3': '28bb59a8-2b3f-f98b-b046-a778a47faa78', 'name_4': None})
        # > We want to get rid of the Nones, and just have "'name': 32234324-afaf..." pair

        edgeindex_correctval_pair = dict()
        for e in final_union.es:
            # e.g. {'name_1': '291d4494-7ab2-1cdf-3bd6-6858a47faa78', 'name_2': None, 'name_3': None, 'name_4': None}
            edge_attrs = e.attributes() 
            # Try to get the correct value for all edges (e.g. '291d4494-7ab2-1cdf-3bd6-6858a47faa78') * there will be only 1 or None
            not_None_vals = [ v for k,v in edge_attrs.items() if v != None ]
            if len(not_None_vals) == 0:
                # will later need to just drop this (before integraed) edge as carries no value.
                edgeindex_correctval_pair[e.index] = None
            else:             
                edgeindex_correctval_pair[e.index] = not_None_vals[0]   # if exists will be only one.

        # Now delete all "name_<int>"s in the igraph-level (internally, it seems they are contiguous, so generally need caution; but following appears safely deallocates)
        # > https://stackoverflow.com/questions/55291777/how-to-remove-an-edge-attribute-in-python-igraph
        for name_X__attr in edge_attrs:
            del( final_union.es[name_X__attr] )

        # Now

        for e in final_union.es: 
            
            if edgeindex_correctval_pair[e.index] == None:
                # this edge carries no value just get rid of it
                final_union.delete_edges([e.index])
            else:
                # Now use python-igraph "update_attribute" functionality for the edge to only have "name: <correct_value>" 
                e.update_attributes({"name": edgeindex_correctval_pair[e.index]})
                
        # Since we handled the source-of-error, do final_union.simplify
        final = final_union.simplify( multiple = True, combine_edges = ','.join)

    #final.write(os.path.join(graph_root,"Union.dot"))
    final.write_graphml(os.path.join(graph_root,"Union.GraphML"))
    #return final

def get_dicts(dicts_root,file_edge_dict,file_node_dict,file_thread_dict, proc_edge_dict, proc_node_dict, proc_thread_dict,
reg_edge_dict, reg_node_dict, reg_thread_dict, net_edge_dict, net_node_dict, net_thread_dict):
    # Dictinary to hold 'name' attribute with its source and target pair
    # zipiter = zip(final_g.es['name'],final_g.get_edgelist() )
    # my_dict = dict(zipiter)


    # Saving dictionary to file
    # src_tar = open(os.path.join(dicts_root,"src_tar.json"), "w")
    # json.dump(my_dict,src_tar)
    # src_tar.close()


    file_node = open(os.path.join(dicts_root,"file_node.json"), "w")
    json.dump(file_node_dict,file_node)
    file_node.close()

    file_edge = open(os.path.join(dicts_root,"file_edge.json"), "w")
    json.dump(file_edge_dict,file_edge)
    file_edge.close()

    file_thread = open(os.path.join(dicts_root,"file_thread.json"), "w")
    json.dump(file_thread_dict,file_thread)
    file_thread.close()


    net_node = open(os.path.join(dicts_root,"net_node.json"), "w")
    json.dump(net_node_dict,net_node)
    net_node.close()

    net_edge = open(os.path.join(dicts_root,"net_edge.json"), "w")
    json.dump(net_edge_dict,net_edge)
    net_edge.close()

    net_thread = open(os.path.join(dicts_root,"net_thread.json"), "w")
    json.dump(net_thread_dict,net_thread)
    net_thread.close()


    proc_node = open(os.path.join(dicts_root,"proc_node.json"), "w")
    json.dump(proc_node_dict,proc_node)
    proc_node.close()

    proc_edge = open(os.path.join(dicts_root,"proc_edge.json"), "w")
    json.dump(proc_edge_dict,proc_edge)
    proc_edge.close()

    proc_thread = open(os.path.join(dicts_root,"proc_thread.json"), "w")
    json.dump(proc_thread_dict,proc_thread)
    proc_thread.close()


    reg_node = open(os.path.join(dicts_root,"reg_node.json"), "w")
    json.dump(reg_node_dict,reg_node)
    reg_node.close()

    reg_edge = open(os.path.join(dicts_root,"reg_edge.json"), "w")
    json.dump(reg_edge_dict,reg_edge)
    reg_edge.close()

    reg_thread = open(os.path.join(dicts_root,"reg_thread.json"), "w")
    json.dump(reg_thread_dict,reg_thread)
    reg_thread.close()

    
def first_step(idx,root_path):
        
    # Connect with elastic search
    reg_node_dict ={}
    reg_thread_dict = {}
    reg_edge_dict = {}
    file_node_dict = {}
    file_thread_dict ={}
    file_edge_dict = {}
            
    # Network attributes
    net_node_dict = {}
    net_thread_dict ={}
    net_edge_dict = {}
            
    # Process attributes
    proc_node_dict = {}
    proc_thread_dict = {}
    proc_edge_dict = {}
    mapping = {}
    mapp = {}
    #print(len(k))
        ##  used for csv files to 
    reg_uid = []
    reg_thread_uid = []
    reg_edge_uid = []
    file_uid = []
    file_thread_uid = []
    file_edge_uid = []
    net_uid = []
    net_thread_uid = []
    net_edge_uid = []
    proc_uid = []
    proc_thread_uid = []
    proc_edge_uid = []
    all_hits_val = elastic(idx)
    print("all_hits_val",len(all_hits_val)) # [[{10000 log entries}],[100000],[]] ----> [{},{},{}]----[{}**]

    # Gethost
    host_name = get_host()

    # Get providers data
    for v , k in enumerate(all_hits_val):
        for num, dic in enumerate(k):
            provider = dic["_source"].get('ProviderId')
            if provider == "{EDD08927-9CC4-4E65-B970-C2560FB5C289}":
                get_file_info(dic,host_name,file_uid, file_thread_uid, file_edge_uid,mapping,file_node_dict,file_thread_dict,file_edge_dict)
            if provider == '{7DD42A49-5329-4832-8DFD-43D979153A88}':
                get_net_info(dic,host_name,net_uid, net_thread_uid, net_edge_uid,net_node_dict,net_thread_dict,net_edge_dict)
            if provider == '{22FB2CD6-0E7B-422B-A0C7-2FAD1FD0E716}':
                get_proc_info(dic,host_name,proc_uid, proc_thread_uid, proc_edge_uid,proc_node_dict,proc_thread_dict,proc_edge_dict)
            if provider == '{70EB4F03-C1DE-4F73-A051-33D13D5413BD}':
                get_reg_info(dic,host_name,reg_uid, reg_thread_uid, reg_edge_uid,mapp,reg_node_dict,reg_thread_dict,reg_edge_dict)

        # get csv files for graph
        csv_file(root_path,file_thread_uid, net_thread_uid, proc_thread_uid, reg_thread_uid,file_edge_uid, net_edge_uid, proc_edge_uid, reg_edge_uid, file_uid, net_uid, proc_uid, reg_uid)

        # Generating computation graphs
        get_graph(root_path,root_path,file_edge_uid, net_edge_uid, proc_edge_uid, reg_edge_uid)


        # Get all dictionaries
        get_dicts(root_path,file_edge_dict,file_node_dict,file_thread_dict, proc_edge_dict, proc_node_dict, proc_thread_dict,
        reg_edge_dict, reg_node_dict, reg_thread_dict, net_edge_dict, net_node_dict, net_thread_dict)
        
    
if __name__ == "__main__":
    main()

##TODO:
# 1. Generate all realtime malware data
# 2. Generate all offline malware data
# 3. to cross check the offline graphs with real time SG's