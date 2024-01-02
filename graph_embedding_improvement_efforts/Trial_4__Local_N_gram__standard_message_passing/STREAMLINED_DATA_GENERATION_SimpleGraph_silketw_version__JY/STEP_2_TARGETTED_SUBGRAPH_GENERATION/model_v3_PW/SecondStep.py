import json
import pickle
import os
#from model_v2 import dataonehot as dot
from dateutil.parser import parse
import pprint


def update_dict(dicts_root,node_dict,edge_dict):
    file_node = open(os.path.join(dicts_root,"file_node.json"), "r")
    node_dict.update(json.load(file_node))

    net_node = open(os.path.join(dicts_root,"net_node.json"), "r")
    node_dict.update(json.load(net_node))

    reg_node = open(os.path.join(dicts_root,"reg_node.json"), "r")
    node_dict.update(json.load(reg_node))

    proc_node = open(os.path.join(dicts_root,"proc_node.json"), "r")
    node_dict.update(json.load(proc_node))

#####################################################################

    ft_node = open(os.path.join(dicts_root,"file_thread.json"), "r")
    node_dict.update(json.load(ft_node))

    nt_node = open(os.path.join(dicts_root,"net_thread.json"), "r")
    node_dict.update(json.load(nt_node))

    rt_node = open(os.path.join(dicts_root,"reg_thread.json"), "r")
    node_dict.update(json.load(rt_node))

    pt_node = open(os.path.join(dicts_root,"proc_thread.json"), "r")
    node_dict.update(json.load(pt_node))

###################################################################
    file_edge = open(os.path.join(dicts_root,"file_edge.json"), "r")
    edge_dict.update(json.load(file_edge))

    net_edge = open(os.path.join(dicts_root,"net_edge.json"), "r")
    edge_dict.update(json.load(net_edge))

    reg_edge = open(os.path.join(dicts_root,"reg_edge.json"), "r")
    edge_dict.update(json.load(reg_edge))

    proc_edge = open(os.path.join(dicts_root,"proc_edge.json"), "r")
    edge_dict.update(json.load(proc_edge))
    
#################################################################


#PW: based on SilkETW EventName
#Registry: EventID(1)-(Opcode:32 Opcodename:Createkey), EventID(2)-(Opcode:33 Opcodename:Openkey),
# EventID(3)-(Opcode:34 Opcodename:Deletekey), EventID(4)-(Opcode:35 Opcodename:Querykey), 
# EventID(5)( Opcode:36, OpcodeName: SetValueKey), EventID(6)( Opcode:37, Opcodename: DeleteValueKey), 
# EventID(7)-(Opcode:38 Opcodename:QueryValueKey),  EventID(8)-(Opcode:39, Opcodename:EnumerateKey), 
# EventID(9)-(Opcode:40 Opcodename: EnumerateValuekey), EventID(10)-(Opcode:41 Opcodename: QueryMultipleValuekey),
# EventID(11)-(Opcode:42 Opcodename: Setinformationkey), EventID(13)-(Opcode:44 Opcodename:Closekey), 
# EventID(14)-(Opcode:45 Opcodename: QuerySecuritykey),EventID(15)-(Opcode:46 Opcodename: SetSecuritykey),
#  Thisgroupofeventstrackstheperformanceofflushinghives - (opcode 13,OpcodeName:RegPerfOpHiveFlushWroteLogFile) 

#KERNEL_NETWORK_TASK_TCPIP/Datasent (Opcode 10), KERNEL_NETWORK_TASK_TCPIP/Datareceived (Opcode:11),
# KERNEL_NETWORK_TASK_TCPIP/connectionattempted(Opcode:12),  KERNEL_NETWORK_TASK_TCPIP/Disconnectissued (Opcode:13), 
# KERNEL_NETWORK_TASK_TCPIP/Dataretransmitted (Opcode: 14), KERNEL_NETWORK_TASK_TCPIP/connectionaccepted (Opcode:15), 
# KERNEL_NETWORK_TASK_TCPIP/Protocolcopieddataonbehalfofuser (Opcode: 18), KERNEL_NETWORK_TASK_TCPIP/DatareceivedoverUDPprotocol (Opcode:43),
# KERNEL_NETWORK_TASK_TCPIP/DatasentoverUDPprotocol (Opcode:42)

#Task Name == EventName in SilkETW
Task_Names = [
    'Cleanup', #1
    'Close', #2
    'Create', #3
    'CreateNewFile', #4
    'DeletePath',#5
    'DirEnum',#6
    'DirNotify',#7
    'Flush',#8
    'FSCTL',#9
    'NameCreate',#10
    'NameDelete',#11
    'OperationEnd',#12
    #'QueINFO',#13
    'QueryInformation',#13
    'QueryEA',#14
    'QuerySecurity',#15
    'Read',#16
    'Write',#17
    'SetDelete',#18
    'SetInformation', #19
    'PagePriorityChange',#20
    'IoPriorityChange',#21
    'CpuBasePriorityChange',#22
    #'IMAGEPriorityChange',#24
    'CpuPriorityChange',#23
    'ImageLoad',#24
    'ImageUnload',#25
    'ProcessStop/Stop',#26
    'ProcessStart/Start',#27
    'ProcessFreeze/Start',#28--------
    #'PSDISKIOATTRIBUTE',#31
    #'PSIORATECONTROL',#32 
    'ThreadStart/Start',#29
    'ThreadStop/Stop',#30
    'ThreadWorkOnBehalfUpdate', #31
    'JobStart/Start',#32--------
    'JobTerminate/Stop',#33--------
    #'LOSTEVENT',#38
    #'PSDISKIOATTRIBUTION',#39
    'Rename',#34
    'Renamepath',#35
    'Thisgroupofeventstrackstheperformanceofflushinghives',#36-------
    'EventID(1)',#37
    'EventID(2)',#38
    'EventID(3)',#39
    'EventID(4)',#40
    'EventID(5)',#41
    'EventID(6)',#42
    'EventID(7)',#43
    'EventID(8)',#44
    'EventID(9)',#45
    'EventID(10)',#46
    'EventID(11)',#47
    'EventID(13)',#48
    'EventID(14)',#49
    'EventID(15)',#50
    'KERNEL_NETWORK_TASK_TCPIP/Datasent.', #51
    'KERNEL_NETWORK_TASK_TCPIP/Datareceived.',#52
    'KERNEL_NETWORK_TASK_TCPIP/Connectionattempted.',#53
    'KERNEL_NETWORK_TASK_TCPIP/Disconnectissued.', #54
    'KERNEL_NETWORK_TASK_TCPIP/Dataretransmitted.',#55
    'KERNEL_NETWORK_TASK_TCPIP/connectionaccepted.' , #56----
    'KERNEL_NETWORK_TASK_TCPIP/Protocolcopieddataonbehalfofuser.', #57
    'KERNEL_NETWORK_TASK_UDPIP/DatareceivedoverUDPprotocol.',#58
    'KERNEL_NETWORK_TASK_UDPIP/DatasentoverUDPprotocol.', #59
    
    #PW: below events are now different in silketw
    # all below 3 task are combined with opcode and having index 43 onwards for all of them in the function TN2int()
    # 'KERNEL_NETWORK_TASK_UDPIP'#index 43 # 42(opcode value) 43,49(https://github.com/repnz/etw-providers-docs/blob/master/Manifests-Win7-7600/Microsoft-Windows-Kernel-Network.xml)
    # 'KERNEL_NETWORK_TASK_TCPIP', # 10-18 (https://github.com/repnz/etw-providers-docs/blob/master/Manifests-Win7-7600/Microsoft-Windows-Kernel-Network.xml)
    # 'MICROSOFT-WINDOWS-KERNEL-REGISTRY', # 32- 46 (https://github.com/repnz/etw-providers-docs/blob/master/Manifests-Win7-7600/Microsoft-Windows-Kernel-Registry.xml)

]
#42+27=69




def RN2int(RelativeName):
    # mul values 
    RelativeName = str(RelativeName)
    new_list = [0] * 20
    if RelativeName == 'None' or RelativeName == '':
        new_list[0]= 1
    else: 
        new_list[19] = 1 #rest of the substring in relativename
        # General info , Registry hierarchy depth 0
        if RelativeName.lower().find('registry') != -1:
            new_list[1] =1
            new_list[19] =0
            # General info , Registry hierarchy depth 1
        if RelativeName.lower().find('user') != -1:
            new_list[2] =1
            new_list[19] =0
        if RelativeName.lower().find('machine') != -1:
            new_list[3] =1
            new_list[19] =0
            # General info , Registry hierarchy depth 2 under machine
        if RelativeName.lower().find('software') != -1:
            new_list[4] =1
            new_list[19] =0
        if RelativeName.lower().find('system') != -1:
            new_list[5] =1
            new_list[19] =0
        # General info , Registry hierarchy depth 3 under software
        if RelativeName.lower().find('classes') != -1:
            new_list[6] =1
            new_list[19] =0

        # General info , Registry hierarchy depth 2 under user
        if "S-1-5-18" in RelativeName[0]:
            new_list[7] =1
            new_list[19] =0
        if  "S-1-5-19" in RelativeName[0]:
            new_list[8] =1
            new_list[19] =0
        if "S-1-5-20" in RelativeName[0]:
            new_list[9] =1
            new_list[19] =0
        if "S-1-5-21" in RelativeName[0]:
            # PW only this available
            new_list[10] =1
            new_list[19] =0
        if "S-1-5-22" in RelativeName[0]:
            new_list[11] =1
            new_list[19] =0
        if  "S-1-5-87" in RelativeName[0]:
            new_list[12] =1
            new_list[19] =0
        # general informatin, not in hierarchy
        # PW: very few 1 or 2
        if RelativeName[0] == "{EDD08927-9CC4-4E65-B970-C2560FB5C289}":
            new_list[13] =1
            new_list[19] =0
        if RelativeName[0] == "{70EB4F03-C1DE-4F73-A051-33D13D5413BD}":
            new_list[14] =1
            new_list[19] =0
        if RelativeName[0] == "{7DD42A49-5329-4832-8DFD-43D979153A88}":
            new_list[15] =1
            new_list[19] =0
        if RelativeName[0] == "{22FB2CD6-0E7B-422B-A0C7-2FAD1FD0E716}":
            new_list[16] =1
            new_list[19] =0
        if RelativeName[0] == "Properties":
            new_list[17] =1
            new_list[19] =0
        # highlighting info which might help in classifying benign and malware 
        # PW: arround 30
        if RelativeName.lower().find('security') != -1:
            new_list[18] =1
            new_list[19] =0
                
        
    return new_list

def IN2int(imagename):
    # new list with mul values 
    imagename = str(imagename)
    new_list = [0] * 6
    if imagename == 'None' or imagename == '':
        new_list[0] = 1
    else:
        new_list[5] = 1
        if imagename.find('\\Windows') != -1:
            new_list[1] = 1
            new_list[5] = 0
        elif imagename.find('\\Program Files') != -1:
            new_list[2] = 1
            new_list[5] = 0
        elif imagename.find('\\Users') != -1: #changed from User-> Users
            new_list[3] = 1
            new_list[5] = 0
        elif imagename.find('\\') == -1:
            new_list[4] = 1
            new_list[5] = 0
    return new_list

#taskname to numerical
# This is pywintrace version TN2int
def TN2int(taskname,opcode): # index0= none

    opcode = str(opcode)
    taskname = str(taskname)
    #print(taskname)
    length=71
    fl = [0]*length
    if taskname == 'None' or taskname == '':
        fl[0] = 1
    else:
        if taskname == 'KERNEL_NETWORK_TASK_UDPIP':
            if opcode == '42':
                fl[43]=1
            if opcode == '43':
                fl[44]=1
            if opcode == '49':
                fl[45]=1
        elif taskname == 'KERNEL_NETWORK_TASK_TCPIP':
            if opcode == '10':
                fl[46]=1
            if opcode == '11':
                fl[47]=1
            if opcode == '12':
                fl[48]=1
            if opcode == '13':
                fl[49]=1
            if opcode == '14':
                fl[50]=1
            if opcode == '15':
                fl[51]=1
            if opcode == '16':
                fl[52]=1
            if opcode == '17':
                fl[53]=1
            if opcode == '18':
                fl[54]=1
        elif taskname == 'MICROSOFT-WINDOWS-KERNEL-REGISTRY':
            if opcode == '32':
                fl[55]=1
            if opcode == '33':
                fl[56]=1
            if opcode == '34':
                fl[57]=1
            if opcode == '35':
                fl[58]=1
            if opcode == '36':
                fl[59]=1
            if opcode == '37':
                fl[60]=1
            if opcode == '38':
                fl[61]=1
            if opcode == '39':
                fl[62]=1
            if opcode == '40':
                fl[63]=1
            if opcode == '41':
                fl[64]=1
            if opcode == '42':
                fl[65]=1
            if opcode == '43':
                fl[66]=1
            if opcode == '44':
                fl[67]=1
            if opcode == '45':
                fl[68]=1
            if opcode == '46':
                fl[69]=1
        else:
            if taskname not in Task_Names:
                fl[70] = 1
            else:
                for index, value in enumerate(Task_Names, 1):
                    if taskname == value: 
                        fl[index]=1
        return fl
    
#taskname to numerical based on SILKETW
def TN2int(taskname):
    length=61
    fl = [0]*length
    if taskname == 'None' or taskname == '':
        fl[0] = 1
    else:
        if taskname not in Task_Names:
                fl[60] = 1
        else:
            for index, value in enumerate(Task_Names, 1):
                if taskname == value: 
                    fl[index]=1
    return fl


#Hexdecimal to numercial
def Hex2Dec(hex,attrs):
    if hex is None:
        return 0
    else:
        int_num = int(hex, 16)
        if attrs == 'Irp':
            #if int_num > 18000000000000000:
            return (int_num / 10000000000000000000)
        else: # ImageSize
            return int_num

def str2int(str : str):
   if str is None:
       return 0
   else:
       a = int(str)
       return a

def FN2int(filename):
      # return list  with 15 values
    filename = str(filename)
    new_list = [0] * 10 # initially it was 15
    if filename == 'None' or filename == '':
        new_list[0] = 1
    else: 
        new_list[9] = 1 # handle rest substrings in filename
        if "Windows".lower() in filename.lower():
            new_list[1] = 1
            new_list[9] = 0
        if "sys" in filename.lower():
            new_list[2] = 1
            new_list[9] = 0
        if '\\Program Files\\'.lower() in filename.lower():
            new_list[3] = 1
            new_list[9] = 0
        if "Users".lower() in filename.lower():
            new_list[4] = 1
            new_list[9] = 0
        if "logs" in filename.lower():
            new_list[5] = 1
            new_list[9] = 0
        if '\\ProgramData'.lower() in filename.lower():
            new_list[6] = 1
            new_list[9] = 0
        if '\\DR0'.lower() in filename.lower(): #The device harddisk0 dr0 has a bad block indicates that there may be a bad block on your hard disk.
            #PW: not available in new subgraphs
            new_list[7] = 1
            new_list[9] = 0
        if "Windows Defender".lower() in filename.lower(): #PW:change from Windows Defender Advanced Threat Protection to Windows Defender 
            new_list[8] = 1
            new_list[9] = 0     
    '''
        if "secure".lower() in filename.lower():#not obsrerved
            new_list[7] = 1
            new_list[14] = 0
        if "securityhealthservice.exe".lower() in filename.lower(): #not obsrerved
            new_list[10] = 1
            new_list[14] = 0  
        if "CRYPTBASE".lower() in filename.lower(): #not obsrerved
            new_list[11] = 1
            new_list[14] = 0
        if "BCRYPT".lower() in filename.lower():#not obsrerved
            new_list[12] = 1
            new_list[14] = 0
        if "NCRYPT".lower() in filename.lower() and "fsencryption".lower() not in filename.lower(): #not obsrerved
            new_list[13] = 1
            new_list[14] = 0
    '''
    return new_list 


#PW: since address is string value in silketw logs e.g, "00000000000000000000000000000001"
def str2int(addr):
    if addr is None or addr == '':
        return 0
    else:
        hex_string = addr
        if ',' in hex_string:
            hex_string = hex_string.replace(',','')
            integer_value = int(hex_string, 16)
            return integer_value
        else:
            integer_value = int(hex_string, 16)
            return integer_value


def address2dec(addr):
    if addr is None or addr == '':
        return 0
    else:
        if addr.find(':') == -1:
            temp = addr.split(".")
            x = int(temp[0])
            x2 = int(temp[1])
            x3 = int(temp[2])

            if x == 127:
                return 1
            if x == 10:
                return 2
            if x == 172 and x2 >= 16 and x2 <= 31:
                return 3
            if x == 192 and x2 == 168:
                return 4
    return 5


def address2dec_dupicated(addr):
    if addr is None:
        return 0
    else:
        if addr.find(':') == -1:
            temp = addr.split(".")
            x = int(temp[0])
            x2 = int(temp[1])
            x3 = int(temp[2])

            if x >= 0 and x <= 127:
                if x == 10:
                    return 6
                elif x == 127:
                    return 7
                else:
                    return 1
            elif x >= 128 and x < 191:
                if x == 172 and x2 >= 16 and x2 <= 31:
                    return 8
                else:
                    return 2
            elif x >= 192 and x <= 223:
                if x == 192 and x2 == 168:
                    return 9
                else:
                    return 3
            elif x >= 224 and x <= 239:
                return 4
            elif x >= 240 and x <= 254:
                return 5
            elif x == 255:
                return 10
        else:
            return 16


def date2time(date):
    if date is None:
        return 0
    else:
        date = date.replace('T', ' ')
        date = date.replace('\u200e', '')

        time = parse('1601-1-1 00:00:00')
        date = parse(date[:-4])

        win32time = int((date - time).total_seconds() * 10000000)

        return win32time

def VN2int(ValueName):
    # remove all path value 
    if ValueName is None:
        return 0
    for index, value in enumerate(VNlist, 2):
        if ValueName == value:
            return index
    return 1

def TS2int(tds):
    if tds == None:
        return 0
    else:
        tds1= int(tds,16)
        return tds1

def v2o(value,length):
    fl = [0]*length
    if value < 0:
        return fl
    fl[value] = 1
    return fl

#ProcessTokenIsElevated
def bool2vec(value): #[0,1]

    value = str(value)
    if value == '0': 
        return [1,0] 
    elif value == '1':
        return [0,1] 
    else:
        return [0]*2

#ProcessTokenElevationType
def ptet2vec(value): # [1,2,3]

    value = str(value)

    if value == '1':
        return [1,0,0]
    elif value == '2': 
        return [0,1,0]
    elif value == '3':
        return [0,0,1]
    else:
        return [0,0,0]

#SubProcessTag
def spt2vec(value): # SubProcessTag = 0 for no service , else can be any service
    #(https://artofpwn.com/2017/06/05/phant0m-killing-windows-event-log.html)
    length = 175
    spt_vec = [0]*length

    value = str(value)

    if value == "117":
        spt_vec[0] = 1
    if value == "38":
        spt_vec[1] = 1
    if value == "47":
        spt_vec[2] = 1
    if value == "260":
        spt_vec[3] = 1
    if value == "187":
        spt_vec[4] = 1
    if value == "82":
        spt_vec[5] = 1
    if value == "89":
        spt_vec[6] = 1
    if value == "6":
        spt_vec[7] = 1
    if value == "170":
        spt_vec[8] = 1
    if value == "217":
        spt_vec[9] = 1
    if value == "225":
        spt_vec[10] = 1
    if value == "98":
        spt_vec[11] = 1
    if value == "172":
        spt_vec[12] = 1
    if value == "79":
        spt_vec[13] = 1
    if value == "148":
        spt_vec[14] = 1
    if value == "110":
        spt_vec[15] = 1
    if value == "95":
        spt_vec[16] = 1
    if value == "102":
        spt_vec[17] = 1
    if value == "121":
        spt_vec[18] = 1
    if value == "144":
        spt_vec[19] = 1
    if value == "274":
        spt_vec[20] = 1
    if value == "57":
        spt_vec[21] = 1
    if value == "120":
        spt_vec[22] = 1
    if value == "127":
        spt_vec[23] = 1
    if value == "100":
        spt_vec[24] = 1
    if value == "145":
        spt_vec[25] = 1
    if value == "203":
        spt_vec[26] = 1
    if value == "216":
        spt_vec[27] = 1
    if value == "277":
        spt_vec[28] = 1
    if value == "18":
        spt_vec[29] = 1
    if value == "167":
        spt_vec[30] = 1
    if value == "130":
        spt_vec[31] = 1
    if value == "171":
        spt_vec[32] = 1
    if value == "125":
        spt_vec[33] = 1
    if value == "269":
        spt_vec[34] = 1
    if value == "88":
        spt_vec[35] = 1
    if value == "122":
        spt_vec[36] = 1
    if value == "247":
        spt_vec[37] = 1
    if value == "40":
        spt_vec[38] = 1
    if value == "69":
        spt_vec[39] = 1
    if value == "74":
        spt_vec[40] = 1
    if value == "285":
        spt_vec[41] = 1
    if value == "156":
        spt_vec[42] = 1
    if value == "230":
        spt_vec[43] = 1
    if value == "181":
        spt_vec[44] = 1
    if value == "126":
        spt_vec[45] = 1
    if value == "49":
        spt_vec[46] = 1
    if value == "44":
        spt_vec[47] = 1
    if value == "178":
        spt_vec[48] = 1
    if value == "243":
        spt_vec[49] = 1
    if value == "56":
        spt_vec[50] = 1
    if value == "176":
        spt_vec[51] = 1
    if value == "275":
        spt_vec[52] = 1
    if value == "0":
        spt_vec[53] = 1
    if value == "194":
        spt_vec[54] = 1
    if value == "15":
        spt_vec[55] = 1
    if value == "94":
        spt_vec[56] = 1
    if value == "268":
        spt_vec[57] = 1
    if value == "115":
        spt_vec[58] = 1
    if value == "51":
        spt_vec[59] = 1
    if value == "124":
        spt_vec[60] = 1
    if value == "199":
        spt_vec[61] = 1
    if value == "113":
        spt_vec[62] = 1
    if value == "154":
        spt_vec[63] = 1
    if value == "138":
        spt_vec[64] = 1
    if value == "153":
        spt_vec[65] = 1
    if value == "259":
        spt_vec[66] = 1
    if value == "166":
        spt_vec[67] = 1
    if value == "7":
        spt_vec[68] = 1
    if value == "63":
        spt_vec[69] = 1
    if value == "41":
        spt_vec[70] = 1
    if value == "75":
        spt_vec[71] = 1
    if value == "131":
        spt_vec[72] = 1
    if value == "78":
        spt_vec[73] = 1
    if value == "134":
        spt_vec[74] = 1
    if value == "35":
        spt_vec[75] = 1
    if value == "24":
        spt_vec[76] = 1
    if value == "201":
        spt_vec[77] = 1
    if value == "96":
        spt_vec[78] = 1
    if value == "161":
        spt_vec[79] = 1
    if value == "200":
        spt_vec[80] = 1
    if value == "188":
        spt_vec[81] = 1
    if value == "211":
        spt_vec[82] = 1
    if value == "276":
        spt_vec[83] = 1
    if value == "58":
        spt_vec[84] = 1
    if value == "182":
        spt_vec[85] = 1
    if value == "270":
        spt_vec[86] = 1
    if value == "65":
        spt_vec[87] = 1
    if value == "226":
        spt_vec[88] = 1
    if value == "116":
        spt_vec[89] = 1
    if value == "184":
        spt_vec[90] = 1
    if value == "222":
        spt_vec[91] = 1
    if value == "53":
        spt_vec[92] = 1
    if value == "175":
        spt_vec[93] = 1
    if value == "244":
        spt_vec[94] = 1
    if value == "251":
        spt_vec[95] = 1
    if value == "234":
        spt_vec[96] = 1
    if value == "68":
        spt_vec[97] = 1
    if value == "111":
        spt_vec[98] = 1
    if value == "235":
        spt_vec[99] = 1
    if value == "273":
        spt_vec[100] = 1
    if value == "54":
        spt_vec[101] = 1
    if value == "252":
        spt_vec[102] = 1
    if value == "256":
        spt_vec[103] = 1
    if value == "257":
        spt_vec[104] = 1
    if value == "28":
        spt_vec[105] = 1
    if value == "286":
        spt_vec[106] = 1
    if value == "240":
        spt_vec[107] = 1
    if value == "250":
        spt_vec[108] = 1
    if value == "27":
        spt_vec[109] = 1
    if value == "46":
        spt_vec[110] = 1
    if value == "25":
        spt_vec[111] = 1
    if value == "149":
        spt_vec[112] = 1
    if value == "17":
        spt_vec[113] = 1
    if value == "164":
        spt_vec[114] = 1
    if value == "52":
        spt_vec[115] = 1
    if value == "10":
        spt_vec[116] = 1
    if value == "84":
        spt_vec[117] = 1
    if value == "233":
        spt_vec[118] = 1
    if value == "9":
        spt_vec[119] = 1
    if value == "190":
        spt_vec[120] = 1
    if value == "284":
        spt_vec[121] = 1
    if value == "261":
        spt_vec[122] = 1
    if value == "231":
        spt_vec[123] = 1
    if value == "183":
        spt_vec[124] = 1
    if value == "236":
        spt_vec[125] = 1
    if value == "262":
        spt_vec[126] = 1
    if value == "177":
        spt_vec[127] = 1
    if value == "90":
        spt_vec[128] = 1
    if value == "192":
        spt_vec[129] = 1
    if value == "109":
        spt_vec[130] = 1
    if value == "39":
        spt_vec[131] = 1
    if value == "280":
        spt_vec[132] = 1
    if value == "189":
        spt_vec[133] = 1
    if value == "165":
        spt_vec[134] = 1
    if value == "204":
        spt_vec[135] = 1
    if value == "83":
        spt_vec[136] = 1
    if value == "143":
        spt_vec[137] = 1
    if value == "212":
        spt_vec[138] = 1
    if value == "249":
        spt_vec[139] = 1
    if value == "267":
        spt_vec[140] = 1
    if value == "141":
        spt_vec[141] = 1
    if value == "202":
        spt_vec[142] = 1
    if value == "114":
        spt_vec[143] = 1
    if value == "8":
        spt_vec[144] = 1
    if value == "193":
        spt_vec[145] = 1
    if value == "62":
        spt_vec[146] = 1
    if value == "272":
        spt_vec[147] = 1
    if value == "283":
        spt_vec[148] = 1
    if value == "103":
        spt_vec[149] = 1
    if value == "32":
        spt_vec[150] = 1
    if value == "5":
        spt_vec[151] = 1
    if value == "221":
        spt_vec[152] = 1
    if value == "22":
        spt_vec[153] = 1
    if value == "48":
        spt_vec[154] = 1
    if value == "30":
        spt_vec[155] = 1
    if value == "16":
        spt_vec[156] = 1
    if value == "20":
        spt_vec[157] = 1
    if value == "14":
        spt_vec[158] = 1
    if value == "76":
        spt_vec[159] = 1
    if value == "224":
        spt_vec[160] = 1
    if value == "61":
        spt_vec[161] = 1
    if value == "55":
        spt_vec[162] = 1
    if value == "12":
        spt_vec[163] = 1
    if value == "59":
        spt_vec[164] = 1
    if value == "99":
        spt_vec[165] = 1
    if value == "186":
        spt_vec[166] = 1
    if value == "106":
        spt_vec[167] = 1
    if value == "67":
        spt_vec[168] = 1
    if value == "36":
        spt_vec[169] = 1
    if value == "245":
        spt_vec[170] = 1
    if value == "263":
        spt_vec[171] = 1
    if value == "86":
        spt_vec[172] = 1
    if value == "197":
        spt_vec[173] = 1
    if value == "248":
        spt_vec[174] = 1
    return spt_vec

#Status
def status2vec(value): 
    status = [0]* 34

    if value == "0xC0000275":
        status[0]=1
    if value == "0xC00000BB":
        status[1]=1
    if value == "0xC0000181":
        status[2]=1
    if value == "0xC0000061":
        status[3]=1
    if value == "0x104":
        status[4]=1
    if value == "0xC00002F0":
        status[5]=1
    if value == "0x368":
        status[6]=1
    if value == "0xC0000043":
        status[7]=1
    if value == "0x80000005":
        status[8]=1
    if value == "0x216":
        status[9]=1
    if value == "0xC0000035":
        status[10]=1
    if value == "0xC0000033":
        status[11]=1
    if value == "0xC0000010":
        status[12]=1
    if value == "0xC0000463":
        status[13]=1
    if value == "0x0":
        status[14]=1
    if value == "0x40000016":
        status[15]=1
    if value == "0xC01C0004":
        status[16]=1
    if value == "0xC0000023":
        status[17]=1
    if value == "0xC00000BA":
        status[18]=1
    if value == "0xC0000120":
        status[19]=1
    if value == "0xC000000D":
        status[20]=1
    if value == "0xC00000E2":
        status[21]=1
    if value == "0xC0000011":
        status[22]=1
    if value == "0xC0000123":
        status[23]=1
    if value == "0xC0000121":
        status[24]=1
    if value == "0x8000001A":
        status[25]=1
    if value == "0xC0000022":
        status[26]=1
    if value == "0xC000003A":
        status[27]=1
    if value == "0x80000006":
        status[28]=1
    if value == "0xC0000103":
        status[29]=1
    if value == "0xC0000101":
        status[30]=1
    if value == "0xC0000034":
        status[31]=1
    if value == "0x10C":
        status[32]=1
    if value == "0xC000000F":
        status[33]=1

    return status

#StatusCode
def sc2vec(value):

    value = str(value)

    if value == "0" or value == "0x0" :
        return [1,0,0,0]
    elif value == "1":
        return [0,1,0,0]
    elif value == "0xC000003A":
        return [0,0,1,0]
    elif value == "1073807364":
        return [0,0,0,1]
    else:
        return [0]*4

#Disposition
def disp2vec(value): #[0,1,2]
    # if value == None or value == '':
        # return [0]*3

    value = str(value)

    if value == '0':
        return [1,0,0]
    elif value == '1':
        return [0,1,0]
    elif value == '2':
        return [0,0,1]
    else:
        return [0]*3


#Mandatory Label
def mlabel2vec(value): 
    if value == None or value == '':
        return [0]*5
    if value == "S-1-16-4096":
        return [1,0,0,0,0]
    if value == "S-1-16-8192":
        return [0,1,0,0,0]
    if value == "S-1-16-16384":
        return [0,0,1,0,0]
    if value == "S-1-16-12288":
        return [0,0,0,1,0]
    else:
        return [0,0,0,0,1]

#PackageRelativeAppId
def appid2vec(value): 
    if value == None or value == '':
        return [0]* 7
    if "microsoft" in value.lower():
        return [1,0,0,0,0,0,0]
    if "runtimebroker07f4358a809ac99a64a67c1" in value:
        return [0,1,0,0,0,0,0]
    if "App" in value:
        return [0,0,1,0,0,0,0]
    if "CortanaUI" in value:
        return [0,0,0,1,0,0,0]
    if "ppleae38af2e007f4358a809ac99a64a67c1" in value: # Faulting package-relative application ID: (https://www.dell.com/community/Windows-8/Live-comm-exe-errors-in-event-viewer/td-p/4349312)
        return [0,0,0,0,1,0,0]
    if "ShellFeedsUI" in value:
        return [0,0,0,0,0,1,0]
    else:
        return [0,0,0,0,0,0,1]

ValueNamepath = '/data/d1/pwakodi1/VN.txt'
VolumeNamepath = '/data/d1/pwakodi1/VoN.txt'
with open(ValueNamepath, 'r') as VN:
    line = VN.read().strip()
    VNlist = line.split("\n")

with open(VolumeNamepath,'r') as VoN:
    line = VoN.read().strip()
    VoNlist = line.split("\n")


def second_step(root_path):
    node_dict_raw = {}
    edge_dict_raw = {}
    node_dict_encoded = {}
    edge_dict_encoded = {}
    # init dict value 
    update_dict(root_path, node_dict_raw, edge_dict_raw)
    
    ####################################################################################################################################
    
    node_all_attrs_defaultvals_dict = {'RelativeName':[0]*20, 'FileName':[0]*10, 'daddr':0, # daddr is a string value- 
                                          'ProcessID':0, 'ProcessId':0,
                                         'ThreadID':0, 'ThreadId':0}
    
    edge_all_attrs_defaultvals_dict = {#'StatusCode': [0]*4,
                                    'Task Name': [0]*62,
                                    #'ReadOperationCount':0,'WriteOperationCount':0,
                                    #'ReadTransferKiloBytes':0,'WriteTransferKiloBytes':0,
                                    #'size':0, 'Irp':0, 'ImageSize':0,'HandleCount':0, 
                                    'Opcode':0,'TimeStamp':0, 'ImageName':0,
                                    # 'ProcessTokenElevationType':[0]*3,
                                    #'ProcessTokenIsElevated': [0]*2, 'SubProcessTag': [0]*175,
                                    #'Status':[0]*34,'Disposition':[0]*3,
                                    }
    #'PackageRelativeAppId':[0]*7 and 'MandatoryLabel':[0]*5 : not in silketw
###################################################################################################################################
    v1_list = []
    # print("|||"*200)
    # print("node_dict start")
    # print("|||"*200)

    for k, v in node_dict_raw.items():
        # print("-"*150)
        # print(f"k: {k}\n v: {v}\n", flush= True)

        #node_dict_encoded.update({k:node_all_attrs_defaultvals_dict}) 
        node_dict_encoded[k] =dict(zip( list(node_all_attrs_defaultvals_dict.keys()), node_all_attrs_defaultvals_dict.values() )) # 4th Dec, 22

        ##################################################################################################################
        
        # print("spot1")
        # print( f"{k} : {node_dict[k]['ProcessTokenElevationType']}")

        for k1,v1 in v.items():
            # print("spot2")
            # print( f"{k} : {node_dict[k]['ProcessTokenElevationType']}")
            #print(k1)
            #my_set_2.add(k1)
            ## return list : FN2int, RN2int,IN2int,TN2int
            ## return int : Hex2Dec, str2int,address2dec,date2time,N2int
            ## return hex int : TS2int
            if k1 == 'FileName': #or k1 == 'FilePath': #TODO: look for all types of FN to get most frequent values
                # JY @ 2022-10-23: 'FileName' is a onehot-attribute and FN2int returns a list.
                node_dict_encoded[k][k1]= FN2int(v1)
                ## ignore based on attribute2onehot- line 98
            if k1 == 'RelativeName': #TODO: look for all types of RN to get most frequent values
                # JY @ 2022-10-23: 'RelativeName' is a onehot-attribute and RN2int returns a list.
                node_dict_encoded[k][k1]= RN2int(v1)  
                ## ignore based on attribute2onehot- line 98            

            # if k1 in {'FileObject', 'KeyObject'}:
                # node_dict[k][k1]= Hex2Dec(v1)


            if k1 == 'daddr': # or k1 == 'saddr':
                #node_dict[k][k1]= address2dec(v1)
                ## UPDATED-----------------------------
                # JY @ 2022-10-23: 'daddr' is a onehot-attribute and address2dec DOES NOT return a list.
                # Thus, onehot-encode as follows.
                value = str2int(v1) # TODO: look for the string format for all types of daddr values
                # v = address2dec(v1)
                # value = v2o(v,6)
                node_dict_encoded[k][k1] = value

            #----- all pool attributes _-----###

            if k1 in {'ProcessID','ProcessId','ThreadID','ThreadId'}:
                if v1 != None:
                    node_dict_encoded[k][k1] = v1
            #--------------------------------###
        # pprint.pprint(node_dict_encoded[k], width = 1000)

    # print("|||"*200)
    # print("node_dict end")
    # print("|||"*200)

    # print("|||"*200)
    # print("edge_dict start")
    # print("|||"*200)
    for k, v in edge_dict_raw.items(): # k event uid v= {attr: value}
        #my_set_3.add(k)
        ##################################################################################################################
        # Added by JY @ 2022-10-22

        # Idea
        # dict(zip(edge_all_attrs, node_all_attrs_defaultvals)) == {"Task Name":0 ,"FilePath":[0,0,0,0,0,0,0,0,0,0,0], .... all possible }
        # edge_dict[k] ==  { "Task Name": "PROCESSSTART"}
        # after assignment
        # edge_dict[k] == {"Task Name": "PROCESSSTART","FilePath":[0,0,0,0,0,0,0,0,0,0,0], .... all possible  }

        #edge_dict_encoded.update({k:edge_all_attrs_defaultvals_dict}) 
        # print("-"*150)
        # print(f"k: {k}\n v: {v}\n", flush= True)
        # 
        edge_dict_encoded[k] = dict(zip( list(edge_all_attrs_defaultvals_dict.keys()), edge_all_attrs_defaultvals_dict.values() )) # 4th Dec, 22
        ##################################################################################################################


        for k1,v1 in v.items():
            # if k1 in {'ReadOperationCount','WriteOperationCount','ReadTransferKiloBytes',
                    # 'WriteTransferKiloBytes','size','HandleCount'}:
                # if v1 != None: 
                    # edge_dict_encoded[k][k1]= v1

            if k1 == 'Task Name':
                # UPDATED -- by JY
                #edge_dict[k][k1]= TN2int(v1)
                # JY @ 2022-10-23: 'Task Name' is a onehot-attribute BUT TN2int DOES NOT return a list.
                #                   Thus, onehot-encode as follows.
                
                edge_dict_encoded[k][k1] = TN2int(v1) # edge_dict_encoded is already encoded at line 1081, so we used 'v'

            #-------------Newly Added edge attrs-------------#

            # if k1 == 'StatusCode':
                # edge_dict_encoded[k][k1]= sc2vec(v1)
            # if k1 == 'Irp' or k1 == 'ImageSize':
                # edge_dict_encoded[k][k1]= Hex2Dec(v1,k1)
            #--------- edge pool attrs------###
            if k1 in {'TimeStamp','ImageName'}:
                if v1 != None: 
                    edge_dict_encoded[k][k1] = v1
            if k1 == 'Opcode':
                if v1 != None: 
                    edge_dict_encoded[k][k1] = v1

            #----- Updated edge attrs---------### 4th Dec, 22
            # if k1 == 'ProcessTokenElevationType':
                # edge_dict_encoded[k][k1] = ptet2vec(v1)
            # if k1 == 'ProcessTokenIsElevated':
                # edge_dict_encoded[k][k1] = bool2vec(v1)
            # if k1 == 'SubProcessTag':
                # edge_dict_encoded[k][k1] = spt2vec(v1)
            # if k1 == 'Status':
                # edge_dict_encoded[k][k1] = status2vec(v1)
            # if k1 == 'Disposition':
                # edge_dict_encoded[k][k1] = disp2vec(v1)
            # if k1 == 'MandatoryLabel':
                # edge_dict_encoded[k][k1] = mlabel2vec(v1)
            # if k1 == 'PackageRelativeAppId':
                # edge_dict_encoded[k][k1] = appid2vec(v1)

        # pprint.pprint(edge_dict_encoded[k], width = 1000)
    # print("|||"*200)
    # print("edge_dict end")
    # print("|||"*200)

    # print("*"*200, flush= True)
    # print("*"*200, flush= True)
    # print("*"*200, flush= True)
    # print("DONE : Final ver 'node_dict_encoded'\n", flush= True)
    # pprint.pprint(node_dict_encoded, width = 1000)

    # print("*"*200, flush= True)
    # print("*"*200, flush= True)
    # print("*"*200, flush= True)
    # print("="*150, flush= True)
    # print("DONE : Final ver 'edge_dict_encoded'\n", flush= True)
    # pprint.pprint(edge_dict_encoded, width = 1000)

    #################################################################################################################

    #node_attrs = node_dict
    #edge_attrs = edge_dict

    #node_path = os.path.join('data',idx,"node_attribute.pickle")
    # we store everything in root path. We can change it later 
    #pickle_path = root_path
    
    node = open(os.path.join(root_path,"global_node_attribute.pickle"), "wb")
    pickle.dump(node_dict_encoded,node)
    node.close()
    
    #new_node = dot.attribute2onehot(node_dict,'node')
    # node = open(os.path.join(pickle_path,"new_node_attribute.pickle"), "wb")
    # pickle.dump(new_node,node)
    # node.close()

    #edge_path = os.path.join('data',idx,"edge_attribute.pickle")
    
    edge = open(os.path.join(root_path,"global_edge_attribute.pickle"), "wb")
    pickle.dump(edge_dict_encoded,edge)
    edge.close()
    
    # new_edge = dot.attribute2onehot(edge_attrs,'edge')
    # edge = open(os.path.join(pickle_path,"new_edge_attribute.pickle"), "wb")
    # pickle.dump(new_edge,edge)
    # edge.close()

if __name__ == "__main__":
    main()


