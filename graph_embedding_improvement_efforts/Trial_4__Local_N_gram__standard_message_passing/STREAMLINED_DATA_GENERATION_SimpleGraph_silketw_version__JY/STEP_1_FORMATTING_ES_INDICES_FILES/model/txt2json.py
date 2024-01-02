# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import ast
import json
import os
import re
#from dateutil.parser import parse

attributes = {
    'ProcessID',
    'ProcessId',
    'CreateTime',
    'ThreadId',
    'ThreadID',
    'FrozenProcessID',
    'Task Name',
    'TimeStamp',
    'UserStackLimit',
    'StackLimit',
    'Win32StartAddr',
    'TebBase',
    'OldPriority',
    'NewPriority',
    'StackBase',
    'SubProcessStack',
    'UserStackBase',
    'StartAddr',
    'Job',
    'ID',
    'ContainerID',
    'FileObject',
    'FileKey',
    'FilePath',
    'ByteOffset',
    'IOFlags',
    'IOSize',
    'Irp',
    'daddr',
    'saddr',
    'sport',
    'dport',
    'connid',
    'ParentProcessID',
    'ImageSize',
    'ImageCheckSum',
    'HandleCount',
    'ImageBase',
    'ImageName',
    'TimeDateStamp',
    'ExitCode',
    'SessionId',
    'ExitTime',
    'ValueName',
    'KeyObject',
    'RelativeName',
    'Index',
    'FileName',
}
no_attributes = {
    'Keyword',
    'Flags',
    'Description',

}
TASK = {
    'Task Name',
}
Task_Names = {
    'CLEANUP',
    'CLOSE',
    'CREATE',
    'CREATENEWFILE',
    'DELETEPATH',
    'DIRENUM',
    'DIRNOTIFY',
    'FLUSH',
    'FSCTL',
    'NAMECREATE',
    'NAMEDELETE',
    'OPERATIONEND',
    'QUERYINFO',
    'QUERYINFORMATION',
    'QUERYEA',
    'QUERYSECURITY',
    'READ',
    'WRITE',
    'SETDELETE',
    'SETINFORMATION',
    'PAGEPRIORITYCHANGE',
    'IOPRIORITYCHANGE',
    'CPUBASEPRIORITYCHANGE',
    'IMAGEPRIORITYCHANGE',
    'CPUPRIORITYCHANGE',
    'IMAGELOAD',
    'IMAGEUNLOAD',
    'PROCESSSTOP',
    'PROCESSSTART',
    'PROCESSFREEZE',
    'PSDISKIOATTRIBUTE',
    'PSIORATECONTROL',
    'KERNEL_NETWORK_TASK_UDPIP',
    'KERNEL_NETWORK_TASK_TCPIP',
    'MICROSOFT-WINDOWS-KERNEL-REGISTRY',
    'THREADSTART',
    'THREADSTOP',
    'THREADWORKONBEHALFUPDATE',
    'JOBSTART',
    'JOBTERMINATE',
    'LOSTEVENT',
    'PSDISKIOATTRIBUTION',
    'RENAME',
    'RENAMEPATH',
    'THIS GROUP OF EVENTS TRACKS THE PERFORMANCE OF FLUSHING HIVES',

}
hexdecimel = {
    'UserStackLimit',
    'StackLimit',
    'Win32StartAddr',
    'TebBase',
    'StackBase',
    'SubProcessStack',
    'UserStackBase',
    'StartAddr',
    'ByteOffset',
    'IOSize',
    'Irp',
    'ImageSize',
    'ImageBase',
    'FileObject',
    'FileKey',
    'IOFlags',

}
IP_address = {
    'daddr',
    'saddr',

}
TIME = {
    'ExitTime',
    'CreateTime',

}

ProviderId= {
    '{70EB4F03-C1DE-4F73-A051-33D13D5413BD}': 1,
    '{22FB2CD6-0E7B-422B-A0C7-2FAD1FD0E716}': 2,
    '{7DD42A49-5329-4832-8DFD-43D979153A88}': 3,
    '{EDD08927-9CC4-4E65-B970-C2560FB5C289}': 4,


}
def get_data(fp):
    line = fp.readline()
    json_str = json.dumps(line)
    json_strg = re.sub("\)\(([0-9]*)",'',json_str)
    data = json.loads(json_strg)

    return data

def txt2json(txt, json1):
    ft1 = open(txt, 'r', encoding = 'utf-8', errors = 'ignore')
    fj2 = open(json1, 'w', encoding = 'utf-8', errors = 'ignore')
    
    
    lines = [ line for line in ft1.readlines() if line != "\n" ]

    for line in lines:

        # Added by JY @ 2023-1-5: To get rid of '|| log-entry-timestamp-wallclock-time: 2023-01-04 23:48:45.839074 || logging-wallclock-time: 2023-01-04 23:48:46.924279"
        if line.find("||") != -1:
            line = line[:line.find("||")]#.rstrip() # SHOULD NOT DO rstrip() since following "new_str = line[i + 1:-2]" assumes there's a trailing space.
        
        what_we_dont_want = ast.literal_eval(line)[0]
        log_entry = ast.literal_eval(line)[1]
        fj2.write(str(log_entry) + "\n")
    
    
    
    '''
    #line = ft1.readline()
    #while line:
        #i = 0
        #temp = ft1.readline()


        for content in line:
            if temp != '':
                i = i + 1
                if content == ",":
                    new_str = line[i + 1:-2]
                    fj2.write(new_str + "\n")
                    break
            else:
                i = i + 1
                if content == ",":
                    if line[-1] == '\n':
                        new_str = line[i + 1:-2]
                        fj2.write(new_str + "\n")
                        break
                    else:
                        new_str = line[i + 1:-1]
                        fj2.write(new_str + "\n")
                        break

        line = temp
    '''
    ft1.close()
    fj2.close()



#reduction

def addr2dec(addr):

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
    elif x >= 128 and x <191:
        if x == 172 and x2 >= 16 and x2 <=31:
            return 8
        else:
            return 2
    elif x >= 192 and x <= 223:
        if x == 192 and x2 == 168:
            return 9
        else:
            return 3
    elif x >= 224 and x <= 239:
            return  4
    elif x >= 240 and x <= 254:
        return 5
    elif x == 255:
        return 10



def checkipv6(ipv6):
    matchobj = re.match(
        r'^\s*((([0-9A-Fa-f]{1,4}:){7}([0-9A-Fa-f]{1,4}|:))|(([0-9A-Fa-f]{1,4}:){6}(:[0-9A-Fa-f]{1,4}|((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})|:))|(([0-9A-Fa-f]{1,4}:){5}(((:[0-9A-Fa-f]{1,4}){1,2})|:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})|:))|(([0-9A-Fa-f]{1,4}:){4}(((:[0-9A-Fa-f]{1,4}){1,3})|((:[0-9A-Fa-f]{1,4})?:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){3}(((:[0-9A-Fa-f]{1,4}){1,4})|((:[0-9A-Fa-f]{1,4}){0,2}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){2}(((:[0-9A-Fa-f]{1,4}){1,5})|((:[0-9A-Fa-f]{1,4}){0,3}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(([0-9A-Fa-f]{1,4}:){1}(((:[0-9A-Fa-f]{1,4}){1,6})|((:[0-9A-Fa-f]{1,4}){0,4}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|(:(((:[0-9A-Fa-f]{1,4}){1,7})|((:[0-9A-Fa-f]{1,4}){0,5}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:)))(%.+)?\s*$',
        ipv6)
    if matchobj:
        return True
    else:
        return False
def ipseg2str(ipseglist):
    ipstr = ''
    for ipseg in ipseglist:
        if len(ipseg) == 1:
            ipseg = '000' + ipseg
        elif len(ipseg) == 2:
            ipseg = '00' + ipseg
        elif len(ipseg) == 3:
            ipseg = '0' + ipseg
        elif len(ipseg) == 4:
            ipseg = ipseg
        else:
            return ""
        ipstr += ipseg
    return ipstr
def noCompressipv62dec(ipv6):
    iplist = ipv6.split(":")
    if iplist:
        ipstr = ipseg2str(iplist)

        return int(ipstr, 16)
    else:
        return ""
def compressipv62dec(ipv6):
    compressList = ipv6.split("::")

    ipstr = ""
    part1 = []
    part2 = []
    if len(compressList) == 2:
        part1 = compressList[0].split(":") if compressList[0] else []
        part2 = compressList[1].split(":") if compressList[1] else []


    if part1 or part2 :
        if part1:
            ipstr += ipseg2str(part1)
            for i in range(8 - len(part1) - len(part2)):
                ipstr += '0000'
            ipstr += ipseg2str(part2)
            return int(ipstr, 16)
        else:
            ipstr = ''.join(part2)

            return int(ipstr)

def IPv62dec(ipv6):
    if checkipv6(ipv6):
        compressIndex = ipv6.find('::')

        ipv4Index = ipv6.find('.')

        if compressIndex == -1 and ipv4Index == -1:
            return noCompressipv62dec(ipv6)
        elif compressIndex != -1 and ipv4Index == -1:
            a = compressipv62dec(ipv6)
            #print(a)
            return a

#def date2time(date):
#    date = date.replace('T',' ')
#    date = date.replace('\u200e','')

#    time = parse('1601-1-1 00:00:00')
#    date = parse(date[:-4])

#    win32time = int((date-time).total_seconds()*10000000)
#    return win32time

def VN2int(ValueName):
    if ValueName != '':
        if ValueName.find('Device') != -1 or ValueName.find('C:') != -1:
            if ValueName.find('System32')!= -1 or ValueName.find('system32') != -1:
                return 1
            elif ValueName.find('Program Files')!= -1:
                return 2
            elif ValueName.find("Users")!= -1:
                return 3
            else:
                return 4
        elif ValueName[:-3] == 'yax' or ValueName[:-3] == 'rkr':
            return 5

        elif ValueName[0] == '{':
            return 6

        elif ValueName in VNlist:
            for index, value in enumerate(VNlist, 10):
                if ValueName == value:
                    return index

        elif ValueName[0] != '{' and ValueName.find('-'):
            return 7
        elif ValueName.find('.dll'):
            return 6
        else:
            return 9
    else:
        return 0

def FN2int(filename):

    if filename.find('\\WINDOWS')!= -1 or filename.find('\\Windows')!= -1:
        return 1
    elif filename.find('\\Program Files\\')!= -1 or filename.find('\\program files\\')!= -1 or filename.find('\\Program Files (x86)')!= -1 :
        return 2
    elif filename.find('\\Users')!= -1 or filename.find('\\users')!= -1:
        return 3
    elif filename.find('\\pywintrace-master')!= -1:

        return 4
    elif filename.find('\\logs')!= -1:
        return 5
    elif filename.find('\\ProgramData')!= -1:
        return 6
    elif filename.find('\\$Secure:$SDH:$INDEX_ALLOCATION')!= -1:
        return 7
    elif filename.find('\\DR0')!= -1:
        return 8
    elif filename.find('\\System Volume Information')!= -1:
        return 9
    else:
        return 0

def TN2int(taskname):
    if taskname == ' ':
        return 0
    else:
        for index, value in enumerate(Task_Names, 1):
            if taskname == value:
                return index

def IN2int(imagename):
    if imagename == '':
        return 0
    elif imagename.find('\\Windows')!= -1:
        return 1
    elif imagename.find('\\Program Files')!= -1:
        return 2
    elif imagename.find('\\User')!= -1:
        return 3
    elif imagename.find('\\') == -1:
        return 4
    else:
        return 5

def RN2int(RelativeName):
    if RelativeName =='':
        return 0
    elif RelativeName.find('SOFTWARE') != -1:
        return 1
    elif RelativeName.find('\\REGISTRY') != -1:
        return 2
    elif RelativeName[0] == '{':
        return 3
    else:
        return 4
def VoN2int(VolumeName):
    for index, value in enumerate(VoNlist, 1):
        if VolumeName == value:
            return index
def MandatoryLabel2num(label):
    if label == 'S-1-16-4096':
        return 1
    elif label == 'S-1-16-8192':
        return 2
    elif label == 'S-1-16-16384':
        return 3
    else:
        return 4

def Hive(Point):
    if Point.find('\\User') != -1:
        return 1
    elif Point.find('\\WC') != -1:
        return 2
    elif Point.find('\\{') != -1:
        return 3
    else:
        return 4

def main():
    

        txt2json(f1, f2)
    

    
if __name__ == "__main__":
    main()



