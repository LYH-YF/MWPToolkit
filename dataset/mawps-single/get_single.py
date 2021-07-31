import re
import json


def write_json_data(data, filename):
    """
    write data to a json file
    """
    with open(filename, 'w+', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    f.close()


def read_json_data(filename):
    '''
    load data from a json file
    '''
    f = open(filename, 'r', encoding="utf-8")
    return json.load(f)

x='25000.0-(1500.0*8.0)=money'
res=re.match(r'([a-zA-Z]+)',x)

datas=read_json_data('dataset/mawps-single/mawps_source.json')
new_datas=[]
for data in datas:
    if len(data['lEquations']) != 1:
        print('**',data['lEquations'])
        continue
    data['lEquations'][0]=''.join(data['lEquations'][0].split(' '))
    if data['lEquations'][0][:2]=='x=':
        new_datas.append(data)
    elif data['lEquations'][0][-2:]=='=x':
        new_datas.append(data)
    elif data['lEquations'][0][:2]=='X=':
        new_datas.append(data)
    elif data['lEquations'][0][-2:]=='=X':
        new_datas.append(data)
    elif re.search(r'^[a-zA-Z]+=',data['lEquations'][0]):
        res=re.findall(r'([a-zA-Z]+)',data['lEquations'][0])
        if len(res) > 1:
            print('>>>'+data['lEquations'][0])
            continue
        equ='x'+data['lEquations'][0][data['lEquations'][0].index('='):]
        print('--'+equ)
        data['lEquations'][0]=equ
        new_datas.append(data)
    elif re.search(r'=([a-zA-Z]+)$',data['lEquations'][0]):
        res=re.findall(r'([a-zA-Z]+)',data['lEquations'][0])
        if len(res) > 1:
            print('>>>'+data['lEquations'][0])
            continue
        equ=data['lEquations'][0][:data['lEquations'][0].index('=')+1]+'x'
        print('-'+equ)
        data['lEquations'][0]=equ
        new_datas.append(data)
    elif not re.search(r'[a-zA-Z]+',data['lEquations'][0]) and '=' not in data['lEquations'][0]:
        new_datas.append(data)
    else:
        pass
        #print(data['lEquations'][0])
print(len(new_datas))
write_json_data(new_datas,'dataset/mawps-single/mawps_single.json')