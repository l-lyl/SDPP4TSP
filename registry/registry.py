from model.ism import ISM
from model.ssm import SSM
from model.snsrec import SNSRec

'''
Registry models, maing these model cls names can be parsed by argument parser.
Example:
registry = {
    'model_cls_name_1': ModelCls1,
    'model_cls_name_2': ModelCls2
}
'''

registry = {
    'SNSRec': SNSRec 
}