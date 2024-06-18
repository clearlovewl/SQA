'''
Author: ll 
LastEditors: ll
无问西东
'''
from init import *
# import lightning.pytorch.cli 
# lightning.pytorch.cli.ReduceLROnPlateau 

def getcliarg():       
    temp=sys.argv[0]  
    argvlists=[] 
    # datas=['VCLpsp','LIVECpsp','koniqpsp','live']  
    # datas=['VCLpsp','TIDpsp','LIVECpsp','koniqpsp','live']  
    # datas=['VCLpsp','LIVECpsp','koniqpsp']
    # datas=['VCLpsp']
    # models=['at_DBCNN','at_NIMA','at_resnet'5,'at_Resnetlwtapre','at_resnetde','at_piece'] 
    # datas=['live'] 
    # datas=['TIDpsp','VCLpsp','LIVECpsp','kadid','koniqpsp']  
    # datas=['ESPL','kadid','SHRQ_Regular','SAUD','LIEQ','TIDpsp','live']   

    # datas=['LIEQ','ESPL','SAUD','SHRQ_Regular']    
    # datas=['SHRQ_Regular']    
    # datas=['CSIQ','VCLpsp','LIVECpsp','koniqpsp']   
    datas=['koniqpsp']  
    
    # datas=['SHRQ_Regular','ESPL','LIEQ','SAUD']  
    # models=['at_piece','at_DBCNN','at_NIMA','at_resnet','at_Resnetlwtapre','at_resnetde']  
    # models=['at_sin','at_DBCNN','at_NIMA','at_resnet','at_Resnetlwtapre','at_resnetde']  
    # models=['OOD_DBCNN']   
    # models=['OOD_resnet','OOD_TOPIQ']   
    # models=['OOD_ResNet','OOD_ResNetinv']   
    # models=['OOD_ResNetinv','OOD_ResNet'] 
    # models=['OOD_DBCNNinv','OOD_DBCNN' ] 
    # models=['OOD_TOPIQinv','OOD_TOPIQ' ] 
    # models=['ZeroShot_TOPIQ'  ] 6
    # models=['OOD_TReS','OOD_TReSinv' ] 
    # models=['OOD_Hypernet'] 
    # for lrtype in ['SGD','Adam']:
    #     for lr in ['1e-4','1e-3','1e-5','1e-2']:   
    #      
    # tensorboard --logdir ./logs/VQdiff
    exs=[


        # {'task':'LCMOS','model':'GDBC_DEIQT','eta': [0.0,1.0] ,'debias': 'False' },  
        # {'task':'LCMOS','model':'GDBC_DEIQT','eta': [1.0,1.0] ,'debias': 'False' }, 
        # {'task':'LCMOS','model':'GDBC_DEIQT','eta': [1.0,1.0] ,'debias': 'True' }, 

        # {'task':'LCMOS','model':'GDBC_DEIQT','eta': [0.0,1.0] ,'debias': 'True' },  
        
        {'task':'LCMOS','model':'GDBC_MAMIQA','eta': [0.0,1.0] ,'debias': 'False' },  
        {'task':'LCMOS','model':'GDBC_MAMIQA','eta': [1.0,1.0] ,'debias': 'False' }, 
        {'task':'LCMOS','model':'GDBC_MAMIQA','eta': [1.0,1.0] ,'debias': 'True' }, 

        {'task':'LCMOS','model':'GDBC_MAMIQA','eta': [0.0,1.0] ,'debias': 'True' },  


        # {'task':'LCMOS','model':'GDBC_CLIPIQA','eta': [0.0,1.0] ,'debias': 'False' },  
        # {'task':'LCMOS','model':'GDBC_CLIPIQA','eta': [1.0,1.0] ,'debias': 'False' }, 
        # {'task':'LCMOS','model':'GDBC_CLIPIQA','eta': [1.0,1.0] ,'debias': 'True' }, 

        # {'task':'LCMOS','model':'GDBC_CLIPIQA','eta': [0.0,1.0] ,'debias': 'True' },  
        

        # {'task':'LCMOS','model':'GDBC_DEIQT','eta': [0.0,1.0] ,'debias': 'False' },  
        # {'task':'LCMOS','model':'GDBC_DEIQT','eta': [1.0,1.0] ,'debias': 'False' }, 

        # {'task':'LCMOS','model':'GDBC_DEIQT','eta': [0.0,1.0] ,'debias': 'True' },  
        # {'task':'LCMOS','model':'GDBC_DEIQT','eta': [1.0,1.0] ,'debias': 'True' }, 





        # {'models':'OOD_TReS','p':'0.00','alpha':'0.0'},
        # {'models':'OOD_TReSinv','p':'0.1','alpha':'0.0000'},
        # {'models':'OOD_TReSinv','p':'0.1',' alpha':'0.1'},
        # {'models':'ZeroShot_TOPIQ','p':'0.2','alpha':'0.1'},
        # {'models':'VQresnet','p':'0.2','alpha':'0.1'},\
        # {'models':'VQdiff','p':'0.2','alpha':'0.1'}, 
        # {'models':'VQdiff','encoder': 'MANIQ384'  },
        # {'models':'VQdiff','encoder': 'MANIQ' }, 
        # {'models':'VQdiff','encoder': 'UNIQUE'  },
        # {'models':'VQdiff','encoder': 'TreS'  },
        # {'models':'VQdiff','encoder': 'TOPIQ'  },
        
        # 
        # {'models':'VQdiff','encoder': 'VQdiffnodisen' }, 
        
        # {'models':'VQdiff','encoder': 'refiqa' },  
        # {'models':'VQdiff','encoder': 'VQdiff' }, 
        # {'task':'LCMOS','model':'resnet','param':str({ 'eta': [0.0,1.0]  , 'iqatask': 'iqa'  }) }, 

        # {'task':'LCMOS','model':'GDBC_LWF','eta': [0.0,1.0] ,'debias': 'False' },  
        # {'task':'LCMOS','model':'GDBC_LWF','eta': [1.0,1.0] ,'debias': 'False' }, 
        # {'task':'LCMOS','model':'GDBC_LWF','eta': [1.0,1.0] ,'debias': 'True' }, 

        # {'task':'LCMOS','model':'GDBC_LWF','eta': [0.0,1.0] ,'debias': 'True' },  

        # {'task':'LCMOS','model':'GDBC_UNIQUE','eta': [0.0,1.0]  ,'debias': 'False'},  
        #  {'task':'LCMOS','model':'GDBC_UNIQUE','eta': [1.0,1.0]  , 'debias': 'False' }, 
        #  {'task':'LCMOS','model':'GDBC_UNIQUE','eta': [1.0,1.0]  , 'debias': 'True' }, 

        # # {'task':'LCMOS','model':'GDBC_UNIQUE','eta': [0.0,1.0]  ,'debias': 'True'},  

        # # {'task':'LCMOS','model':'GDBC_UNIQUE','eta': [0.0,1.0]  ,'debias': 'False'},  
        # #  {'task':'LCMOS','model':'GDBC_UNIQUE','eta': [1.0,1.0]  , 'debias': 'False' }, 
        #  {'task':'LCMOS','model':'GDBC_TOPIQ','eta': [1.0,1.0]  , 'debias': 'True' }, 

        # # {'task':'LCMOS','model':'GDBC_UNIQUE','eta': [0.0,1.0]  ,'debias': 'True'},  
        

        #  {'task':'LCMOS','model':'GDBC_MANIQ','eta': [1.0,1.0]  , 'debias': 'True' }, 

        # {'task':'LCMOS','model':'GDBC_MANIQ','eta': [0.0,1.0] ,'debias': 'False' },  
        # {'task':'LCMOS','model':'GDBC_MANIQ','eta': [1.0,1.0] ,'debias': 'False' }, 

        # {'task':'LCMOS','model':'GDBC_MANIQ','eta': [0.0,1.0] ,'debias': 'True' },  
        # {'task':'LCMOS','model':'GDBC_MANIQ','eta': [1.0,1.0] ,'debias': 'True' }, 

        # {'task':'LCMOS','model':'GDBC_resnet','eta': [0.0,1.0] ,'debias': 'False' },  
        # {'task':'LCMOS','model':'GDBC_resnet','eta': [1.0,1.0] ,'debias': 'False' }, 

        # {'task':'LCMOS','model':'GDBC_resnet','eta': [0.0,1.0] ,'debias': 'True' },  
        # {'task':'LCMOS','model':'GDBC_resnet','eta': [1.0,1.0] ,'debias': 'True' }, 


        # {'task':'LCMOS','model':'GDBC_DBCNN','eta': [0.0,1.0] ,'debias': 'False' },  
        # {'task':'LCMOS','model':'GDBC_DBCNN','eta': [1.0,1.0] ,'debias': 'False' }, 

        # {'task':'LCMOS','model':'GDBC_DBCNN','eta': [0.0,1.0] ,'debias': 'True' },  
        # {'task':'LCMOS','model':'GDBC_DBCNN','eta': [1.0,1.0] ,'debias': 'True' }, 

        # {'task':'LCMOS','model':'GDBC_NIMA','eta': [0.0,1.0] ,'debias': 'False' },  
        # {'task':'LCMOS','model':'GDBC_NIMA','eta': [1.0,1.0] ,'debias': 'False' }, 

        # {'task':'LCMOS','model':'GDBC_NIMA','eta': [0.0,1.0] ,'debias': 'True' },  
        # {'task':'LCMOS','model':'GDBC_NIMA','eta': [1.0,1.0] ,'debias': 'True' }, 

        # {'task':'LCMOS','model':'GDBC_NIMA','eta': [0.0,1.0] ,'debias': 'False' },  
        # {'task':'LCMOS','model':'GDBC_NIMA','eta': [1.0,1.0] ,'debias': 'False' }, 



        
        # {'task':'LCMOS','model':'GDBC_HyperNet','eta': [0.0,1.0]  ,'debias': 'False'},  
        # {'task':'LCMOS','model':'GDBC_HyperNet','eta': [1.0,1.0]  , 'debias': 'False' }, 

       
        # {'task':'LCMOS','model':'GDBC_DBCNN','eta': [0.0,1.0]  ,'debias': 'False'},  
        #  {'task':'LCMOS','model':'GDBC_DBCNN','eta': [1.0,1.0]  , 'debias': 'False' }, 

       


        # {'task':'LCMOS','model':'GDBC_TOPIQ','eta': [0.0,1.0] ,'debias': 'False' },  
        # {'task':'LCMOS','model':'GDBC_TOPIQ','eta': [1.0,1.0] ,'debias': 'False' }, 
        # {'task':'LCMOS','model':'GDBC_TOPIQ','eta': [0.0,1.0] ,'debias': 'True' },  
        # {'task':'LCMOS','model':'GDBC_TOPIQ','eta': [1.0,1.0] ,'debias': 'True' }, 

        # {'task':'LCMOS','model':'GDBC_TreS','eta': [0.0,1.0] ,'debias': 'False' },  
        # {'task':'LCMOS','model':'GDBC_TreS','eta': [1.0,1.0] ,'debias': 'False' },  

        # {'task':'LCMOS','model':'GDBC_MANIQ','eta': [0.0,1.0] ,'debias': 'False' },  
        # {'task':'LCMOS','model':'GDBC_MANIQ','eta': [1.0,1.0] ,'debias': 'False' }, 
        

        # {'task':'LCMOS','model':'GDBC_MANIQ','eta': [1.0,1.0] ,'debias': 'True' }, 
        # {'task':'LCMOS','model':'GDBC_MANIQ','eta': [0.0,1.0] ,'debias': 'True' },  




        # {'task':'LCMOS','model':'GDBC_resnet','param':str({ 'eta': [1.0,1.0]  , 'iqatask': 'iqa'  }) }, 
        ]          
    
    # +['--data.init_args.param.eta','['+rate+','+inten+']']
    # lrs={'SGD':['5e-1','1e-1','1e-2'],'AdamW':['3e-5','1e-4','1e-5','1e-3']}
    # lrs={'AdamW':['1e-4','5e-4','5e-5']} 
    # tensorboard --logdir ./logs/VQdiff
    # lrs={'AdamW':['2e-4']}
    # lrs={'AdamW':['2e-4']}
    # lrs={'AdamW':['1e-5']}
    # lrs={'AdamW':['1e-5']}
    # for lrtype in lrs.keys():
    #     for lr in lrs[lrtype]:

    # for debias in ['True','False']:
    for data in datas:  
        for ex in exs:    
    # for debias in ['False' ]: 
            task=ex['task'] 
            model=ex['model']   
            debias=ex['debias']
            eta=ex['eta']
            argvlist=[temp]+['--config','MODEL/'+task+'/'+'model.yaml']+['--config','DATA/'+data+'/'+data+'.yaml']+['--config','DATA/'+data+'/'+model+'.yaml']+['--model.model.debias',str(debias)]\
            +['--data.eta',str( eta)]
            # +['--data.param',str(param)]
            
            # +['--data.init_args.param.eta','['+'0.1'+','+'0.2'+']']
            # +['--data.init_args.param.eta','['+datap['rate']+','+datap['inten']+']']
            # +['--optimizer.class_path','torch.optim.'+lrtype]+['--optimizer.init_args.lr',lr]  
            pathname=['--trainer.default_root_dir', os.path.join('logs',task, data, str(eta)[1:-1] +'model'+model+'debias'+debias )]   

                # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_SHRQ_Regular_encoerVQdiff\lightning_logs\version_40\checkpoints\epoch=69-step=2520.ckpt']
                # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_0\checkpoints\epoch=69-step=14000.ckpt']
                # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_5\checkpoints\epoch=19-step=4000.ckpt']
                # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_22\checkpoints\epoch=39-step=8000.ckpt']
                # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_35\checkpoints\epoch=99-step=20000.ckpt']
                # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_57\checkpoints\epoch=109-step=22000.ckpt']
                
                
                
                
                
            ckpathname=[]
            if os.path.exists(pathname[1]):
                pass
            else:
                argvlists.append(argvlist+pathname+ckpathname)



    out=[]
    for i in     argvlists:
        out.append('\t'.join(i)       )
    printout='\n \n'.join(out) 
    print (printout)          
    return argvlists









def eta():       
    temp=sys.argv[0]  
    argvlists=[] 
    # datas=['VCLpsp','LIVECpsp','koniqpsp','live']  
    # datas=['VCLpsp','TIDpsp','LIVECpsp','koniqpsp','live']  
    # datas=['VCLpsp','LIVECpsp','koniqpsp']
    # datas=['VCLpsp']
    # models=['at_DBCNN','at_NIMA','at_resnet'5,'at_Resnetlwtapre','at_resnetde','at_piece'] 
    # datas=['live'] 
    # datas=['TIDpsp','VCLpsp','LIVECpsp','kadid','koniqpsp']  
    # datas=['ESPL','kadid','SHRQ_Regular','SAUD','LIEQ','TIDpsp','live']   

    # datas=['LIEQ','ESPL','SAUD','SHRQ_Regular']    
    # datas=['SHRQ_Regular']    
    # datas=['koniqpsp','LIVECpsp','VCLpsp','CSIQ']   
    datas=['koniqpsp']  
    
    # datas=['SHRQ_Regular','ESPL','LIEQ','SAUD']  
    # models=['at_piece','at_DBCNN','at_NIMA','at_resnet','at_Resnetlwtapre','at_resnetde']  
    # models=['at_sin','at_DBCNN','at_NIMA','at_resnet','at_Resnetlwtapre','at_resnetde']  
    # models=['OOD_DBCNN']   
    # models=['OOD_resnet','OOD_TOPIQ']   
    # models=['OOD_ResNet','OOD_ResNetinv']   
    # models=['OOD_ResNetinv','OOD_ResNet'] 
    # models=['OOD_DBCNNinv','OOD_DBCNN' ] 
    # models=['OOD_TOPIQinv','OOD_TOPIQ' ] 
    # models=['ZeroShot_TOPIQ'  ] 6
    # models=['OOD_TReS','OOD_TReSinv' ] 
    # models=['OOD_Hypernet'] 
    # for lrtype in ['SGD','Adam']:
    #     for lr in ['1e-4','1e-3','1e-5','1e-2']:   
    #      
    # tensorboard --logdir ./logs/VQdiff
    exs=[
        # {'models':'OOD_TReS','p':'0.00','alpha':'0.0'},
        # {'models':'OOD_TReSinv','p':'0.1','alpha':'0.0000'},
        # {'models':'OOD_TReSinv','p':'0.1',' alpha':'0.1'},
        # {'models':'ZeroShot_TOPIQ','p':'0.2','alpha':'0.1'},
        # {'models':'VQresnet','p':'0.2','alpha':'0.1'},\
        # {'models':'VQdiff','p':'0.2','alpha':'0.1'}, 
        # {'models':'VQdiff','encoder': 'MANIQ384'  },
        # {'models':'VQdiff','encoder': 'MANIQ' }, 
        # {'models':'VQdiff','encoder': 'UNIQUE'  },
        # {'models':'VQdiff','encoder': 'TreS'  },
        # {'models':'VQdiff','encoder': 'TOPIQ'  },
        
        # 
        # {'models':'VQdiff','encoder': 'VQdiffnodisen' }, 
        
        # {'models':'VQdiff','encoder': 'refiqa' },  
        # {'models':'VQdiff','encoder': 'VQdiff' }, 
        # {'task':'LCMOS','model':'resnet','param':str({ 'eta': [0.0,1.0]  , 'iqatask': 'iqa'  }) },  


        # {'task':'LCMOS','model':'GDBC_resnet','eta': [1.0,1.0] ,'debias': 'True' }, 
 
        # {'task':'LCMOS','model':'GDBC_TreS','eta': [1.0,1.0] ,'debias': 'True' }, 
        # {'task':'LCMOS','model':'GDBC_TreS','eta': [0.0,1.0] ,'debias': 'True' },  
        # {'task':'LCMOS','model':'GDBC_TreS','eta': [1.0,1.0] ,'debias': 'True' }, 

 
        # {'task':'LCMOS','model':'GDBC_MANIQ','eta': [1.0,1.0] ,'debias': 'True' }, 



        # {'task':'LCMOS','model':'GDBC_NIMA','eta': [1.0,1.0] ,'debias': 'True' }, 



        
        # {'task':'LCMOS','model':'GDBC_HyperNet','eta': [1.0,1.0]  , 'debias': 'True' }, 

        #  {'task':'LCMOS','model':'GDBC_DBCNN','eta': [1.0,1.0]  , 'debias': 'True' }, 

        
        # {'task':'LCMOS','model':'GDBC_MAMIQA','eta': [1.0,1.0]  , 'debias': 'True' }, 
        {'task':'LCMOS','model':'GDBC_DEIQT','eta': [1.0,1.0]  , 'debias': 'True' }, 
        # {'task':'LCMOS','model':'GDBC_LWF','eta': [1.0,1.0]  , 'debias': 'True' }, 

        # {'task':'LCMOS','model':'GDBC_TOPIQ','eta': [1.0,1.0] ,'debias': 'True' }, 









        # {'task':'LCMOS','model':'GDBC_resnet','param':str({ 'eta': [1.0,1.0]  , 'iqatask': 'iqa'  }) }, 
        ]          
    
    # +['--data.init_args.param.eta','['+rate+','+inten+']']
    # lrs={'SGD':['5e-1','1e-1','1e-2'],'AdamW':['3e-5','1e-4','1e-5','1e-3']}
    # lrs={'AdamW':['1e-4','5e-4','5e-5']} 
    # tensorboard --logdir ./logs/VQdiff
    # lrs={'AdamW':['2e-4']}
    # lrs={'AdamW':['2e-4']}
    # lrs={'AdamW':['1e-5']}
    # lrs={'AdamW':['1e-5']}
    # for lrtype in lrs.keys():
    #     for lr in lrs[lrtype]:

    # for debias in ['False','True']:
    for debias in [ 'False']:
        for eta in [  [0.8,1.0]  ,  [0.6,1.0]  ,   [0.4,1.0]  , 
                      [1.0,2.0]   ,
                      [1.0,4.0]  ,  
                     [1.0,8.0]  
                    



                    ]:
        
            for data in datas:  
                for ex in exs:    
                    task=ex['task'] 
                    model=ex['model']    
                    argvlist=[temp]+['--config','MODEL/'+task+'/'+'model.yaml']+['--config','DATA/'+data+'/'+data+'.yaml']+['--config','DATA/'+data+'/'+model+'.yaml']+['--data.eta',str(eta)]+['--model.model.debias',str(debias)]+['--trainer.check_val_every_n_epoch',str(50)] +['--trainer.max_epochs',str(50)]
                    # +['--data.init_args.param.eta','['+'0.1'+','+'0.2'+']']
                    # +['--data.init_args.param.eta','['+datap['rate']+','+datap['inten']+']']
                    # +['--optimizer.class_path','torch.optim.'+lrtype]+['--optimizer.init_args.lr',lr]  
                    pathname=['--trainer.default_root_dir', os.path.join('logs/LC/LCeta',task, data, str(eta)[1:-1] +'model'+model+'debias'+debias )]   

                        # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_SHRQ_Regular_encoerVQdiff\lightning_logs\version_40\checkpoints\epoch=69-step=2520.ckpt']
                        # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_0\checkpoints\epoch=69-step=14000.ckpt']
                        # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_5\checkpoints\epoch=19-step=4000.ckpt']
                        # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_22\checkpoints\epoch=39-step=8000.ckpt']
                        # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_35\checkpoints\epoch=99-step=20000.ckpt']
                        # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_57\checkpoints\epoch=109-step=22000.ckpt']
                        
                        
                        
                        
                        
                    ckpathname=[]
                    if os.path.exists(pathname[1]):
                        pass
                    else:
                        argvlists.append(argvlist+pathname+ckpathname)



    out=[]
    for i in     argvlists:
        out.append('\t'.join(i)       )
    printout='\n \n'.join(out) 
    print (printout)          
    return argvlists




def iqalr():       
    temp=sys.argv[0]  
    argvlists=[]  
    # datas=['koniqpsp','CSIQ','LIVECpsp','VCLpsp']   
    datas=[ 'CSIQ'  ]   
    # datas=[ 'LIVECpsp','koniqpsp','CSIQ','VCLpsp']   
    exs=[

        {'task':'LCMOS','model':'GDBC_MAMIQA','eta': [0.0,1.0] ,'debias': 'False' },  
        {'task':'LCMOS','model':'GDBC_MAMIQA','eta': [1.0,1.0] ,'debias': 'False' },  
        {'task':'LCMOS','model':'GDBC_MAMIQA','eta': [1.0,1.0] ,'debias': 'True' }, 
        {'task':'LCMOS','model':'GDBC_MAMIQA','eta': [0.0,1.0] ,'debias': 'True' },  
        # 

        # {'task':'LCMOS','model':'GDBC_CLIPIQA','eta': [0.0,1.0] ,'debias': 'False' },  
        # {'task':'LCMOS','model':'GDBC_DEIQT','eta': [0.0,1.0] ,'debias': 'False' },  
        # {'task':'LCMOS','model':'GDBC_DEIQT','eta': [1.0,1.0] ,'debias': 'False' }, 

        # {'task':'LCMOS','model':'GDBC_DEIQT','eta': [0.0,1.0] ,'debias': 'True' },  
        # {'task':'LCMOS','model':'GDBC_DEIQT','eta': [1.0,1.0] ,'debias': 'True' }, 

        # {'models':'OOD_TReS','p':'0.00','alpha':'0.0'},
        # {'models':'OOD_TReSinv','p':'0.1','alpha':'0.0000'},
        # {'models':'OOD_TReSinv','p':'0.1',' alpha':'0.1'},
        # {'models':'ZeroShot_TOPIQ','p':'0.2','alpha':'0.1'},
        # {'models':'VQresnet','p':'0.2','alpha':'0.1'},\
        # {'models':'VQdiff','p':'0.2','alpha':'0.1'}, 
        # {'models':'VQdiff','encoder': 'MANIQ384'  },
        # {'models':'VQdiff','encoder': 'MANIQ' }, 
        # {'models':'VQdiff','encoder': 'UNIQUE'  },
        # {'models':'VQdiff','encoder': 'TreS'  },
        # {'models':'VQdiff','encoder': 'TOPIQ'  },
        
        # 
        # {'models':'VQdiff','encoder': 'VQdiffnodisen' }, 
        
        # {'models':'VQdiff','encoder': 'refiqa' },  
        # {'models':'VQdiff','encoder': 'VQdiff' }, 
        # {'task':'LCMOS','model':'resnet','param':str({ 'eta': [0.0,1.0]  , 'iqatask': 'iqa'  }) }, 


        # {'task':'LCMOS','model':'GDBC_TreS','eta': [0.0,1.0] ,'debias': 'False' },  
        # {'task':'LCMOS','model':'GDBC_TreS','eta': [1.0,1.0] ,'debias': 'False' }, 
        # {'task':'LCMOS','model':'GDBC_TreS','eta': [0.0,1.0] ,'debias': 'True' },  
        # {'task':'LCMOS','model':'GDBC_TreS','eta': [1.0,1.0] ,'debias': 'True' }, 


        # {'task':'LCMOS','model':'GDBC_LWF','eta': [0.0,1.0] ,'debias': 'False'      },  
        # {'task':'LCMOS','model':'GDBC_MANIQ','eta': [1.0,1.0] ,'debias': 'False' }, 
        # {'task':'LCMOS','model':'GDBC_MANIQ','eta': [1.0,1.0] ,'debias': 'True' }, 
        # {'task':'LCMOS','model':'GDBC_MANIQ','eta': [0.0,1.0] ,'debias': 'False' },  
        # {'task':'LCMOS','model':'GDBC_MANIQ','eta': [0.0,1.0] ,'debias': 'True' },  


        # {'task':'LCMOS','model':'GDBC_resnet','eta': [1.0,1.0] ,'debias': 'False' }, 
        # {'task':'LCMOS','model':'GDBC_resnet','eta': [1.0,1.0] ,'debias': 'True' }, 
        # {'task':'LCMOS','model':'GDBC_resnet','eta': [0.0,1.0] ,'debias': 'False' },  
        # {'task':'LCMOS','model':'GDBC_resnet','eta': [0.0,1.0] ,'debias': 'True' }, 

        # {'task':'LCMOS','model':'GDBC_NIMA','eta': [0.0,1.0] ,'debias': 'False' },  
        # {'task':'LCMOS','model':'GDBC_NIMA','eta': [1.0,1.0] ,'debias': 'False' }, 

        # {'task':'LCMOS','model':'GDBC_HyperNet','eta': [1.0,1.0]  , 'debias': 'False' }, 
        # {'task':'LCMOS','model':'GDBC_HyperNet','eta': [0.0,1.0]  ,'debias': 'False'},   

        # {'task':'LCMOS','model':'GDBC_HyperNet','eta': [1.0,1.0]  , 'debias': 'True' }, 
        # {'task':'LCMOS','model':'GDBC_HyperNet','param':{ 'eta': [0.0,0.0]  , 'iqatask': 'iqa'  } , 'debias': 'True' }, 


        # {'task':'LCMOS','model':'GDBC_DBCNN','eta': [1.0,1.0]  , 'debias': 'False' }, 
        # {'task':'LCMOS','model':'GDBC_DBCNN','eta': [0.0,1.0]  ,'debias': 'False'},  
        # {'task':'LCMOS','model':'GDBC_LWF','eta': [0.0,1.0]  ,'debias': 'False'}, 
        # 

        # {'task':'LCMOS','model':'GDBC_LWF','eta': [0.0,1.0]  ,'debias': 'False'}, 
        # {'task':'LCMOS','model':'GDBC_LWF','eta': [1.0,1.0]  , 'debias': 'False' }, 
        # {'task':'LCMOS','model':'GDBC_LWF','eta': [1.0,1.0]  , 'debias': 'True' }, 
        
         
        # {'task':'LCMOS','model':'GDBC_LWF','eta': [0.0,1.0]  ,'debias': 'True'},  

        # {'task':'LCMOS','model':'GDBC_UNIQUE','eta': [0.0,1.0]  ,'debias': 'False'}, 
        # {'task':'LCMOS','model':'GDBC_UNIQUE','eta': [1.0,1.0]  , 'debias': 'False' }, 
        # {'task':'LCMOS','model':'GDBC_UNIQUE','eta': [1.0,1.0]  , 'debias': 'True' }, 
        
         
        # {'task':'LCMOS','model':'GDBC_UNIQUE','eta': [0.0,1.0]  ,'debias': 'True'},  

        

        # {'task':'LCMOS','model':'GDBC_TOPIQ','eta': [0.0,1.0] ,'debias': 'False' },  
        # {'task':'LCMOS','model':'GDBC_TOPIQ','eta': [1.0,1.0] ,'debias': 'False' }, 

        # {'task':'LCMOS','model':'GDBC_TOPIQ','eta': [0.0,1.0] ,'debias': 'False' },  
        # {'task':'LCMOS','model':'GDBC_TOPIQ','eta': [1.0,1.0] ,'debias': 'False' }, 
        # {'task':'LCMOS','model':'GDBC_RRNet','eta': [1.0,1.0] ,'debias': 'False' }, 






        # {'task':'LCMOS','model':'GDBC_resnet','param':str({ 'eta': [1.0,1.0]  , 'iqatask': 'iqa'  }) }, 
        ]          
    
    # +['--data.init_args.param.eta','['+rate+','+inten+']']
    # lrs={'SGD':['5e-1','1e-1','1e-2'],'AdamW':['3e-5','1e-4','1e-5','1e-3']}
    # lrs={'Adam':['2e-4','1e-4','5e-5','1e-3','1e-5','5e-4']} 
    # lrs={'Adam':[ '1e-5','5e-5' ]} 
    # tensorboard --logdir ./logs/VQdiff
    # lrs={'AdamW':['2e-4']}
    # lrs={'AdamW':['2e-4']}
    # lrs={'AdamW':['1e-5']}
    # lrs={'Adam':[  '1e-5' ,'5e-5']}
    lrs={'Adam':[  '1e-4'   ]}
    for indexseednum in range(  1,2):
        for seednum in [0]:
            for seed_everything in range(1,2):
                for lrtype in lrs.keys():   
                    for lr in lrs[lrtype]: 
                        for data in datas:  
                            for ex in exs:    
                                task=ex['task'] 
                                model=ex['model']  
                                eta=ex['eta']
                                debias=ex['debias'] 
                                argvlist=[temp]+['--config','MODEL/'+task+'/'+'model.yaml']+['--config','DATA/'+data+'/'+data+'.yaml']+['--config','DATA/'+data+'/'+model+'.yaml']+['--data.eta',str(eta)]+['--model.model.debias',str(debias)]+['--optimizer.class_path','torch.optim.'+lrtype]+['--optimizer.init_args.lr',lr]  \
                                +['--seed_everything',str(seed_everything)]    +['--data.seednum',str(seednum)]    +['--data.indexseednum',str(indexseednum)]  
                                
                                # +['--data.init_args.param.eta','['+'0.1'+','+'0.2'+']']
                                # +['--data.init_args.param.eta','['+datap['rate']+','+datap['inten']+']']
                                
                                pathname=['--trainer.default_root_dir', os.path.join('logs/LC/LClr',task, data, str(eta)[1:-1] +'model'+model+'debias'+debias+lrtype+lr +str(seed_everything)+'seednum'+str(seednum)+'indexseednum' +str(indexseednum))]    
                                
                                
                                
                                
                                
                                ckpathname=[]

                                # ckpathname=['--model.model.encoder.init_args.checkpointpath' , os.path.join("logs",'LC/LClr',task,"VCL*", str(eta)[1:-1]  +'model'+model+'debias'+debias+'*', "lightning_logs","version_0" ,"checkpoints",'*.ckpt')   ]
                                # ckpathname=['--model.model.encoder.checkpointpath' ,r'logs\CL\LCMOS\koniqpsp\1.0,*1.0modelGDBC_RRNetdebiasFalse\lightning_logs\version_0\checkpoints\epoch=49-step=22050.ckpt' ]
                                # if os.path.exists(pathname[1]):
                                #     pass
                                # else:
                                argvlists.append(argvlist+pathname+ckpathname)



    out=[]
    for i in     argvlists:
        out.append('\t'.join(i)       )
    printout='\n \n'.join(out) 
    print (printout)          
    return argvlists





def num():       
    temp=sys.argv[0]  
    argvlists=[] 
    # datas=['VCLpsp','LIVECpsp','koniqpsp','live']  
    # datas=['VCLpsp','TIDpsp','LIVECpsp','koniqpsp','live']  
    # datas=['VCLpsp','LIVECpsp','koniqpsp']
    # datas=['VCLpsp']
    # models=['at_DBCNN','at_NIMA','at_resnet'5,'at_Resnetlwtapre','at_resnetde','at_piece'] 
    # datas=['live'] 
    # datas=['TIDpsp','VCLpsp','LIVECpsp','kadid','koniqpsp']  
    # datas=['ESPL','kadid','SHRQ_Regular','SAUD','LIEQ','TIDpsp','live']   

    # datas=['LIEQ','ESPL','SAUD','SHRQ_Regular']    
    # datas=['SHRQ_Regular']    
    # datas=['koniqpsp','LIVECpsp','VCLpsp','CSIQ']   
    datas=['VCLpsp']  
    
    # datas=['SHRQ_Regular','ESPL','LIEQ','SAUD']  
    # models=['at_piece','at_DBCNN','at_NIMA','at_resnet','at_Resnetlwtapre','at_resnetde']  
    # models=['at_sin','at_DBCNN','at_NIMA','at_resnet','at_Resnetlwtapre','at_resnetde']  
    # models=['OOD_DBCNN']   
    # models=['OOD_resnet','OOD_TOPIQ']   
    # models=['OOD_ResNet','OOD_ResNetinv']   
    # models=['OOD_ResNetinv','OOD_ResNet'] 
    # models=['OOD_DBCNNinv','OOD_DBCNN' ] 
    # models=['OOD_TOPIQinv','OOD_TOPIQ' ] 
    # models=['ZeroShot_TOPIQ'  ] 6
    # models=['OOD_TReS','OOD_TReSinv' ] 
    # models=['OOD_Hypernet'] 
    # for lrtype in ['SGD','Adam']:
    #     for lr in ['1e-4','1e-3','1e-5','1e-2']:   
    #      
    # tensorboard --logdir ./logs/VQdiff
    exs=[
        # {'models':'OOD_TReS','p':'0.00','alpha':'0.0'},
        # {'models':'OOD_TReSinv','p':'0.1','alpha':'0.0000'},
        # {'models':'OOD_TReSinv','p':'0.1',' alpha':'0.1'},
        # {'models':'ZeroShot_TOPIQ','p':'0.2','alpha':'0.1'},
        # {'models':'VQresnet','p':'0.2','alpha':'0.1'},\
        # {'models':'VQdiff','p':'0.2','alpha':'0.1'}, 
        # {'models':'VQdiff','encoder': 'MANIQ384'  },
        # {'models':'VQdiff','encoder': 'MANIQ' }, 
        # {'models':'VQdiff','encoder': 'UNIQUE'  },
        # {'models':'VQdiff','encoder': 'TreS'  },
        # {'models':'VQdiff','encoder': 'TOPIQ'  },
        
        # 
        # {'models':'VQdiff','encoder': 'VQdiffnodisen' }, 
        
        # {'models':'VQdiff','encoder': 'refiqa' },  
        # {'models':'VQdiff','encoder': 'VQdiff' }, 
        # {'task':'LCMOS','model':'resnet','param':str({ 'eta': [0.0,1.0]  , 'iqatask': 'iqa'  }) },  


        # {'task':'LCMOS','model':'GDBC_resnet','eta': [1.0,1.0] ,'debias': 'True' }, 
 
        # {'task':'LCMOS','model':'GDBC_TreS','eta': [1.0,1.0] ,'debias': 'True' }, 
        # {'task':'LCMOS','model':'GDBC_TreS','eta': [0.0,1.0] ,'debias': 'True' },  
        # {'task':'LCMOS','model':'GDBC_TreS','eta': [1.0,1.0] ,'debias': 'True' }, 


        # {'task':'LCMOS','model':'GDBC_MANIQ','eta': [1.0,1.0] ,'debias': 'True' }, 



        {'task':'LCMOS','model':'GDBC_NIMA','eta': [1.0,1.0] ,'debias': 'True' }, 
        {'task':'LCMOS','model':'GDBC_resnet','eta': [1.0,1.0] ,'debias': 'True' }, 



        
        {'task':'LCMOS','model':'GDBC_HyperNet','eta': [1.0,1.0]  , 'debias': 'True' }, 

         {'task':'LCMOS','model':'GDBC_DBCNN','eta': [1.0,1.0]  , 'debias': 'True' }, 

        
        {'task':'LCMOS','model':'GDBC_UNIQUE','eta': [1.0,1.0]  , 'debias': 'True' }, 
        {'task':'LCMOS','model':'GDBC_LWF','eta': [1.0,1.0]  , 'debias': 'True' }, 

        {'task':'LCMOS','model':'GDBC_TOPIQ','eta': [1.0,1.0] ,'debias': 'True' }, 









        # {'task':'LCMOS','model':'GDBC_resnet','param':str({ 'eta': [1.0,1.0]  , 'iqatask': 'iqa'  }) }, 
        ]          
    
    # +['--data.init_args.param.eta','['+rate+','+inten+']']
    # lrs={'SGD':['5e-1','1e-1','1e-2'],'AdamW':['3e-5','1e-4','1e-5','1e-3']}
    # lrs={'AdamW':['1e-4','5e-4','5e-5']} 
    # tensorboard --logdir ./logs/VQdiff
    # lrs={'AdamW':['2e-4']}
    # lrs={'AdamW':['2e-4']}
    # lrs={'AdamW':['1e-5']}
    # lrs={'AdamW':['1e-5']}
    # for lrtype in lrs.keys():
    #     for lr in lrs[lrtype]:

    
    for data in datas:  
        for ex in exs:    
            for num in [1,3,5,7,9,11,25,45]:
                task=ex['task'] 
                model=ex['model']    
                eta=ex['eta']
                debias=ex['debias']                    
                argvlist=[temp]+['--config','MODEL/'+task+'/'+'model.yaml']+['--config','DATA/'+data+'/'+data+'.yaml']+['--config','DATA/'+data+'/'+model+'.yaml']+['--data.eta',str(eta)]+['--model.model.debias',str(debias)]\
                +['--model.model.snum',str(num)] 
                # +['--trainer.check_val_every_n_epoch',str(50)] +['--trainer.max_epochs',str(50)]
                # +['--data.init_args.param.eta','['+'0.1'+','+'0.2'+']']
                # +['--data.init_args.param.eta','['+datap['rate']+','+datap['inten']+']']
                # +['--optimizer.class_path','torch.optim.'+lrtype]+['--optimizer.init_args.lr',lr]  
                pathname=['--trainer.default_root_dir', os.path.join('logs/LC/num',task, data, str(eta)[1:-1] +'model'+model+'debias'+debias +'num'+str(num))]   

                    # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_SHRQ_Regular_encoerVQdiff\lightning_logs\version_40\checkpoints\epoch=69-step=2520.ckpt']
                    # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_0\checkpoints\epoch=69-step=14000.ckpt']
                    # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_5\checkpoints\epoch=19-step=4000.ckpt']
                    # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_22\checkpoints\epoch=39-step=8000.ckpt']
                    # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_35\checkpoints\epoch=99-step=20000.ckpt']
                    # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_57\checkpoints\epoch=109-step=22000.ckpt']
                    
                    
                    
                    
                    
                ckpathname=[]
                if os.path.exists(pathname[1]):
                    pass
                else:
                    argvlists.append(argvlist+pathname+ckpathname)



    out=[]
    for i in     argvlists:
        out.append('\t'.join(i)       )
    printout='\n \n'.join(out) 
    print (printout)          
    return argvlists



def time():       
    temp=sys.argv[0]  
    argvlists=[] 
    # datas=['VCLpsp','LIVECpsp','koniqpsp','live']  
    # datas=['VCLpsp','TIDpsp','LIVECpsp','koniqpsp','live']  
    # datas=['VCLpsp','LIVECpsp','koniqpsp']
    # datas=['VCLpsp']
    # models=['at_DBCNN','at_NIMA','at_resnet'5,'at_Resnetlwtapre','at_resnetde','at_piece'] 
    # datas=['live'] 
    # datas=['TIDpsp','VCLpsp','LIVECpsp','kadid','koniqpsp']  
    # datas=['ESPL','kadid','SHRQ_Regular','SAUD','LIEQ','TIDpsp','live']   

    # datas=['LIEQ','ESPL','SAUD','SHRQ_Regular']    
    # datas=['SHRQ_Regular']    
    datas=['koniqpsp']   
    # datas=['koniqpsp']  
    
    # datas=['SHRQ_Regular','ESPL','LIEQ','SAUD']  
    # models=['at_piece','at_DBCNN','at_NIMA','at_resnet','at_Resnetlwtapre','at_resnetde']  
    # models=['at_sin','at_DBCNN','at_NIMA','at_resnet','at_Resnetlwtapre','at_resnetde']  
    # models=['OOD_DBCNN']   
    # models=['OOD_resnet','OOD_TOPIQ']   
    # models=['OOD_ResNet','OOD_ResNetinv']   
    # models=['OOD_ResNetinv','OOD_ResNet'] 
    # models=['OOD_DBCNNinv','OOD_DBCNN' ] 
    # models=['OOD_TOPIQinv','OOD_TOPIQ' ] 
    # models=['ZeroShot_TOPIQ'  ] 6
    # models=['OOD_TReS','OOD_TReSinv' ] 
    # models=['OOD_Hypernet'] 
    # for lrtype in ['SGD','Adam']:
    #     for lr in ['1e-4','1e-3','1e-5','1e-2']:   
    #      
    # tensorboard --logdir ./logs/VQdiff
    exs=[
        # {'models':'OOD_TReS','p':'0.00','alpha':'0.0'},
        # {'models':'OOD_TReSinv','p':'0.1','alpha':'0.0000'},
        # {'models':'OOD_TReSinv','p':'0.1',' alpha':'0.1'},
        # {'models':'ZeroShot_TOPIQ','p':'0.2','alpha':'0.1'},
        # {'models':'VQresnet','p':'0.2','alpha':'0.1'},\
        # {'models':'VQdiff','p':'0.2','alpha':'0.1'}, 
        # {'models':'VQdiff','encoder': 'MANIQ384'  },
        # {'models':'VQdiff','encoder': 'MANIQ' }, 
        # {'models':'VQdiff','encoder': 'UNIQUE'  },
        # {'models':'VQdiff','encoder': 'TreS'  },
        # {'models':'VQdiff','encoder': 'TOPIQ'  },
        
        # 
        # {'models':'VQdiff','encoder': 'VQdiffnodisen' }, 
        
        # {'models':'VQdiff','encoder': 'refiqa' },  
        # {'models':'VQdiff','encoder': 'VQdiff' }, 
        # {'task':'LCMOS','model':'resnet','param':str({ 'eta': [0.0,1.0]  , 'iqatask': 'iqa'  }) }, 

        
 
        {'task':'LCMOS','model':'GDBC_resnet','eta': [1.0,1.0] ,'debias': 'False' }, 







 
        {'task':'LCMOS','model':'GDBC_NIMA','eta': [1.0,1.0] ,'debias': 'False' }, 


  
        {'task':'LCMOS','model':'GDBC_HyperNet','eta': [1.0,1.0]  , 'debias': 'False' }, 

        
         {'task':'LCMOS','model':'GDBC_DBCNN','eta': [1.0,1.0]  , 'debias': 'False' }, 

        
         {'task':'LCMOS','model':'GDBC_UNIQUE','eta': [1.0,1.0]  , 'debias': 'False' }, 
 
        {'task':'LCMOS','model':'GDBC_TOPIQ','eta': [1.0,1.0] ,'debias': 'False' }, 

 
        {'task':'LCMOS','model':'GDBC_LWF','eta': [1.0,1.0] ,'debias': 'False' },  

        {'task':'LCMOS','model':'GDBC_MANIQ','eta': [1.0,1.0] ,'debias': 'False' },  






        # {'task':'LCMOS','model':'GDBC_resnet','param':str({ 'eta': [1.0,1.0]  , 'iqatask': 'iqa'  }) }, 
        ]          
    
    # +['--data.init_args.param.eta','['+rate+','+inten+']']
    # lrs={'SGD':['5e-1','1e-1','1e-2'],'AdamW':['3e-5','1e-4','1e-5','1e-3']}
    # lrs={'AdamW':['1e-4','5e-4','5e-5']} 
    # tensorboard --logdir ./logs/VQdiff
    # lrs={'AdamW':['2e-4']}
    # lrs={'AdamW':['2e-4']}
    # lrs={'AdamW':['1e-5']}
    # lrs={'AdamW':['1e-5']}
    # for lrtype in lrs.keys():
    #     for lr in lrs[lrtype]:

    # for debias in ['True','False']:
    for data in datas:  
        for ex in exs:    
            for debias in ['False','True']:
    # for debias in ['False' ]: 
                task=ex['task'] 
                model=ex['model']   
                eta=ex['eta']
                # debias=ex['debias']
                argvlist=[temp]+['--config','MODEL/'+task+'/'+'model.yaml']+['--config','DATA/'+data+'/'+data+'.yaml']+['--config','DATA/'+data+'/'+model+'.yaml']+['--data.eta',str(eta)]+['--model.model.debias',str(debias)]+['--trainer.check_val_every_n_epoch',str(1)] +['--trainer.max_epochs',str(10)]+['--trainer.limit_train_batches',str(50)]+['--trainer.limit_val_batches',str(50)]
                # +['--data.init_args.param.eta','['+'0.1'+','+'0.2'+']']
                # +['--data.init_args.param.eta','['+datap['rate']+','+datap['inten']+']']
                # +['--optimizer.class_path','torch.optim.'+lrtype]+['--optimizer.init_args.lr',lr]  
                pathname=['--trainer.default_root_dir', os.path.join('logs/LC/time',task, data, str(eta)[1:-1] +'model'+model+'debias'+debias )]   

                    # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_SHRQ_Regular_encoerVQdiff\lightning_logs\version_40\checkpoints\epoch=69-step=2520.ckpt']
                    # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_0\checkpoints\epoch=69-step=14000.ckpt']
                    # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_5\checkpoints\epoch=19-step=4000.ckpt']
                    # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_22\checkpoints\epoch=39-step=8000.ckpt']
                    # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_35\checkpoints\epoch=99-step=20000.ckpt']
                    # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_57\checkpoints\epoch=109-step=22000.ckpt']
                    
                    
                    
                    
                    
                ckpathname=[]
                if os.path.exists(pathname[1]):
                    pass
                else:
                    argvlists.append(argvlist+pathname+ckpathname)



    out=[]
    for i in     argvlists:
        out.append('\t'.join(i)       )
    printout='\n \n'.join(out) 
    print (printout)          
    return argvlists



def calibratemse():       
    temp=sys.argv[0]  
    argvlists=[] 
    # datas=['VCLpsp','LIVECpsp','koniqpsp','live']  
    # datas=['VCLpsp','TIDpsp','LIVECpsp','koniqpsp','live']  
    # datas=['VCLpsp','LIVECpsp','koniqpsp']
    # datas=['VCLpsp']
    # models=['at_DBCNN','at_NIMA','at_resnet'5,'at_Resnetlwtapre','at_resnetde','at_piece'] 
    # datas=['live'] 
    # datas=['TIDpsp','VCLpsp','LIVECpsp','kadid','koniqpsp']  
    # datas=['ESPL','kadid','SHRQ_Regular','SAUD','LIEQ','TIDpsp','live']   

    # datas=['LIEQ','ESPL','SAUD','SHRQ_Regular']    
    # datas=['SHRQ_Regular']    
    datas=['CSIQ','VCLpsp','LIVECpsp','koniqpsp']   
    # datas=['koniqpsp']  
    
    # datas=['SHRQ_Regular','ESPL','LIEQ','SAUD']  
    # models=['at_piece','at_DBCNN','at_NIMA','at_resnet','at_Resnetlwtapre','at_resnetde']  
    # models=['at_sin','at_DBCNN','at_NIMA','at_resnet','at_Resnetlwtapre','at_resnetde']  
    # models=['OOD_DBCNN']   
    # models=['OOD_resnet','OOD_TOPIQ']   
    # models=['OOD_ResNet','OOD_ResNetinv']   
    # models=['OOD_ResNetinv','OOD_ResNet'] 
    # models=['OOD_DBCNNinv','OOD_DBCNN' ] 
    # models=['OOD_TOPIQinv','OOD_TOPIQ' ] 
    # models=['ZeroShot_TOPIQ'  ] 6
    # models=['OOD_TReS','OOD_TReSinv' ] 
    # models=['OOD_Hypernet'] 
    # for lrtype in ['SGD','Adam']:
    #     for lr in ['1e-4','1e-3','1e-5','1e-2']:   
    #      
    # tensorboard --logdir ./logs/VQdiff
    exs=[
        # {'models':'OOD_TReS','p':'0.00','alpha':'0.0'},
        # {'models':'OOD_TReSinv','p':'0.1','alpha':'0.0000'},
        # {'models':'OOD_TReSinv','p':'0.1',' alpha':'0.1'},
        # {'models':'ZeroShot_TOPIQ','p':'0.2','alpha':'0.1'},
        # {'models':'VQresnet','p':'0.2','alpha':'0.1'},\
        # {'models':'VQdiff','p':'0.2','alpha':'0.1'}, 
        # {'models':'VQdiff','encoder': 'MANIQ384'  },
        # {'models':'VQdiff','encoder': 'MANIQ' }, 
        # {'models':'VQdiff','encoder': 'UNIQUE'  },
        # {'models':'VQdiff','encoder': 'TreS'  },
        # {'models':'VQdiff','encoder': 'TOPIQ'  },
        
        # 
        # {'models':'VQdiff','encoder': 'VQdiffnodisen' }, 
        
        # {'models':'VQdiff','encoder': 'refiqa' },  
        # {'models':'VQdiff','encoder': 'VQdiff' }, 
        # {'task':'LCMOS','model':'resnet','param':str({ 'eta': [0.0,1.0]  , 'iqatask': 'iqa'  }) }, 

        # {'task':'LCMOS','model':'GDBC_LWF','eta': [0.0,1.0] ,'debias': 'False' },  
        # {'task':'LCMOS','model':'GDBC_LWF','eta': [1.0,1.0] ,'debias': 'False' }, 
        # {'task':'LCMOS','model':'GDBC_LWF','eta': [1.0,1.0] ,'debias': 'True' }, 

        # {'task':'LCMOS','model':'GDBC_LWF','eta': [0.0,1.0] ,'debias': 'True' },  

        # {'task':'LCMOS','model':'GDBC_UNIQUE','eta': [0.0,1.0]  ,'debias': 'False'},  
        #  {'task':'LCMOS','model':'GDBC_UNIQUE','eta': [1.0,1.0]  , 'debias': 'False' }, 
        #  {'task':'LCMOS','model':'GDBC_UNIQUE','eta': [1.0,1.0]  , 'debias': 'True' }, 

        # {'task':'LCMOS','model':'GDBC_UNIQUE','eta': [0.0,1.0]  ,'debias': 'True'},  

        # {'task':'LCMOS','model':'GDBC_UNIQUE','eta': [0.0,1.0]  ,'debias': 'False'},  
        #  {'task':'LCMOS','model':'GDBC_UNIQUE','eta': [1.0,1.0]  , 'debias': 'False' }, 
        #  {'task':'LCMOS','model':'GDBC_TOPIQ','eta': [1.0,1.0]  , 'debias': 'True' }, 

        # {'task':'LCMOS','model':'GDBC_UNIQUE','eta': [0.0,1.0]  ,'debias': 'True'},  
        

         {'task':'LCMOS','model':'GDBC_MANIQ','eta': [1.0,1.0]  , 'debias': 'True' }, 

        # {'task':'LCMOS','model':'GDBC_resnet','eta': [0.0,1.0] ,'debias': 'False' },  
        # {'task':'LCMOS','model':'GDBC_resnet','eta': [1.0,1.0] ,'debias': 'False' }, 








        # {'task':'LCMOS','model':'GDBC_NIMA','eta': [0.0,1.0] ,'debias': 'False' },  
        # {'task':'LCMOS','model':'GDBC_NIMA','eta': [1.0,1.0] ,'debias': 'False' }, 



        
        # {'task':'LCMOS','model':'GDBC_HyperNet','eta': [0.0,1.0]  ,'debias': 'False'},  
        # {'task':'LCMOS','model':'GDBC_HyperNet','eta': [1.0,1.0]  , 'debias': 'False' }, 

       
        # {'task':'LCMOS','model':'GDBC_DBCNN','eta': [0.0,1.0]  ,'debias': 'False'},  
        #  {'task':'LCMOS','model':'GDBC_DBCNN','eta': [1.0,1.0]  , 'debias': 'False' }, 

       


        # {'task':'LCMOS','model':'GDBC_TOPIQ','eta': [0.0,1.0] ,'debias': 'False' },  
        # {'task':'LCMOS','model':'GDBC_TOPIQ','eta': [1.0,1.0] ,'debias': 'False' }, 
        # {'task':'LCMOS','model':'GDBC_TOPIQ','eta': [0.0,1.0] ,'debias': 'True' },  
        # {'task':'LCMOS','model':'GDBC_TOPIQ','eta': [1.0,1.0] ,'debias': 'True' }, 

        # {'task':'LCMOS','model':'GDBC_TreS','eta': [0.0,1.0] ,'debias': 'False' },  
        # {'task':'LCMOS','model':'GDBC_TreS','eta': [1.0,1.0] ,'debias': 'False' },  

        # {'task':'LCMOS','model':'GDBC_MANIQ','eta': [0.0,1.0] ,'debias': 'False' },  
        # {'task':'LCMOS','model':'GDBC_MANIQ','eta': [1.0,1.0] ,'debias': 'False' }, 
        

        # {'task':'LCMOS','model':'GDBC_MANIQ','eta': [1.0,1.0] ,'debias': 'True' }, 
        # {'task':'LCMOS','model':'GDBC_MANIQ','eta': [0.0,1.0] ,'debias': 'True' },  




        # {'task':'LCMOS','model':'GDBC_resnet','param':str({ 'eta': [1.0,1.0]  , 'iqatask': 'iqa'  }) }, 
        ]          
    
    # +['--data.init_args.param.eta','['+rate+','+inten+']']
    # lrs={'SGD':['5e-1','1e-1','1e-2'],'AdamW':['3e-5','1e-4','1e-5','1e-3']}
    # lrs={'AdamW':['1e-4','5e-4','5e-5']} 
    # tensorboard --logdir ./logs/VQdiff
    # lrs={'AdamW':['2e-4']}
    # lrs={'AdamW':['2e-4']}
    # lrs={'AdamW':['1e-5']}
    # lrs={'AdamW':['1e-5']}
    # for lrtype in lrs.keys():
    #     for lr in lrs[lrtype]:

    # for debias in ['True','False']:
    for data in datas:  
        for ex in exs:    
    # for debias in ['False' ]: 
            task=ex['task'] 
            model=ex['model']   
            debias=ex['debias']
            eta=ex['eta']
            argvlist=[temp]+['--config','MODEL/'+task+'/'+'model.yaml']+['--config','DATA/'+data+'/'+data+'.yaml']+['--config','DATA/'+data+'/'+model+'.yaml']+['--model.model.debias',str(debias)]\
            +['--data.eta',str( eta)] +['--trainer.check_val_every_n_epoch',str(50)]
            # +['--data.param',str(param)]
            
            # +['--data.init_args.param.eta','['+'0.1'+','+'0.2'+']']
            # +['--data.init_args.param.eta','['+datap['rate']+','+datap['inten']+']']
            # +['--optimizer.class_path','torch.optim.'+lrtype]+['--optimizer.init_args.lr',lr]  
            pathname=['--trainer.default_root_dir', os.path.join('logs/LC/','calibratemse', data, str(eta)[1:-1] +'model'+model+'debias'+debias )]   

                # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_SHRQ_Regular_encoerVQdiff\lightning_logs\version_40\checkpoints\epoch=69-step=2520.ckpt']
                # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_0\checkpoints\epoch=69-step=14000.ckpt']
                # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_5\checkpoints\epoch=19-step=4000.ckpt']
                # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_22\checkpoints\epoch=39-step=8000.ckpt']
                # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_35\checkpoints\epoch=99-step=20000.ckpt']
                # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_57\checkpoints\epoch=109-step=22000.ckpt']
                
                
                
                
                
            ckpathname=[]
            if os.path.exists(pathname[1]):
                pass
            else:
                argvlists.append(argvlist+pathname+ckpathname)



    out=[]
    for i in     argvlists:
        out.append('\t'.join(i)       )
    printout='\n \n'.join(out) 
    print (printout)          
    return argvlists

def ema():       
    temp=sys.argv[0]  
    argvlists=[] 
    # datas=['VCLpsp','LIVECpsp','koniqpsp','live']  
    # datas=['VCLpsp','TIDpsp','LIVECpsp','koniqpsp','live']  
    # datas=['VCLpsp','LIVECpsp','koniqpsp']
    # datas=['VCLpsp']
    # models=['at_DBCNN','at_NIMA','at_resnet'5,'at_Resnetlwtapre','at_resnetde','at_piece'] 
    # datas=['live'] 
    # datas=['TIDpsp','VCLpsp','LIVECpsp','kadid','koniqpsp']  
    # datas=['ESPL','kadid','SHRQ_Regular','SAUD','LIEQ','TIDpsp','live']   

    # datas=['LIEQ','ESPL','SAUD','SHRQ_Regular']    
    # datas=['SHRQ_Regular']    
    # datas=['koniqpsp','LIVECpsp','VCLpsp','CSIQ']   
    
    datas=['koniqpsp']  
    # datas=['SHRQ_Regular','ESPL','LIEQ','SAUD']  
    # models=['at_piece','at_DBCNN','at_NIMA','at_resnet','at_Resnetlwtapre','at_resnetde']  
    # models=['at_sin','at_DBCNN','at_NIMA','at_resnet','at_Resnetlwtapre','at_resnetde']  
    # models=['OOD_DBCNN']   
    # models=['OOD_resnet','OOD_TOPIQ']   
    # models=['OOD_ResNet','OOD_ResNetinv']   
    # models=['OOD_ResNetinv','OOD_ResNet'] 
    # models=['OOD_DBCNNinv','OOD_DBCNN' ] 
    # models=['OOD_TOPIQinv','OOD_TOPIQ' ] 
    # models=['ZeroShot_TOPIQ'  ] 6
    # models=['OOD_TReS','OOD_TReSinv' ] 
    # models=['OOD_Hypernet'] 
    # for lrtype in ['SGD','Adam']:
    #     for lr in ['1e-4','1e-3','1e-5','1e-2']:   
    #      
    # tensorboard --logdir ./logs/VQdiff
    exs=[
        # {'models':'OOD_TReS','p':'0.00','alpha':'0.0'},
        # {'models':'OOD_TReSinv','p':'0.1','alpha':'0.0000'},
        # {'models':'OOD_TReSinv','p':'0.1',' alpha':'0.1'},
        # {'models':'ZeroShot_TOPIQ','p':'0.2','alpha':'0.1'},
        # {'models':'VQresnet','p':'0.2','alpha':'0.1'},\
        # {'models':'VQdiff','p':'0.2','alpha':'0.1'}, 
        # {'models':'VQdiff','encoder': 'MANIQ384'  },
        # {'models':'VQdiff','encoder': 'MANIQ' }, 
        # {'models':'VQdiff','encoder': 'UNIQUE'  },
        # {'models':'VQdiff','encoder': 'TreS'  },
        # {'models':'VQdiff','encoder': 'TOPIQ'  },
        
        # 
        # {'models':'VQdiff','encoder': 'VQdiffnodisen' }, 
        
        # {'models':'VQdiff','encoder': 'refiqa' },  
        # {'models':'VQdiff','encoder': 'VQdiff' }, 
        # {'task':'LCMOS','model':'resnet','param':str({ 'eta': [0.0,1.0]  , 'iqatask': 'iqa'  }) },  


        # {'task':'LCMOS','model':'GDBC_resnet','eta': [1.0,1.0] ,'debias': 'True' }, 
 
        # {'task':'LCMOS','model':'GDBC_TreS','eta': [1.0,1.0] ,'debias': 'True' }, 
        # {'task':'LCMOS','model':'GDBC_TreS','eta': [0.0,1.0] ,'debias': 'True' },  
        # {'task':'LCMOS','model':'GDBC_TreS','eta': [1.0,1.0] ,'debias': 'True' }, 


        # 



        # {'task':'LCMOS','model':'GDBC_NIMA','eta': [1.0,1.0] ,'debias': 'True' }, 
        # {'task':'LCMOS','model':'GDBC_resnet','eta': [1.0,1.0] ,'debias': 'True' }, 

        # {'task':'LCMOS','model':'GDBC_HyperNet','eta': [1.0,1.0]  , 'debias': 'True' }, 

        # {'task':'LCMOS','model':'GDBC_DBCNN','eta': [1.0,1.0]  , 'debias': 'True' }, 

        # {'task':'LCMOS','model':'GDBC_TOPIQ','eta': [1.0,1.0] ,'debias': 'True' }, 
        # {'task':'LCMOS','model':'GDBC_MANIQ','eta': [1.0,1.0] ,'debias': 'True' }, 
        # {'task':'LCMOS','model':'GDBC_UNIQUE','eta': [1.0,1.0]  , 'debias': 'True' }, 
        # {'task':'LCMOS','model':'GDBC_LWF','eta': [1.0,1.0]  , 'debias': 'True' }, 

        {'task':'LCMOS','model':'GDBC_MAMIQA','eta': [1.0,1.0]  , 'debias': 'True' }, 
        {'task':'LCMOS','model':'GDBC_DEIQT','eta': [1.0,1.0]  , 'debias': 'True' }, 





        # {'task':'LCMOS','model':'GDBC_resnet','param':str({ 'eta': [1.0,1.0]  , 'iqatask': 'iqa'  }) }, 
        ]          
    
    # +['--data.init_args.param.eta','['+rate+','+inten+']']
    # lrs={'SGD':['5e-1','1e-1','1e-2'],'AdamW':['3e-5','1e-4','1e-5','1e-3']}
    # lrs={'AdamW':['1e-4','5e-4','5e-5']} 
    # tensorboard --logdir ./logs/VQdiff
    # lrs={'AdamW':['2e-4']}
    # lrs={'AdamW':['2e-4']}
    # lrs={'AdamW':['1e-5']}
    # lrs={'AdamW':['1e-5']}
    # for lrtype in lrs.keys():
    #     for lr in lrs[lrtype]:

    
    for data in datas:  
        for ex in exs:    
            for ema in [0.0,0.1,0.3,0.5,0.7,0.9,1.0]:
                task=ex['task'] 
                model=ex['model']    
                eta=ex['eta']
                debias=ex['debias']                    
                argvlist=[temp]+['--config','MODEL/'+task+'/'+'model.yaml']+['--config','DATA/'+data+'/'+data+'.yaml']+['--config','DATA/'+data+'/'+model+'.yaml']+['--data.eta',str(eta)]+['--model.model.debias',str(debias)]\
                +['--model.model.ema',str(ema)] +['--trainer.check_val_every_n_epoch',str(50)]
                # +['--trainer.check_val_every_n_epoch',str(50)] +['--trainer.max_epochs',str(50)]
                # +['--data.init_args.param.eta','['+'0.1'+','+'0.2'+']']
                # +['--data.init_args.param.eta','['+datap['rate']+','+datap['inten']+']']
                # +['--optimizer.class_path','torch.optim.'+lrtype]+['--optimizer.init_args.lr',lr]  
                pathname=['--trainer.default_root_dir', os.path.join('logs/LC/ema',task, data, str(eta)[1:-1] +'model'+model+'debias'+debias +'ema'+str(ema))]   

                    # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_SHRQ_Regular_encoerVQdiff\lightning_logs\version_40\checkpoints\epoch=69-step=2520.ckpt']
                    # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_0\checkpoints\epoch=69-step=14000.ckpt']
                    # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_5\checkpoints\epoch=19-step=4000.ckpt']
                    # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_22\checkpoints\epoch=39-step=8000.ckpt']
                    # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_35\checkpoints\epoch=99-step=20000.ckpt']
                    # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_57\checkpoints\epoch=109-step=22000.ckpt']
                    
                    
                    
                    
                    
                ckpathname=[]
                if os.path.exists(pathname[1]):
                    pass
                else:
                    argvlists.append(argvlist+pathname+ckpathname)



    out=[]
    for i in     argvlists:
        out.append('\t'.join(i)       )
    printout='\n \n'.join(out) 
    print (printout)          
    return argvlists


def sth():       
    temp=sys.argv[0]  
    argvlists=[] 
    # datas=['VCLpsp','LIVECpsp','koniqpsp','live']  
    # datas=['VCLpsp','TIDpsp','LIVECpsp','koniqpsp','live']  
    # datas=['VCLpsp','LIVECpsp','koniqpsp']
    # datas=['VCLpsp']
    # models=['at_DBCNN','at_NIMA','at_resnet'5,'at_Resnetlwtapre','at_resnetde','at_piece'] 
    # datas=['live'] 
    # datas=['TIDpsp','VCLpsp','LIVECpsp','kadid','koniqpsp']  
    # datas=['ESPL','kadid','SHRQ_Regular','SAUD','LIEQ','TIDpsp','live']   

    # datas=['LIEQ','ESPL','SAUD','SHRQ_Regular']    
    # datas=['SHRQ_Regular']    
    # datas=['koniqpsp','LIVECpsp','VCLpsp','CSIQ']   
    datas=['VCLpsp']  
    
    # datas=['SHRQ_Regular','ESPL','LIEQ','SAUD']  
    # models=['at_piece','at_DBCNN','at_NIMA','at_resnet','at_Resnetlwtapre','at_resnetde']  
    # models=['at_sin','at_DBCNN','at_NIMA','at_resnet','at_Resnetlwtapre','at_resnetde']  
    # models=['OOD_DBCNN']   
    # models=['OOD_resnet','OOD_TOPIQ']   
    # models=['OOD_ResNet','OOD_ResNetinv']   
    # models=['OOD_ResNetinv','OOD_ResNet'] 
    # models=['OOD_DBCNNinv','OOD_DBCNN' ] 
    # models=['OOD_TOPIQinv','OOD_TOPIQ' ] 
    # models=['ZeroShot_TOPIQ'  ] 6
    # models=['OOD_TReS','OOD_TReSinv' ] 
    # models=['OOD_Hypernet'] 
    # for lrtype in ['SGD','Adam']:
    #     for lr in ['1e-4','1e-3','1e-5','1e-2']:   
    #      
    # tensorboard --logdir ./logs/VQdiff
    exs=[
        # {'models':'OOD_TReS','p':'0.00','alpha':'0.0'},
        # {'models':'OOD_TReSinv','p':'0.1','alpha':'0.0000'},
        # {'models':'OOD_TReSinv','p':'0.1',' alpha':'0.1'},
        # {'models':'ZeroShot_TOPIQ','p':'0.2','alpha':'0.1'},
        # {'models':'VQresnet','p':'0.2','alpha':'0.1'},\
        # {'models':'VQdiff','p':'0.2','alpha':'0.1'}, 
        # {'models':'VQdiff','encoder': 'MANIQ384'  },
        # {'models':'VQdiff','encoder': 'MANIQ' }, 
        # {'models':'VQdiff','encoder': 'UNIQUE'  },
        # {'models':'VQdiff','encoder': 'TreS'  },
        # {'models':'VQdiff','encoder': 'TOPIQ'  },
        
        # 
        # {'models':'VQdiff','encoder': 'VQdiffnodisen' }, 
        
        # {'models':'VQdiff','encoder': 'refiqa' },  
        # {'models':'VQdiff','encoder': 'VQdiff' }, 
        # {'task':'LCMOS','model':'resnet','param':str({ 'eta': [0.0,1.0]  , 'iqatask': 'iqa'  }) },  


        # {'task':'LCMOS','model':'GDBC_resnet','eta': [1.0,1.0] ,'debias': 'True' }, 
 
        # {'task':'LCMOS','model':'GDBC_TreS','eta': [1.0,1.0] ,'debias': 'True' }, 
        # {'task':'LCMOS','model':'GDBC_TreS','eta': [0.0,1.0] ,'debias': 'True' },  
        # {'task':'LCMOS','model':'GDBC_TreS','eta': [1.0,1.0] ,'debias': 'True' }, 


        # 



        {'task':'LCMOS','model':'GDBC_NIMA','eta': [1.0,1.0] ,'debias': 'True' }, 
        {'task':'LCMOS','model':'GDBC_resnet','eta': [1.0,1.0] ,'debias': 'True' }, 

        {'task':'LCMOS','model':'GDBC_HyperNet','eta': [1.0,1.0]  , 'debias': 'True' }, 

        {'task':'LCMOS','model':'GDBC_DBCNN','eta': [1.0,1.0]  , 'debias': 'True' }, 

        {'task':'LCMOS','model':'GDBC_TOPIQ','eta': [1.0,1.0] ,'debias': 'True' }, 
        {'task':'LCMOS','model':'GDBC_MANIQ','eta': [1.0,1.0] ,'debias': 'True' }, 
        {'task':'LCMOS','model':'GDBC_UNIQUE','eta': [1.0,1.0]  , 'debias': 'True' }, 
        {'task':'LCMOS','model':'GDBC_LWF','eta': [1.0,1.0]  , 'debias': 'True' }, 







        # {'task':'LCMOS','model':'GDBC_resnet','param':str({ 'eta': [1.0,1.0]  , 'iqatask': 'iqa'  }) }, 
        ]          
    
    # +['--data.init_args.param.eta','['+rate+','+inten+']']
    # lrs={'SGD':['5e-1','1e-1','1e-2'],'AdamW':['3e-5','1e-4','1e-5','1e-3']}
    # lrs={'AdamW':['1e-4','5e-4','5e-5']} 
    # tensorboard --logdir ./logs/VQdiff
    # lrs={'AdamW':['2e-4']}
    # lrs={'AdamW':['2e-4']}
    # lrs={'AdamW':['1e-5']}
    # lrs={'AdamW':['1e-5']}
    # for lrtype in lrs.keys():
    #     for lr in lrs[lrtype]:

    
    for data in datas:  
        for ex in exs:    
            for sth in [0.001,0.01,0.1,1.0]:
                task=ex['task'] 
                model=ex['model']    
                eta=ex['eta']
                debias=ex['debias']                    
                argvlist=[temp]+['--config','MODEL/'+task+'/'+'model.yaml']+['--config','DATA/'+data+'/'+data+'.yaml']+['--config','DATA/'+data+'/'+model+'.yaml']+['--data.eta',str(eta)]+['--model.model.debias',str(debias)]\
                +['--model.model.sth',str(sth)] +['--trainer.check_val_every_n_epoch',str(50)]
                # +['--trainer.check_val_every_n_epoch',str(50)] +['--trainer.max_epochs',str(50)]
                # +['--data.init_args.param.eta','['+'0.1'+','+'0.2'+']']
                # +['--data.init_args.param.eta','['+datap['rate']+','+datap['inten']+']']
                # +['--optimizer.class_path','torch.optim.'+lrtype]+['--optimizer.init_args.lr',lr]  
                pathname=['--trainer.default_root_dir', os.path.join('logs/LC/sth',task, data, str(eta)[1:-1] +'model'+model+'debias'+debias +'sth'+str(sth))]   

                    # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_SHRQ_Regular_encoerVQdiff\lightning_logs\version_40\checkpoints\epoch=69-step=2520.ckpt']
                    # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_0\checkpoints\epoch=69-step=14000.ckpt']
                    # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_5\checkpoints\epoch=19-step=4000.ckpt']
                    # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_22\checkpoints\epoch=39-step=8000.ckpt']
                    # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_35\checkpoints\epoch=99-step=20000.ckpt']
                    # ckpathname=['--model.init_args.checkpointpath',r'logs\VQdiff\VQdiff_kadid_encoerVQdiff\lightning_logs\version_57\checkpoints\epoch=109-step=22000.ckpt']
                    
                    
                    
                    
                    
                ckpathname=[]
                if os.path.exists(pathname[1]):
                    pass
                else:
                    argvlists.append(argvlist+pathname+ckpathname)



    out=[]
    for i in     argvlists:
        out.append('\t'.join(i)       )
    printout='\n \n'.join(out) 
    print (printout)          
    return argvlists

if __name__ == "__main__": 
    # argvlists=calibratemse()      
    # argvlists=eta()    
    # argvlists=iqalr()  
    # argvlists=num()    
    # argvlists=ema()    
    # argvlists=time()     
    argvlists=getcliarg()  
       
    for i in argvlists: 
        sys.argv=i  
        torch.cuda.empty_cache() 
        setup_seed(20) 
        cli=getcli()
        # cli.trainer.test(cli.model, datamodule=cli.datamodule) 
        setup_seed(20) 
        cli.trainer.fit(cli.model, datamodule=cli.datamodule) 


