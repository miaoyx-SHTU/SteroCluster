#!/usr/bin/env python
# SteroCluster
# written by Yang Yiwen in 2023/3/12

import os
import sys
#import logging
# import the command line parsing moudle
import argparse

#get path and save to PATH temporarily
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/")


'''
#saving log 
# Remove all handlers associated with the root logger object.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'Stero_Cluster.log')
print(f'saving log in: {filename}')

logging.basicConfig(filename=filename,
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=getattr(logging, "INFO"))
'''

#import other python script to load tools 
import Main # main function of SteroCluster
import bulidLog # use to save print into log files

#get parameter from command line
'''
SteroCluster [--input, -i] input_file [--output, -o] path_of_output [--figout, -f] path_to_save_figs [--logout, -lo] path_to_save_logs [--log -l] 
SteroCluster without any parameters which default run Main.runTools with input as the example file: 'data/E14.5_E1S3_Dorsal_Midbrain_GEM_CellBin.tsv', and output in path of this script.
Can also add more parameter for this script just in same way.
'''
def build_parser() -> argparse.ArgumentParser:
    # build a base parser
    parser = argparse.ArgumentParser(description="Run SteroCluster")

    # option for specifying input file
    parser.add_argument('--input', '-i',
                             type=str,
                             action='store',
                             dest='input_file',
                             #default=,
                             help='Load txt(or tsv)file of express genes in cell which created by Stereo-seq')
   
    # option for specifying output file
    parser.add_argument('--output', '-o',
                             type=str,
                             action='store',
                             dest='output_dir',
                             help='The dir to saving AnnData in HDF5 format')
    
    # option for specifying figure output file
    parser.add_argument('--figout', '-f',
                            type=str,
                            action='store',
                            dest='fig_dir',
                            #default=os.path.dirname(__file__),
                            help='The dir to saving figures')
    
    # option for saving log in files(default：not save, use -log/-l to save )
    parser.add_argument('--log', '-l',
                             #type=bool,
                             action='store_true',
                             dest='save_log',
                             default=False,
                             help='Whether to save log')
    
    # option for specifying log file
    parser.add_argument('--logout', '-lo',
                            type=str,
                            action='store',
                            dest='log_dir',
                            #default=os.path.dirname(__file__),
                            help='The dir to saving log')
    
    #the defaults parameter saving in parser
    parser.set_defaults(func=Main.runTools, # save function
                        default_input_file=os.path.join(os.path.dirname(__file__), # save default input path
                                                  'data/E14.5_E1S3_Dorsal_Midbrain_GEM_CellBin.tsv'),
                        default_output_dir=os.path.dirname(__file__)) # save default out put path
    


    # print help without -h
    if len(sys.argv) < 2: parser.print_help(sys.stderr)
    return parser




if __name__ == '__main__':

    parser = build_parser()
    args = parser.parse_args()

    #create output dir if needed
    if args.output_dir is None:
        args.get_output_in_commandline = False # use to decide wether to print message
        output_dir = args.default_output_dir 
        #print(f"No output dir specified, will save in: {output_dir}!")
    else:
        args.get_output_in_commandline = True
        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    args.output_dir = output_dir # use args.default_output_dir  to replace empty args.output_dir

    #saving log 
    if args.save_log:#wether saving logs
        if args.log_dir is None:
            args.log_dir = args.output_dir # defaults save logs in output_path/log
        log_dir =  os.path.join(args.log_dir,'log')
        #see details in buildLog.py
        sys.stdout = bulidLog.Logger(log_dir,
                                     log_name='SteroCluster',
                                     stream=sys.stdout)  # record log
        sys.stderr = bulidLog.Logger(log_dir,
                                     log_name='SteroCluster',
                                     stream=sys.stderr)  # record error 

    #run tools
    args.func(args)



