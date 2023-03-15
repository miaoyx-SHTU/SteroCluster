import sys
import os
import argparse # import for typing
import platform as pl# import for get Session information
from datetime import datetime #import for caculate time cost
from typing import Any, Callable

import anndata 
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages # use to save figs as PDF format

class Timer:
    def __init__(self, perfix: str) -> None:
        #allow customilized perfix of time messages
        self.perfix = perfix

    def __call__(self, func: Callable) -> Callable:

        def wrapper(*args: Any, **kwds: Any) -> Callable:
            
            start = datetime.now()
            ret = func(*args, **kwds)
            time_cost = datetime.now() - start
            minute = time_cost.seconds // 60 # get minute
            second = time_cost.seconds % 60  # get second
            microsecond = time_cost.microseconds / 1000 # get microsecond
            print(f'{self.perfix}:{minute}Min {second}Sec {microsecond}Ms') # print time message

            return ret
        
        return wrapper

'''
A decorator, use to print runing time of any function
the function is equal to :
timer = Timer(perfix)
func = timer(func)
func(args)
'''

class printPDF:
    def __init__(self,name: str, path: str = os.path.dirname(__file__)) -> None:
        
        self.name = name # allow to customilized the name of fig file
        self.path = path # allow to customilized save path when don't get path in kwds

    def __call__(self, func: Callable) -> Callable:

        def wrapper(*args: Any, **kwds: Any) -> Callable:
            # use path in kwds if possible 
            if 'path' in kwds.keys():
                self.path = os.path.realpath(kwds['path'])
            # make dir for saving figs
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            print(f'Image {self.name} will save in {self.path}')
            # saving function 
            with PdfPages(os.path.join(self.path,f"{self.name}.pdf")) as pdf:

                ret = func(*args, **kwds)
                pdf.savefig()
                plt.close()
            
            return ret
        
        return wrapper

'''
A decorator, 
use to save plot function as pdf
'''


@Timer(perfix = 'Total Run Time') # add a decorator to function
def runTools(args: argparse.Namespace) -> Any:
    printSessionInfo() # print SessionInfo of working place
    # get input 
    if args.input_file is None:
        print("No input file specified, using example file instead!")
        input_file = args.default_input_file
    else:
        print(f"Loading input file: {args.input_file} ")
        input_file = args.input_file
    '''   
    if args.output_dir is None:
        output_dir = args.default_output_dirm
        print(f"No output dir specified, will save in: {output_diri}!")
    else:
        output_dir = args.output_dira
        if not os.path.exists(output_diro):
            os.makedirs(output_dir)
        # build yx
    '''
    # get output path
    output_dir = args.output_dir
    if not args.get_output_in_commandline:
        print(f"No output dir specified, will save in: {output_dir}!")

    # get figs output path
    fig_dir = args.fig_dir
    if fig_dir is None:
        fig_dir = os.path.join(output_dir,'fig')
    


    #print(input_file,output_dir)
    ann_data = generateExpAnn(input_file) # Generate counts matrix and coordinate of cells and make a Anndata
    ann_data.write(os.path.join(output_dir, 'row_martix.h5ad'), compression="gzip") # save Anndata as HDF5 format with raw counts matrix
    ann_data = spatialBasicAnalysis(ann_data, fig_dir = fig_dir ) # spatial basic-analysis in Scanpy, stop after clustered cells.
    ann_data.write(os.path.join(output_dir, 'clustered_martix.h5ad'), compression="gzip") # save clustered Anndata as HDF5 format with filtered counts matrix


#print session information
def printSessionInfo() -> None:

    profile = [
    'architecture',
    'machine',
    'node',
    'platform',
    'processor',
    'python_build',
    'python_compiler',
    'python_version',
    'release',
    'system',
    'version',
    ]


    class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

    print("Session Information:\n" )
    for key in profile:
        if hasattr(pl, key):
            print(key + bcolors.BOLD + ": " + str(getattr(pl, key)()) + bcolors.ENDC)
    print("\n")
    sc.logging.print_versions() # print packages information
    print("\n")

@Timer(perfix='Generate Matrix Run Time')
def generateExpAnn(file: str) -> anndata.AnnData: # Generate counts matrix and coordinate of cells and make a Anndata
    #read input file
    cellexp_df = pd.read_csv(filepath_or_buffer = file,sep = '\t',
                dtype = {'geneID':str,
                         'x':int,
                         'y':int,
                         'MIDCounts':np.int32,
                         'cell':int} )
    # use pivot_table function to change long data into wide data (using to generate counts matrix and coordinate of cells)
    cellexp_wide = cellexp_df.pivot_table(index='cell', columns='geneID', values=['MIDCounts','x','y'])
    # save counts matrix
    counts = cellexp_wide.MIDCounts.fillna(0)
    print(f'A expression matrix generated! Shape: {counts.shape[0]} Cells x {counts.shape[1]} Genes')
    # calculate mean coordinate of each cells
    x_coord = cellexp_wide.x.apply(np.nanmean,axis=1)
    y_coord = cellexp_wide.y.apply(np.nanmean,axis=1)
    #save as DataFrame
    coor_df = pd.DataFrame(data = {'x':x_coord,'y':y_coord},
                           index = x_coord.index)
    #add a perfix of cell number
    counts.index = ['Cell_'+str(x) for x in counts.index]
    coor_df.index = coor_df.index.map(lambda x: 'Cell_'+str(x))
    # create Anndata from counts martix
    ann_data = sc.AnnData(counts, dtype=np.float32)
    ann_data.var_names_make_unique()
    # add coordinate of cells to Anndata, save in Anndata.obsm
    coor_df = coor_df.loc[ann_data.obs_names, ['y', 'x']]
    ann_data.obsm["spatial"] = coor_df.to_numpy()

    return ann_data

@Timer(perfix='spatial Basic Analysis Run Time')
def spatialBasicAnalysis(ann_data: anndata.AnnData,fig_dir: str) -> anndata.AnnData: # Refer to: https://scanpy-tutorials.readthedocs.io/en/latest/spatial/basic-analysis.html
    #  calculate standards QC metrics with pp.calculate_qc_metrics and percentage of mitochondrial read counts per sample.
    ann_data.var["mt"] = ann_data.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(ann_data, qc_vars=["mt"], inplace=True)
    # visualizing QC metrics 
    drawQCFig(ann_data,path = fig_dir)
    # perform some basic filtering of spots based on total counts and expressed genes
    ann_data = runQC(ann_data)
    # proceed to normalize Visium counts data with the built-in normalize_total method from Scanpy, and detect highly-variable genes (for later).
    ann_data = runPreprocessing(ann_data)
    # To embed and cluster the manifold encoded by transcriptional similarity, we proceed as in the standard clustering tutorial.
    ann_data = runCluster(ann_data)
    # plot some covariates to check if there is any particular structure in the UMAP associated with total counts and detected genes.
    drawClusterFig(ann_data,path = fig_dir)
    #  take a look at how n_genes_by_counts behave in spatial coordinates and visualizing clustered samples in spatial dimensions.
    drawSpatialFig(ann_data,path = fig_dir)


    return ann_data
# function of QC
def runQC(ann_data: anndata.AnnData) -> anndata.AnnData:
    sc.pp.filter_cells(ann_data, min_counts=50) # here change to 50 beacuse of example data, the value is setting casually 
    sc.pp.filter_cells(ann_data, max_counts=35000)
    ann_data = ann_data[ann_data.obs["pct_counts_mt"] < 20]
    print(f"#cells after MT filter: {ann_data.n_obs}")
    sc.pp.filter_genes(ann_data, min_cells=10)

    return ann_data
# function of preprocessing
def runPreprocessing(ann_data: anndata.AnnData) -> anndata.AnnData:
    sc.pp.normalize_total(ann_data, inplace=True)
    sc.pp.log1p(ann_data)
    sc.pp.highly_variable_genes(ann_data, flavor="seurat", n_top_genes=2000)

    return ann_data
# function of standard clustering tutorial (include run pca, fund neighbors, run umap, clustering use leiden)
@Timer(perfix='PCA and Cluster Run Time')
def runCluster(ann_data: anndata.AnnData) -> anndata.AnnData:
    sc.pp.pca(ann_data)
    sc.pp.neighbors(ann_data)
    sc.tl.umap(ann_data)
    sc.tl.leiden(ann_data, key_added="clusters")

    return ann_data

'''
Here is some figure drawing functions, all save in PDF format.
'''

@printPDF(name='QC_counts_fig')
def drawQCFig(ann_data: anndata.AnnData, path: str) -> None:
    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    #`distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    # use  `histplot` instead
    sns.histplot(ann_data.obs["total_counts"], kde=False, ax=axs[0])
    sns.histplot(ann_data.obs["total_counts"][ann_data.obs["total_counts"] < 10000], kde=False, bins=40, ax=axs[1])
    sns.histplot(ann_data.obs["n_genes_by_counts"], kde=False, bins=60, ax=axs[2])
    sns.histplot(ann_data.obs["n_genes_by_counts"][ann_data.obs["n_genes_by_counts"] < 4000], kde=False, bins=60, ax=axs[3])

@printPDF(name='UMAP_cluster_fig')
def drawClusterFig(ann_data: anndata.AnnData, path: str) -> None:
    plt.rcParams["figure.figsize"] = (4, 4)
    sc.pl.umap(ann_data, color=["total_counts", "n_genes_by_counts", "clusters"], wspace=0.4, show=False)

@printPDF(name='Spatial_coordinates_fig')
def drawSpatialFig(ann_data: anndata.AnnData, path: str) -> None:
    plt.rcParams["figure.figsize"] = (8, 8)
    sc.pl.embedding(ann_data, basis="spatial", color=["n_genes_by_counts",'clusters'], show=False)

