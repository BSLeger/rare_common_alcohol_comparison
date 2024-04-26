#import functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import matplotlib
from scipy.stats import hypergeom
from matplotlib_venn import venn2 
import ndex2
import networkx as nx
from netcoloc import netprop_zscore
from netcoloc import netprop
from netcoloc import network_colocalization
import re

#variables
#pcnet v1.4- used for prior analysis
#pcnet v1.3- used for rat bmi network paper
#pcnet v2.0- not currently available to the public
UUIDs= {
    'pcnet_v14':'c3554b4e-8c81-11ed-a157-005056ae23aa',
    'pcnet_v13':'4de852d9-9908-11e9-bcaf-0ac135e8bacf',
    'string':'98ba6a19-586e-11e7-8f50-0ac135e8bacf',
    'humanNet_v3_FN': '40913318-3a9c-11ed-ac45-0ac135e8bacf',
    'pcnet_v20':'f5767e8b-cdcc-11ee-93f7-005056ae23aa',
    'ACN':'f81a3f67-4215-11ee-aa50-005056ae23aa',
    'ACN2':'3537279e-ceea-11ee-93f7-005056ae23aa',
	'ACN_strin':'48de252c-3d50-11ee-aa50-005056ae23aa',
	'UKBB_PCNET2':'87c318a5-ce41-11ee-93f7-005056ae23aa'
}
UUID_tag={
    'pcnet_v14':'',
    'pcnet_v13':'_pcnet_v13',
    'pcnet_v20':'_pcnet_v20',
    'string':'_string',
    'humanNet_v3_FN': '_humanNet_v3_FN'
}
color_dict={
    'rare':'#25BE93',
    'rare_alt':'#09694e',
    'common':'#C673dc',
    'common_alt':'#642475',
    'shared':'#3636eb',
    'other':'#CCCCCC',
    'SKAT':'#ff03c8',
    'SKAT-O':'#474b96',
    'burden':'#fcba03',
    'pLoF':'#fcba03',
    'misLC':'#189B48',
    'syn':'#ff03c8',
    'pLoF_alt':'#af7500',
    'misLC_alt':'#16542d',
    'syn_alt':'#6b0355'
}

#network formating functions
def format_network(network, traitr, traitc, seedr, seedc,zr, zc):
    """
    Formats the colocalized network for easy secondary analysis
    This function takes a network (graph) and updates its nodes with several attributes:
    - Seed gene indicators for two traits with secondary "color scheme" indicator for use in cytoscape color scheme assignment
    - Z-scores for each trait and their combination

    Parameters:
    - network (NetworkX graph): The graph to be formatted, where nodes represent genes.
    - traitr (str): Identifier for the rare trait.
    - traitc (str): Identifier for the common trait.
    - seedr (list): List of seed genes associated with the rare trait.
    - seedc (list): List of seed genes associated with the common trait.
    - zr (dict): Dictionary mapping genes to their z-scores for the rare trait.
    - zc (dict): Dictionary mapping genes to their z-scores for the common trait.

    Returns:
    - NetworkX graph: The original network updated with node attributes for seed gene status and z-scores for traits and their combination.

    Note:
    The function utilizes pandas DataFrames for intermediate data manipulation and requires 
    NetworkX for working with the network. It expects `network.nodes()` to return a list-like 
    object of genes.
    """
    nodes_df=pd.DataFrame(network.nodes())
    nodes_df.columns=['Gene']
    #make node seed gene dataframe from which to make dictionaries
    nodes_df[('seed_'+traitr)]=nodes_df['Gene'].isin(seedr)
    nodes_df[('seed_'+traitc)]=nodes_df['Gene'].isin(seedc)
    nodes_df['seed_both']=(nodes_df['Gene'].isin(seedr) & nodes_df['Gene'].isin(seedc))
    
    nodes_df['color_scheme']=0
    nodes_df['color_scheme'] = np.where(nodes_df[('seed_'+traitr)] == True, 1, nodes_df['color_scheme'])
    nodes_df['color_scheme'] = np.where(nodes_df[('seed_'+traitc)] == True, 2, nodes_df['color_scheme'])
    nodes_df['color_scheme'] = np.where((nodes_df["seed_both"] == True), 3, nodes_df['color_scheme'])    
    nodes_df.index=nodes_df['Gene']
    #set zscores as node attributes
    nx.set_node_attributes(network, dict(zr), ('z_'+traitr))
    nx.set_node_attributes(network, dict(zc),('z_'+traitc))
    nx.set_node_attributes(network, dict(zr*zc), 'z_comb')
    #add seed genes as node attributes
    nx.set_node_attributes(network,dict(zip(nodes_df['Gene'], nodes_df[('seed_'+traitr)])), ('seed_'+traitr))
    nx.set_node_attributes(network,dict(zip(nodes_df['Gene'], nodes_df[('seed_'+traitc)])), ('seed_'+traitc))
    nx.set_node_attributes(network,dict(zip(nodes_df['Gene'], nodes_df['seed_both'])), ('seed_both'))
    nx.set_node_attributes(network,dict(zip(nodes_df['Gene'], nodes_df['color_scheme'])), ('seed_color_scheme'))
    return(network)
    
def export_network(network, name, user, password, ndex_server='public.ndexbio.org'):
    '''
    shell for net_cx upload network function, that creates nicecx network, then exports to NDEx in the CX format.

    This function converts a NetworkX graph to a NiceCXNetwork object using the ndex2 Python package,
    sets the network's name, and uploads it to the specified NDEx server. Upon successful upload, 
    the function returns the UUID of the network in the NDEx platform.

    Parameters:
    - network (NetworkX graph): The graph to be exported.
    - name (str): The name to assign to the network in NDEx.
    - user (str): NDEx account username.
    - password (str): NDEx account password.
    - ndex_server (str, optional): The URL of the NDEx server to which the network is to be uploaded. Defaults to 'public.ndexbio.org'.

    Returns:
    - exports network to NDEx
    - str: The UUID of the uploaded network on NDEx.
    Notes:
    Tequires the ndex2 package. Ensure that you have a valid NDEx account and that the specified server URL is correct.
    """
    '''
    print(user)
    print(password)
    if ((user==None) | (password==None)):
        print('please provide a NDEx username and password.')
    else:
        net_cx = ndex2.create_nice_cx_from_networkx(network)
        net_cx.set_name(name)
        network_uuid = net_cx.upload_to(ndex_server, user, password)

#plotting functions

#adapted from BMI
def venn_net(tblr, tblc, tblr_label, tblc_label, p_net_overlap,tblr_lim=1.5, tblc_lim=1.5, comb_lim=3, savefig=False):    
    """
    Generates a Venn diagram to visualize the overlap between genes  based on z-score thresholds.

    This function combines two input tables with z-scores, calculates a combined z-score by multiplying
    the z-scores from each table, and identifies items present in both tables based on specified z-score 
    thresholds. It then generates a Venn diagram visualizing the overlap and unique items in each dataset,
    and saves the figure as an SVG file.

    Parameters:
    - tblr (pandas.DataFrame): First input table with z-scores for each item.
    - tblc (pandas.DataFrame): Second input table with z-scores for each item.
    - tblr_label (str): Label for the first dataset to be used in the Venn diagram.
    - tblc_label (str): Label for the second dataset to be used in the Venn diagram.
    - p_net_overlap (float): p-value for the network overlap to be displayed in the diagram title.
    - tblr_lim (float, optional): Threshold for z-scores in the first table. Defaults to 1.5.
    - tblc_lim (float, optional): Threshold for z-scores in the second table. Defaults to 1.5.
    - comb_lim (float, optional): Threshold for the combined z-score to consider an item present
    - savefig (boolean, optional): determine whether or not the figure will be saved as an SVG
      in the network. Defaults to 3.

    Returns:
    None: The function saves the Venn diagram as an SVG file and does not return any value.

    Note:
    - The function assumes that the input tables are pandas DataFrames with z-scores.
    - The `color_dict` variable, used to define set colors, and the `UUID_tag` and `interactome_name`
      variables, used in the filename, are assumed to be defined outside of this function.
    - This function utilizes the matplotlib_venn.venn2 function for creating the Venn diagram and
      matplotlib for plotting and saving the figure. Ensure these libraries are imported and properly configured.
    """    
    
    #combine zscore tables
    tbl_z=pd.concat([tblr, tblc], axis=1)
    tbl_z.columns=('z1','z2')
    tbl_z['z_comb']=tbl_z['z1']*tbl_z['z2']
    #subset table to those within network limit parameters
    inNetwork=tbl_z[(tbl_z['z1']>=tblr_lim) & (tbl_z['z2']>=tblc_lim) & (tbl_z['z_comb']>=comb_lim)]
    #plot venn diagram
    venn2(((len(tbl_z[tbl_z['z1']>tblr_lim])-len(inNetwork)), (len(tbl_z[tbl_z['z2']>tblc_lim])-len(inNetwork)), len(inNetwork)), 
          set_labels=(tblr_label, tblc_label), 
          set_colors=(color_dict['rare'], color_dict['common']), alpha = 0.7)
    plt.title('p='+str(p_net_overlap)+ ', single cut='+str(tblr_lim)+', comb cut='+str(comb_lim))
    if savefig:
        plt.savefig('figures/network_venn/network_venn_'+tblr_label+'_'+tblc_label+UUID_tag[interactome_name]+'.svg',bbox_inches='tight')
    plt.show()
    
#adapted from BMI
def venn_seeds(tblr_seed, tblc_seed, tblr_label, tblc_label, all_nodes, savefig=False):
    tblr_seed=list(set(tblr_seed).intersection(all_nodes))
    tblc_seed=list(set(tblc_seed).intersection(all_nodes))  
    #define overlap for seed genes plot
    seed_overlap=set(tblr_seed).intersection(set(tblc_seed))
    print(seed_overlap)
    #compute significance of seed genes overlap- same test as used in BMI paper
    hyper = hypergeom(M=len(all_nodes), n=len(tblr_seed), N=len(tblc_seed))
    p_intersect_seed = hyper.sf(len(seed_overlap))
    
    venn2((len(tblr_seed)-len(seed_overlap), len(tblc_seed)-len(seed_overlap), len(seed_overlap)), 
          set_labels=(tblr_label, tblc_label), 
          set_colors=(color_dict['rare'], color_dict['common']), alpha = 0.7)
    plt.title(' Seed Gene Overlap, p='+str(p_intersect_seed))
    if (savefig):
        plt.savefig('figures/seed_venn/seed_venn_'+tblr_label+'_'+tblc_label+UUID_tag[interactome_name]+'.svg',bbox_inches='tight')
    plt.show()

#adapted from BMI functions
def plt_histogram (tblr, tblc, tblr_label, tblc_label, tblr_seed, tblc_seed, tblr_lim=1.5, tblc_lim=1.5, comb_lim=3, savefig=False):   
    fig, [ax1, ax2, ax3, ax4] = plt.subplots(nrows=1, ncols=4, figsize=(25, 5))
    _, bins, _ = ax1.hist(tblr, bins=100, alpha=0.7, density=True, label=tblr_label, color=color_dict['rare'])
    _ = ax1.hist(tblc, bins=bins, alpha=0.7, density=True, label=tblc_label, color=color_dict['common'])
    ax1.set_ylabel("density")
    ax1.set_xlabel("proximity zscore")
    ax1.legend()

    _, bins, _ = ax2.hist(tblr[~tblr.index.isin(tblr_seed)], bins=100, alpha=0.7, density=True, label=tblr_label,color=color_dict['rare'] )
    _ = ax2.hist(tblc[~tblc.index.isin(tblc_seed)], bins=bins, alpha=0.7, density=True, label=tblc_label, color=color_dict['common'])
    ax2.set_ylabel("density")
    ax2.set_xlabel("proximity zscore (no seed genes)")
    ax2.legend()
    
    _, bins, _ = ax3.hist(tblc['z']*tblr['z'], bins=bins, alpha=0.7, density=True, label='combined score', color=color_dict['shared'])
    ax3.set_xlabel("Combined proximity zscore")
    ax3.set_ylabel("density")
    
    #combine zscore tables
    tbl_z=pd.concat([tblr, tblc], axis=1)
    tbl_z.columns=('z1','z2')
    tbl_z['z_comb']=tbl_z['z1']*tbl_z['z2']
    inNetwork=tbl_z[(tbl_z['z1']>tblr_lim) & (tbl_z['z2']>tblc_lim) & (tbl_z['z_comb']>comb_lim)]
    outNetwork=tbl_z[(tbl_z['z1']<=tblr_lim) | (tbl_z['z2']<=tblc_lim) | (tbl_z['z_comb']<=comb_lim)]
    
    ax4.scatter(x=outNetwork['z1'], y=outNetwork['z2'], s=1, color=color_dict['other'])
    ax4.scatter(x=inNetwork['z1'], y=inNetwork['z2'], s=1, color=color_dict['shared'])
        
    ax4 = plt.xlabel(tblr_label)
    
    plt.ylabel(tblc_label)
    plt.axvline(x = tblr_lim, color=color_dict['rare'], linestyle = 'dashed', linewidth=1)
    plt.axhline(y = tblc_lim, color = color_dict['common'], linestyle = 'dashed', linewidth=1)
    x_points = [(i+0.0001)/10 for i in range(-50,250)]
    combo_line = [comb_lim/x for x in x_points if x > comb_lim/50]
    plt.plot([x for x in x_points if x > comb_lim/40], combo_line, color=color_dict['shared'], linestyle='dashed',linewidth=1)
    plt.axvline(x = 0, color='black', linestyle = 'solid', linewidth=1)
    plt.axhline(y = 0, color = 'black', linestyle = 'solid', linewidth=1)
    if (savefig):
        plt.savefig('figures/histogram/histogram_'+tblr_label+'_'+tblc_label+UUID_tag[interactome_name]+'.svg',bbox_inches='tight')
    plt.show()


def import_seedgenes(path,pcol='P',gene_col='GENE NAME',delim='comma', cutoff=None):
    if delim=='comma':
        df=pd.read_csv(path,sep=',')
    else:
        df=pd.read_csv(path,sep='\t')
    if pcol==None:
        print('pvalue column not specified- all genes will be used')
        cutoff=None
    if cutoff=='bonferroni':
        df=df[df[pcol]<0.05/len(df)]
    elif cutoff=='FDR':
        df['pval_FDR']=statsmodels.stats.multitest.fdrcorrection(df[pcol],alpha=0.05,method='indep',is_sorted=False)[1]
        df=df[df['pval_FDR']<0.05]
    else:
        print('cutoff not defined/custom- using all genes ')
        df=df
    print(df.head())
    #gene_ls=list(set(df[gene_col]))
    #return(gene_ls)
    return(df)
def import_nps_zscores(z_path, interactome_name='pcnet_v14'):
    if ((interactome_name=='pcnet_v14')|(interactome_name==None)):
        zscore_rare_df=pd.read_csv(z_path.lower(),header=None, sep='\t')
        print('importing file: '+z_path.lower())
    else:
        z_path_head=z_path[0:len(z_path)-11:1]
        z_path_tail=z_path[len(z_path)-11:len(z_path):1]
        zscore_rare_df=pd.read_csv((z_path_head+'_'+interactome_name+z_path_tail).lower(),header=None, sep='\t')
        print('importing file: '+(z_path_head+'_'+interactome_name+z_path_tail).lower())
    zscore_rare_df.index=zscore_rare_df[0]
    zscore_rare_df=zscore_rare_df.drop(columns=[0])
    zscore_rare=zscore_rare_df[1].squeeze()
    zscore_rare_df = pd.DataFrame({'z':zscore_rare})
    print(zscore_rare_df.head())
    return(zscore_rare_df)

#network propagation functions
def run_net_prop(path, trait_name,pcol,gene_col,delim,cutoff=None, graph=None, w_double_prime=None, interactome='pcnet_v14', ndex_user=None, ndex_password=None):
    """
    Executes network propagation analysis for a given trait using provided seed genes and provided interactome.

    Parameters:
    - path (str): The file path to the seed gene file.
    - trait_name (str): The name of the trait for which the analysis is being run.
    - pcol (str): The column name in the seed genes file that contains the p-values.
    - gene_col (str): The column name in the seed genes file that specifies the gene names.
    - delim (str): The delimiter used in the seed genes file.
    - cutoff (float, optional): The p-value cutoff for filtering seed genes. If None (Default), no filtering is applied. Defaults to None.
    - graph (NetworkX graph, optional): The interactome network graph. If None, the graph is imported using the interactome parameter. Defaults to None.
    - w_double_prime (numpy.ndarray, optional): Pre-calculated matrix for network propagation. If None, it is calculated in the function. Defaults to None.
    - interactome (str, optional): The name of the interactome. If no graph is provided, this will be imported using the import_interactome function which accepts UUIDs or keys to the UUIDs dictionary. Will used as a label for exported interactome files. Defaults to 'pcnet_v14', which was used for this analysis.
    - ndex_user (str, optional): NDEx account username, required if uploading results to NDEx. Defaults to None.
    - ndex_password (str, optional): NDEx account password, required if uploading results to NDEx. Defaults to None.

    Returns:
    None. It saves the z-scores and heats results to files.

    Notes:
    - The function requires an external library for network propagation calculations.
    - The seed genes file should contain a column for genes and a column for their associated p-values.
    - The function saves three files: z-scores, raw heats, and randomized heats for the network analysis,
      with the trait name and optionally the interactome name as part of the filenames.
    - If using a private interactome, ensure the ndex_user and ndex_password are correctly provided.
    """
    data=import_seedgenes(path,pcol,gene_col,delim)
    if (graph==None):
        graph=import_interactome(interactome)
        print("importing network "+interactome)
    if (w_double_prime==None):    
        # pre calculate mats used for netprop
        print('\ncalculating w_prime')
        w_prime = netprop.get_normalized_adjacency_matrix(graph, conserve_heat=True) 
        print('\ncalculating w_double_prime')
        w_double_prime = netprop.get_individual_heats_matrix(w_prime, .5)
    else:
        print("using provided w_double_prime- please ensure that w_double_prime aligns to graph provided")
    graph_nodes = list(graph.nodes)
    data=list(set(df[gene_col]).intersection(graph_nodes))
    ##calculate heats
    z_score, Fnew_score, Fnew_rand_score = netprop_zscore.calculate_heat_zscores(w_double_prime,  
                                                                graph_nodes,
                                                                dict(graph.degree), 
                                                                data, num_reps=1000,
                                                                minimum_bin_size=100)
    export_path='network_scores/'
    if ((graph==None)&(interactome=='pcnet_v14')):
        prefix=(export_path+trait_name).lower()
    elif ((graph==None)&(interactome!='pcnet_v14')):
        prefix=(export_path+trait_name+'_'+interactome).lower()
    elif ((graph!=None)&(interactome!='pcnet_v14')):
        prefix=(export_path+trait_name+'_'+interactome).lower()
    else:
        "saving file without interactome_prefix, please provide an interactome name if prefix wanted"
        prefix=('network_scores/'+trait_name).lower()
    z_score.to_csv((prefix+'_zscore.tsv'),sep='\t',header=False)
    Fnew_score.to_csv((prefix+'_heats.tsv'),sep='\t',header=False)
    pd.DataFrame(Fnew_rand_score, columns=z_score.index).to_csv((prefix+'_randheats.tsv'),sep='\t')

def run_net_prop_subsampling(path, trait_name,pcol,gene_col,delim,cutoff=None, graph=None, w_double_prime=None, interactome='pcnet_v14', ndex_user=None, ndex_password=None):
    data=import_seedgenes(path,pcol,gene_col,delim)
    if (graph==None):
        graph=import_interactome(interactome)
        print("importing network "+interactome)
    if (w_double_prime==None):    
        # pre calculate mats used for netprop
        print('\ncalculating w_prime')
        w_prime = netprop.get_normalized_adjacency_matrix(graph, conserve_heat=True) 
        print('\ncalculating w_double_prime')
        w_double_prime = netprop.get_individual_heats_matrix(w_prime, .5)
    else:
        print("using provided w_double_prime- please ensure that w_double_prime aligns to graph provided")
    graph_nodes = list(graph.nodes)
    data=list(set(df[gene_col]).intersection(graph_nodes))
    ##calculate heats
    z_score, Fnew_score, Fnew_rand_score = netprop_zscore.calculate_heat_zscores(w_double_prime,  
                                                                graph_nodes,
                                                                dict(graph.degree), 
                                                                data, num_reps=1000,
                                                                minimum_bin_size=100)
    export_path='network_scores/subsampling/'
    if ((graph==None)&(interactome=='pcnet_v14')):
        prefix=(export_path+trait_name).lower()
    elif ((graph==None)&(interactome!='pcnet_v14')):
        prefix=(export_path+trait_name+'_'+interactome).lower()
    elif ((graph!=None)&(interactome!='pcnet_v14')):
        prefix=(export_path+trait_name+'_'+interactome).lower()
    else:
        "saving file without interactome_prefix, please provide an interactome name if prefix wanted"
        prefix=('network_scores/'+trait_name).lower()
    prefix=('')
    z_score.to_csv((prefix+'_zscore.tsv'),sep='\t',header=False)
    Fnew_score.to_csv((prefix+'_heats.tsv'),sep='\t',header=False)
    pd.DataFrame(Fnew_rand_score, columns=z_score.index).to_csv((prefix+'_randheats.tsv'),sep='\t')


def import_interactome(interactome_name=None, ndex_user=None, ndex_password=None,UUID=None):
    interactome_uuid=UUIDs[interactome_name]
    print(interactome_name)
    ndex_server='public.ndexbio.org'
    #import network based on provided interactome key
    if (interactome_name in UUIDs.keys()):
        graph = ndex2.create_nice_cx_from_server(
                    ndex_server, 
                    username=ndex_user, 
                    password=ndex_password, 
                    uuid=interactome_uuid
                ).to_networkx()
        if (interactome_name=='pcnet_v14'):
            graph=nx.relabel_nodes(graph, nx.get_node_attributes(graph, 'HGNC Symbol'))
        # print out interactome num nodes and edges for diagnostic purposes
        print('number of nodes:')
        print(len(graph.nodes))
        print('\nnumber of edges:')
        print(len(graph.edges))
        return(graph)
    elif(interactome_name==None & UUID!=None):
        print('using novel UUID. For UUIDs used in this study, see UUID_dict')
        graph = ndex2.create_nice_cx_from_server(
            ndex_server, 
            username=ndex_user, 
            password=ndex_password, 
            uuid=UUID
        ).to_networkx()
        # print out interactome num nodes and edges for diagnostic purposes
        print('number of nodes:')
        print(len(graph.nodes))
        print('\nnumber of edges:')
        print(len(graph.edges))
        return(graph)
    else:
        print('UUID/interactome name not provided- please provide either to import interactome.')
    #relabel the nodes with the gene name instead of an arbitrary number


#non-network functions

#manhattan
def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)
def porcupine(p1, test1, pos1, chr1, label1,
                cut_SKAT=5e-8,
                cut_SKATO=5e-8,
                cut_burden=5e-8,
                colors_test=['#fcba03','#474b96','#ff03c8'],
               chrs_plot=None, chrs_names=None,
               cut = 2,
               colors = ['k', '0.5'],
               title='Title',
               xlabel='chromosome',
               ylabel='-log10(p-value)',
               top1 = 0,
               top2 = 0,
               lines = [10, 15],
               lines_colors = ['g', 'r'],
               lines_styles = ['-', '--'],
               lines_widths = [1,1],
               zoom = None,
               scaling = '-log10', 
              plot_grid_lines=True,
              **kwargs):
    '''
    #update meta
    Static Porcupine plot
    :param p1: p-values for the top panel
    :param pos1: positions
    :param chr1: chromosomes numbers
    :param label1: label
    :param color_peaks: ordered ['burden','skato','skat']- for the peaks
    :param chrs_plot: list of chromosomes that should be plotted. If empty [] all chromosomes will be plotted
    :param cut: lower cut (default 2)
    :param colors: sequence of colors (default: black/gray)
    :param title: defines the title of the plot
    :param xlabel: defines the xlabel of the plot
    :param ylabel: defines the ylabel of the plot
    :param top: Defines the upper limit of the plot. If 0, it is detected automatically.
    :param lines: Horizontal lines to plot.
    :param lines_colors: Colors for the horizontal lines.
    :param lines_styles: Styles for the horizontal lines.
    :param lines_widths: Widths for the horizontal lines.
    :param zoom: [chromosome, position, range] Zooms into a region.
    :param scaling: '-log10' or 'none' (default -log10)
    :param plot_grid_lines: Should chromosome dividers be plotted? (default True)
    :return:
    '''
    # Setting things up
    shift=np.array([0.0])
    plt.clf()

    # If chrs_plot is empty, we need to generate a list of chromosomes
    if chrs_plot is None:
        chrs_list = np.unique(chr1)
        if isinstance(chrs_list[0], str):
            chrs_list = sorted_nicely(chrs_list)
        else:
            chrs_list.sort()
    else:
        chrs_list = chrs_plot

    # If chrs_names is empty, we need to generate a list of names for chromosomes
    if chrs_names is None:
        chrs_names = [str(chrs_list[i]) for i in range(len(chrs_list))]

    plot_positions = False
    if len(chrs_list) == 1:
        plot_positions = True

    if scaling=='-log10':
        cut_burden=-np.log10(cut_burden)
        print(cut_burden)
        cut_SKATO=-np.log10(cut_SKATO)
        print(cut_SKATO)
        cut_SKAT=-np.log10(cut_SKAT)
        print(cut_SKAT)
        
    for ii, i in enumerate(chrs_list):     
        plt.subplot(1,1,1)
        # print(i)
        filt = np.where(chr1==i)[0]
        x = shift[-1]+pos1[filt]
        if scaling=='-log10':
            y = -np.log10(p1[filt])
        elif scaling=='none':
            y = p1[filt]
        else:
            raise ValueError('Wrong "scaling" mode. Choose between "-log10" and "none"')
        f=test1[filt]
        plt.plot(x[y>cut], y[y>cut], '.', color=colors[ii % len(colors)], **kwargs)
        if scaling=='-log10':  
            plt.plot(x[(y>cut) & (f=='Burden') &(y>cut_burden)], y[(y>cut) & (f=='Burden') &(y>cut_burden)], '.', color=colors_test[0],**kwargs)
            plt.plot(x[(y>cut) & (f=='SKATO') &(y>cut_SKATO)], y[(y>cut) & (f=='SKATO') &(y>cut_SKATO)], '.', color=colors_test[1], **kwargs)
            plt.plot(x[(y>cut) & (f=='SKAT') &(y>cut_SKAT)], y[(y>cut) & (f=='SKAT') &(y>cut_SKAT)], '.', color=colors_test[2], **kwargs)
        else:  
            plt.plot(x[(y>cut) & (f=='Burden') &(y>cut_burden)], y[(y>cut) & (f=='Burden') &(y<cut_burden)], '.', color=colors_test[0], **kwargs)
            plt.plot(x[(y>cut) & (f=='SKATO') &(y>cut_SKATO)], y[(y>cut) & (f=='SKATO') &(y<cut_SKATO)], '.', color=colors_test[1], **kwargs)
            plt.plot(x[(y>cut) & (f=='SKAT') &(y>cut_SKAT)], y[(y>cut) & (f=='SKAT') &(y<cut_SKAT)], '.', color=colors_test[2], **kwargs)

        shift_f = np.max(x)


        if zoom is not None:
            if zoom[0] == i:
                zoom_shift = zoom[1] + shift[-1]

        shift_m = 0
        shift = np.append(shift, np.max([shift_f, shift_m]))

        plt.subplot(1,1,1)
        if plot_grid_lines:
            plt.plot([shift[-1], shift[-1]], [0, 1000], '-', lw=0.5, color='lightgray', **kwargs)
            plt.xlim([0, shift[-1]])

        plt.xlim([0, shift[-1]])
        # print(shift)

    # Defining top boundary of a plot
    if top1 == 0:
        if scaling == '-log10':
            top1 = np.ceil(np.max(-np.log10(p1)))
        elif scaling == 'none':
            top1 = np.ceil(np.max(p1))
        else:
            raise ValueError('Wrong "scaling" mode. Choose between "-log10" and "none"')

    # Setting up the position of labels:
    shift_label = shift[-1]
    shift = (shift[1:]+shift[:-1])/2.
    labels = chrs_names

    # Plotting horizontal lines
    for i, y in enumerate(lines):
        plt.subplot(1,1,1)
        plt.axhline(y=y, color=lines_colors[i], linestyle=lines_styles[i], linewidth=lines_widths[i])
    plt.subplot(1,1,1)
    plt.ylim([cut, top1])
    plt.title(title)

    if not plot_positions:
        plt.xticks(shift, labels)

    plt.text(shift_label*0.95,top1*0.95,label1,#bbox=dict(boxstyle="round", fc="1.0"),
            verticalalignment='top', horizontalalignment='right')

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    if zoom is not None:
        plt.subplot(1,1,1)
        plt.xlim([zoom_shift-zoom[2], zoom_shift+zoom[2]])

    return plt
def manhattan(p1, pos1, chr1, label1,
               p2=None, pos2=None, chr2=None, label2=None,
               plot_type='single',
               chrs_plot=None, chrs_names=None,
               cut = 2,
               colors = ['k', '0.5'],
               title='Title',
               xlabel='chromosome',
               ylabel='-log10(p-value)',
               top1 = 0,
               top2 = 0,
               lines = [10, 15],
               lines_colors = ['g', 'r'],
               lines_styles = ['-', '--'],
               lines_widths = [1,1],
               zoom = None,
               scaling = '-log10', 
              plot_grid_lines=True,
              **kwargs):
    '''
    Static Manhattan plot
    :param p1: p-values for the top panel
    :param pos1: positions
    :param chr1: chromosomes numbers
    :param label1: label
    :param p2: p-values for the bottom panel
    :param pos2: positions
    :param chr2: chromosomes numbers
    :param label2: label
    :param type: Can be 'single', 'double' or 'inverted'
    :param chrs_plot: list of chromosomes that should be plotted. If empty [] all chromosomes will be plotted
    :param cut: lower cut (default 2)
    :param colors: sequence of colors (default: black/gray)
    :param title: defines the title of the plot
    :param xlabel: defines the xlabel of the plot
    :param ylabel: defines the ylabel of the plot
    :param top: Defines the upper limit of the plot. If 0, it is detected automatically.
    :param lines: Horizontal lines to plot.
    :param lines_colors: Colors for the horizontal lines.
    :param lines_styles: Styles for the horizontal lines.
    :param lines_widths: Widths for the horizontal lines.
    :param zoom: [chromosome, position, range] Zooms into a region.
    :param scaling: '-log10' or 'none' (default -log10)
    :param plot_grid_lines: Should chromosome dividers be plotted? (default True)
    :return:
    '''

    # Setting things up
    shift=np.array([0.0])
    plt.clf()

    # If chrs_plot is empty, we need to generate a list of chromosomes
    if chrs_plot is None:
        chrs_list = np.unique(chr1)
        if isinstance(chrs_list[0], str):
            chrs_list = sorted_nicely(chrs_list)
        else:
            chrs_list.sort()
    else:
        chrs_list = chrs_plot


    # If chrs_names is empty, we need to generate a list of names for chromosomes
    if chrs_names is None:
        chrs_names = [str(chrs_list[i]) for i in range(len(chrs_list))]

    plot_positions = False
    if len(chrs_list) == 1:
        plot_positions = True


    for ii, i in enumerate(chrs_list):
        if plot_type != 'single':
            ax1 = plt.subplot(2,1,1)
        else:
            plt.subplot(1,1,1)
        # print(i)
        filt = np.where(chr1==i)[0]
        x = shift[-1]+pos1[filt]
        if scaling=='-log10':
            y = -np.log10(p1[filt])
        elif scaling=='none':
            y = p1[filt]
        else:
            raise ValueError('Wrong "scaling" mode. Choose between "-log10" and "none"')
        plt.plot(x[y>cut], y[y>cut], '.', color=colors[ii % len(colors)], **kwargs)
        shift_f = np.max(x)

        if zoom is not None:
            if zoom[0] == i:
                zoom_shift = zoom[1] + shift[-1]

        if plot_type != 'single':
            plt.subplot(2,1,2)#, sharex=ax1)
            filt = np.where(chr2==i)[0]
            x = shift[-1]+pos2[filt]
            if scaling=='-log10':
                y = -np.log10(p2[filt])
            elif scaling=='none':
                y = p2[filt]
            else:
                raise ValueError('Wrong "scaling" mode. Choose between "-log10" and "none"')
            plt.plot(x[y>cut], y[y>cut], '.', color=colors[ii % len(colors)])
            shift_m = np.max(x)
        else:
            shift_m = 0

        shift = np.append(shift, np.max([shift_f, shift_m]))

        if plot_type != 'single':
            plt.subplot(2,1,1)
        else:
            plt.subplot(1,1,1)
        if plot_grid_lines:
            plt.plot([shift[-1], shift[-1]], [0, 1000], '-', lw=0.5, color='lightgray', **kwargs)
            plt.xlim([0, shift[-1]])

            if plot_type != 'single':
                plt.subplot(2,1,2)
                plt.plot([shift[-1], shift[-1]], [0, 1000], '-', lw=0.5, color='lightgray', zorder=0)
        plt.xlim([0, shift[-1]])
        # print(shift)

    # Defining top boundary of a plot
    if top1 == 0:
        if plot_type != 'single':
            if scaling == '-log10':
                top1 = np.ceil(np.max([np.max(-np.log10(p1)), np.max(-np.log10(p2))]))
            elif scaling == 'none':
                top1 = np.ceil(np.max([np.max(p1), np.max(p2)]))
            else:
                raise ValueError('Wrong "scaling" mode. Choose between "-log10" and "none"')
        else:
            if scaling == '-log10':
                top1 = np.ceil(np.max(-np.log10(p1)))
            elif scaling == 'none':
                top1 = np.ceil(np.max(p1))
            else:
                raise ValueError('Wrong "scaling" mode. Choose between "-log10" and "none"')


    if top2 == 0:
        if plot_type != 'single':
            top2 = top1

    # Setting up the position of labels:
    shift_label = shift[-1]
    shift = (shift[1:]+shift[:-1])/2.
    labels = chrs_names

    # Plotting horizontal lines
    for i, y in enumerate(lines):
        if plot_type != 'single':
            plt.subplot(2,1,1)
            plt.axhline(y=y, color=lines_colors[i], linestyle=lines_styles[i], linewidth=lines_widths[i])
            plt.subplot(2,1,2)
            plt.axhline(y=y, color=lines_colors[i], linestyle=lines_styles[i], linewidth=lines_widths[i])
        else:
            plt.subplot(1,1,1)
            plt.axhline(y=y, color=lines_colors[i], linestyle=lines_styles[i], linewidth=lines_widths[i])

    if plot_type != 'single':
        plt.subplot(2,1,1)
        if not plot_positions:
            plt.xticks(shift, labels)
        plt.ylim([cut+0.05, top1])
    else:
        plt.subplot(1,1,1)
        plt.ylim([cut, top1])
    plt.title(title)
    if plot_type != 'single':
        plt.setp(plt.gca().get_xticklabels(), visible=False)
        if not plot_positions:
            plt.xticks(shift)
    else:
        if not plot_positions:
            plt.xticks(shift, labels)

    plt.text(shift_label*0.95,top1*0.95,label1,#bbox=dict(boxstyle="round", fc="1.0"),
            verticalalignment='top', horizontalalignment='right')

    if plot_type != 'single':
        plt.subplot(2,1,2)
        plt.ylim([cut, top2])
        if plot_type == 'inverted':
            plt.gca().invert_yaxis()
        if not plot_positions:
            plt.xticks(shift, labels)
        if plot_type == 'inverted':
            plt.text(shift_label*0.95,top2*0.95,label2,#bbox=dict(boxstyle="round", fc="1.0"),
                verticalalignment='bottom', horizontalalignment='right')
        else:
            plt.text(shift_label*0.95,top2*0.95,label2,#bbox=dict(boxstyle="round", fc="1.0"),
                verticalalignment='top', horizontalalignment='right')
        plt.ylabel(ylabel)
        plt.gca().yaxis.set_label_coords(-0.065,1.)
        plt.xlabel(xlabel)
        # plt.tight_layout(hspace=0.001)
        plt.subplots_adjust(hspace=0.00)
    else:
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)

    if zoom is not None:
        if plot_type != 'single':
            plt.subplot(2,1,1)
            plt.xlim([zoom_shift-zoom[2], zoom_shift+zoom[2]])
            plt.subplot(2,1,2)
            plt.xlim([zoom_shift-zoom[2], zoom_shift+zoom[2]])
        else:
            plt.subplot(1,1,1)
            plt.xlim([zoom_shift-zoom[2], zoom_shift+zoom[2]])

    return plt