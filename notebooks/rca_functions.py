#import functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
import matplotlib
from scipy.stats import hypergeom
import seaborn as sns
from matplotlib_venn import venn2
from matplotlib_venn import venn3
import ndex2
import networkx as nx
from netcoloc import netprop_zscore
from netcoloc import netprop
from netcoloc import network_colocalization
import re
from scipy.stats import norm
import math
from tqdm import tqdm


#variables
#pcnet v1.4- used for prior analysis
#pcnet v1.3- used for rat bmi network paper
#pcnet v2.0- not currently available to the public
UUIDs= {
    'pcnet_v14':'c3554b4e-8c81-11ed-a157-005056ae23aa',
    'pcnet_v13':'4de852d9-9908-11e9-bcaf-0ac135e8bacf',
    'string':'98ba6a19-586e-11e7-8f50-0ac135e8bacf',
    'humanNet_v3_FN': '40913318-3a9c-11ed-ac45-0ac135e8bacf',
    'ACN':'29b2d215-07fd-11ef-9621-005056ae23aa',
	'ACN_unannot':'f81a3f67-4215-11ee-aa50-005056ae23aa',
	'ACN_strin':'48de252c-3d50-11ee-aa50-005056ae23aa',
}
UUID_tag={
    'pcnet_v14':'',
    'pcnet_v13':'_pcnet_v13',
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
	'Burden':'#fcba03',
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
    print(tblr_lim)
    #combine zscore tables
    tbl_z=combine_nps_table(tblr, tblc)
    #subset table to those within network limit parameters
    inNetwork=tbl_z[(tbl_z['NPSr']>tblr_lim) & (tbl_z['NPSc']>tblc_lim) & (tbl_z['NPScr']>comb_lim)]
    print(len(inNetwork))
    #plot venn diagram
    Nr=(len(tbl_z[tbl_z['NPSr']>tblr_lim])-len(inNetwork))
    Nc=(len(tbl_z[tbl_z['NPSc']>tblc_lim])-len(inNetwork))
    Nboth=len(inNetwork)
    venn2((Nr,Nc,Nboth), 
		  set_labels=(tblr_label, tblc_label),
      set_colors=(color_dict['rare'], color_dict['common']), alpha = 0.7)
    plt.title('p='+str(p_net_overlap)+ ', single cut='+str(tblr_lim)+', comb cut='+str(comb_lim))
    if savefig:
        plt.savefig('figures/network_venn/network_venn_'+tblr_label+'_'+tblc_label+'.svg',bbox_inches='tight')
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
        plt.savefig('figures/seed_venn/seed_venn_'+tblr_label+'_'+tblc_label+'.svg',bbox_inches='tight')
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
        plt.savefig('figures/histogram/histogram_'+tblr_label+'_'+tblc_label+'.svg',bbox_inches='tight')
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

def run_net_prop(path, trait_name, pcol, gene_col, delim, cutoff=None, graph=None, w_double_prime=None, interactome='pcnet_v14', ndex_user=None, ndex_password=None, savefile=False):
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
	NPS zscores
 
    Notes:
    - The function requires an external library for network propagation calculations.
    - The seed genes file should contain a column for genes and a column for their associated p-values.
    - The function saves three files: z-scores, raw heats, and randomized heats for the network analysis,
      with the trait name and optionally the interactome name as part of the filenames.
    - If using a private interactome, ensure the ndex_user and ndex_password are correctly provided.
    """
    data = import_seedgenes(path, pcol, gene_col, delim)
    data = list(data[gene_col])
    if graph is None:
        graph = import_interactome(interactome)
        print("importing network " + interactome)
    if w_double_prime is None:
        # pre calculate mats used for netprop
        print('\ncalculating w_prime')
        w_prime = netprop.get_normalized_adjacency_matrix(graph, conserve_heat=True) 
        print('\ncalculating w_double_prime')
        w_double_prime = netprop.get_individual_heats_matrix(w_prime, 0.5)
    else:
        print("using provided w_double_prime - please ensure that w_double_prime aligns to graph provided")
    graph_nodes = list(graph.nodes())
    #print(graph_nodes)
    data = list(set(data).intersection(graph_nodes))
    #print(data)
    ##calculate heats
    z_score, Fnew_score, Fnew_rand_score = netprop_zscore.calculate_heat_zscores(
        w_double_prime,  
        graph_nodes,
        dict(graph.degree), 
        data, num_reps=1000,
        minimum_bin_size=100
    )
    if savefile:
        export_path = 'calculated_values/network_scores/'
        if graph is None and interactome == 'pcnet_v14':
            prefix = (export_path + trait_name).lower()
        elif graph is None and interactome != 'pcnet_v14':
            prefix = (export_path + trait_name + '_' + interactome).lower()
        elif graph is not None and interactome != 'pcnet_v14':
            prefix = (export_path + trait_name + '_' + interactome).lower()
        else:
            print("saving file without interactome_prefix, please provide an interactome name if prefix wanted")
            prefix = ('network_scores/' + trait_name).lower()

        z_score.to_csv(prefix + '_zscore.tsv', sep='\t', header=False)
        if saveheat:
            Fnew_score.to_csv(prefix + '_heats.tsv', sep='\t', header=False)
            pd.DataFrame(Fnew_rand_score, columns=z_score.index).to_csv((prefix+'_randheats.tsv'),sep='\t')
        else:
            print('calculated NPS not saved')
    return z_score


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

def plot_permutation_histogram(permuted, observed, title="", xlabel="Observed vs Permuted", color='#CCCCCC', arrow_color="blue",save_fig=False, filename=None):
    """Plot an observed value against a distribution of permuted values. Adapted from BMI

    Args:
        permuted (list): A list of permuted values that form the distribution
        observed (float): The observed value of interest
        title (str): Plot title. Defaults to "".
        xlabel (str): The x axis title. Defaults to "Observed vs Permuted".
        color (str, optional): The color of the histogram. Defaults to "cornflowerblue".
        arrow_color (str, optional): The color of the arrow pointing to observed value. Defaults to "red".
    """
    plt.figure(figsize=(5, 4))
    dfig = sns.histplot(permuted, label='Permuted', alpha=0.4, stat='density', bins=25, kde=True, 
                        edgecolor='w', color=color)
    params = {'mathtext.default': 'regular'}          
    plt.rcParams.update(params)
    plt.xlabel(xlabel, fontsize=16)
    diff = max(observed, max(permuted))-min(permuted)
    plt.arrow(x=observed, y=dfig.dataLim.bounds[3]/2, dx=0, dy=-1 * dfig.dataLim.bounds[3]/2, label="Observed",
              width=diff/100, head_width=diff/15, head_length=dfig.dataLim.bounds[3]/20, overhang=0.5, 
              length_includes_head=True, color=arrow_color, zorder=50)
    plt.ylabel("Density", fontsize=16)
    plt.legend(fontsize=12, loc=(0.6,0.75))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.locator_params(axis="y", nbins=6)
    plt.title(title + " (p=" + str(get_p_from_permutation_results(observed, permuted)) + ")", fontsize=16)
    
    if save_fig:
        plt.savefig('figures/' + filename + '.svg', bbox_inches='tight')


## Extensions to NetColoc ------------------------------------------------------------------------------------
def get_p_from_permutation_results(observed, permuted):
    """Calculates the significance of the observed mean relative to the empirical normal distribution of permuted means.

    Args:
        observed (float): The observed value to be tested
        permuted (list): List of values that make up the expected distribution
    
    Returns:
        float: p-value from z-test of observed value versus the permuted distribution
    """
    p = norm.sf((observed - np.mean(permuted)) / np.std(permuted))
    try:
        p = round(p, 4 - int(math.floor(math.log10(abs(p)))) - 1)
    except ValueError:
        print("Cannot round result, p=", p)
    return p

def calculate_mean_z_score_distribution(z1, z2, num_reps=1000, zero_double_negatives=True, 
                                        overlap_control="remove", seed1=[], seed2=[]):
    """Determines size of expected mean combined `z=z1*z2` by randomly shuffling gene names

    Args:
        z1 (pd.Series, pd.DataFrame): Vector of z-scores from network propagation of trait 1
        z2 (pd.Series, pd.DataFrame): Vector of z-scores from network propagation of trait 2
        num_reps (int): Number of perumation analyses to perform. Defaults to 1000
        zero_double_negatives (bool, optional): Should genes that have a negative score in both `z1` and `z2` be ignored? Defaults to True.
        overlap_control (str, optional): 'bin' to permute overlapping seed genes separately, 'remove' to not consider overlapping seed genes. Any other value will do nothing. Defaults to "remove".
        seed1 (list, optional): List of seed genes used to generate `z1`. Required if `overlap_control!=None`. Defaults to [].
        seed2 (list, optional): List of seed genes used to generate `z2`. Required if `overlap_control!=None`. Defaults to [].

    Returns:
        float: The observed mean combined z-score from network colocalization
        list: List of permuted mean combined z-scores
    """
    if isinstance(z1, pd.Series):
        z1 = pd.DataFrame(z1, columns=["z"])
    if isinstance(z2, pd.Series):
        z2 = pd.DataFrame(z2, columns=["z"])
    z1z2 = z1.join(z2, lsuffix="1", rsuffix="2")
    z1z2 = z1z2.assign(zz=z1z2.z1 * z1z2.z2)
    #print(z1z2.head())
    if overlap_control == "remove":
        seed_overlap = list(set(seed1).intersection(set(seed2)))
        print("Overlap seed genes:", len(seed_overlap))
        z1z2.drop(seed_overlap, axis=0, inplace=True)
    elif overlap_control == "bin":
        seed_overlap = list(set(seed1).intersection(set(seed2)))
        print("Overlap seed genes:", len(seed_overlap))
        overlap_z1z2 = z1z2.loc[seed_overlap]
        overlap_z1 = np.array(overlap_z1z2.z1)
        z1z2.drop(seed_overlap, axis=0, inplace=True)
    z1 = np.array(z1z2.z1)
    z2 = np.array(z1z2.z2)
    if zero_double_negatives:
        for node in z1z2.index:
            if (z1z2.loc[node].z1 < 0 and z1z2.loc[node].z2 < 0):
                z1z2.loc[node, 'zz'] = 0
    permutation_means = np.zeros(num_reps)
    for i in tqdm(range(num_reps)):
        perm_z1z2 = np.zeros(len(z1))
        np.random.shuffle(z1)

        for node in range(len(z1)):
            if not zero_double_negatives or not (z1[node] < 0 and z2[node] < 0):
                perm_z1z2[node] = z1[node] * z2[node]
            else:
                perm_z1z2[node] = 0
        if overlap_control == "bin":
            overlap_perm_z1z2 = np.zeros(len(overlap_z1))
            np.random.shuffle(overlap_z1) 
            for node in range(len(overlap_z1)):
                if zero_double_negatives and (overlap_z1[node] < 0 and z2[node] < 0):
                    overlap_perm_z1z2[node] = 0
                else:
                    overlap_perm_z1z2[node] = overlap_z1[node] * z2[node]
            perm_z1z2 = np.concatenate([perm_z1z2, overlap_perm_z1z2])
                    
        permutation_means[i] = np.mean(perm_z1z2)
    return np.mean(z1z2.zz), permutation_means

def format_catalog(catalog=None):
	try:
		#make all annotations lowercase for consistency for querying
		catalog['MAPPED_TRAIT']=catalog['MAPPED_TRAIT'].str.lower()
		catalog['DISEASE/TRAIT']=catalog['DISEASE/TRAIT'].str.lower()
		#filter for genes that were mapped
		mapped=catalog[~catalog['MAPPED_GENE'].isna()]
		mapped=mapped[~mapped['MAPPED_TRAIT'].isna()]
		mapped=mapped[['MAPPED_GENE','MAPPED_TRAIT','DISEASE/TRAIT','PUBMEDID']]
		mapped.columns=['GENE','MAPPED_TRAIT','DISEASE/TRAIT','PUBMEDID']
		#filter for genes that were reported
		rep=catalog[~catalog['REPORTED GENE(S)'].isna()]
		rep=rep[~rep['MAPPED_TRAIT'].isna()]
		rep=rep[~rep['REPORTED GENE(S)'].str.contains('Intergenic')]
		rep=rep[['REPORTED GENE(S)','MAPPED_TRAIT','DISEASE/TRAIT','PUBMEDID']]
		rep.columns=['GENE','MAPPED_TRAIT','DISEASE/TRAIT','PUBMEDID']
		cat=pd.concat([rep, mapped])
		cat['GENE']=cat['GENE'].str.split('; ')
		cat=cat.explode('GENE')
		cat=cat[~(cat['GENE'].str.contains('mapped'))]
		cat['GENE']=cat['GENE'].str.split(', ')
		cat=cat.explode('GENE')
		cat['GENE']=cat['GENE'].str.split(' - ')
		cat=cat.explode('GENE')
		cat['GENE']=cat['GENE'].astype('str')
		cat=cat[~(cat['GENE'].str.contains('intergenic'))]
		cat['TRAIT']=cat['MAPPED_TRAIT'] + ": " +cat['DISEASE/TRAIT']+ " (PMID: "+(cat['PUBMEDID'].astype(str))+")"
		cat=cat.dropna()
		return(cat)
	except:
		print('please add gwas catalog file.')
def subset_catalog(cat=None, trait_group=None):
	if (trait_group=='alcohol'):
		cat=cat[(cat['TRAIT'].str.contains('alcohol'))
			|(cat['TRAIT'].str.contains('drinking'))
		   |(cat['TRAIT'].str.contains('wine'))
		   |(cat['TRAIT'].str.contains('liquor'))
		   |(cat['TRAIT'].str.contains('beer'))]
		#include every measure- could also remove add wine, beer, liquor-doesnt change number of genes
		cat=cat[~(cat['TRAIT'].str.contains('nonalcohol'))]
		cat=cat[~(cat['TRAIT'].str.contains('non-alcohol'))]
	elif(trait_group=='nicotine'):
		cat=cat[(cat['TRAIT'].str.contains('smok'))&(~cat['TRAIT'].str.contains('taste'))]
	elif(trait_group=='SUD'):
		cat=cat[
		(cat['TRAIT'].str.contains('substance')
		|cat['TRAIT'].str.contains('addiction')
		|cat['TRAIT'].str.contains('cocaine')
		|cat['TRAIT'].str.contains('opioid')
		|cat['TRAIT'].str.contains('cannabis')
		|cat['TRAIT'].str.contains('hallucinogen')
		|cat['TRAIT'].str.contains('abuse')
		|cat['TRAIT'].str.contains('dependence'))
		&
		((~cat['TRAIT'].str.contains('externalizing'))
		&(~cat['TRAIT'].str.contains('food'))
		&(~cat['TRAIT'].str.contains('internet'))
		&(~cat['TRAIT'].str.contains('taste'))
		&(~cat['TRAIT'].str.contains('eating'))
		&(~cat['TRAIT'].str.contains('nicotine'))
		&(~cat['TRAIT'].str.contains('alcohol'))
		&(~cat['TRAIT'].str.contains('response to opioid')))
		]
	elif(trait_group=='neuropsych'):
		alc_cat=subset_catalog(cat, 'alcohol')['TRAIT'].to_list()
		smok_cat=subset_catalog(cat, 'nicotine')['TRAIT'].to_list()
		SUD_cat=subset_catalog(cat, 'SUD')['TRAIT'].to_list()
		SUD_inclusive=set(alc_cat+smok_cat+SUD_cat)
		behav_ls=set(cat[
				((cat['TRAIT'].str.contains('depre'))|
				    (cat['TRAIT'].str.contains('neurotic'))|
				    (cat['TRAIT'].str.contains('cogn'))|
				    (cat['TRAIT'].str.contains('behav'))|
				    (cat['TRAIT'].str.contains('anorex'))|
				    ((cat['TRAIT'].str.contains('mani'))&(~cat['TRAIT'].str.contains('holdemania')))|
				    (cat['TRAIT'].str.contains('parkinson'))|
				    (cat['TRAIT'].str.contains('alzheim'))|
				    (cat['TRAIT'].str.contains('feeling'))|
				    (cat['TRAIT'].str.contains('language'))|
				    (cat['TRAIT'].str.contains('schizo'))|
				    (cat['TRAIT'].str.contains('risk'))|
				    (cat['TRAIT'].str.contains('demen'))|
				    (cat['TRAIT'].str.contains('autis'))|
				    (cat['MAPPED_TRAIT'].str.contains('risk'))|
				    (cat['TRAIT'].str.contains('anxiet'))|
				    (cat['TRAIT'].str.contains('memory')&(~cat['TRAIT'].str.contains('cell')))|
				    (cat['TRAIT'].str.contains('externalizing')))
				             &
				    (~cat['TRAIT'].str.contains('cancer')
				     & ~cat['TRAIT'].str.contains('protein')
				     & ~cat['TRAIT'].str.contains('blood pressure')
				     & ~cat['TRAIT'].str.contains('renal')
				     & ~cat['TRAIT'].str.contains('diabete')
				     & ~cat['TRAIT'].str.contains('allergen')
				     & ~cat['TRAIT'].str.contains('radiation')
				     & ~cat['TRAIT'].str.contains('cardio')
				     & ~cat['TRAIT'].str.contains('visceral')
				    )]['TRAIT'])
		behav_ls=behav_ls.difference(set(SUD_inclusive))
		cat=cat[cat['TRAIT'].isin(behav_ls)]
	else:
		print('trait not in list. please provide trait group that matches criteria')
	return(cat)

def format_subset_cat(cat):
	cat=cat.groupby('GENE').agg(tuple).applymap(set).reset_index()
	cat=cat[['GENE','TRAIT']]
	return(cat)
def f_test(group1, group2):
   f = np.var(group1, ddof=1)/np.var(group2, ddof=1)
   nun = np.array(group1).size-1
   dun = np.array(group2).size-1
   p_value = 1-stats.f.cdf(f, nun, dun)
   return f, p_value

def combine_nps_table(tblr, tblc):
    tbl_z=pd.concat([tblr, tblc], axis=1)
    tbl_z.columns=('z1','z2')
    tbl_z['z_comb']=tbl_z['z1']*tbl_z['z2']
    tbl_z.columns=['NPSr','NPSc','NPScr']
    return(tbl_z)

def venn_rare_test(t1, t2, t3, labels,colors):
    only_t1 = len(t1 - t2 - t3)
    only_t2 = len(t2 - t1 - t3)
    only_t3 = len(t3 - t1 - t2)
    
    only_t1_t2 = len(t1 & t2 - t3)
    only_t1_t3 = len(t1 & t3 - t2)
    only_t2_t3 = len(t2 & t3 - t1)

    t1_t2_t3 = len(t1 & t2 & t3)
    venn3(subsets=(only_t1, only_t2, only_t1_t2, only_t3, only_t1_t3, only_t2_t3, t1_t2_t3), set_labels=labels,set_colors=colors,alpha=.6)
    plt.show()
	
def NPS_lineplot(df,metric, filename, xrange=None, yrange=None, save_fig=False, sigline=False):
    matplotlib.rcParams.update({'font.size': 8})
    df['-log10(p)']=-np.log10(df['empirical_p'])

    # Group data by NPS_single and plot each group separately
    groups = df.groupby('NPS_single')
    
    # Initialize a plot
    fig, ax = plt.subplots(figsize=(2.75, 2))
    
    # Plot lines for each NPS_single group
    for name, group in groups:
        ax.plot(group['NPS_common-rare'], group[metric], marker='o', label=f'NPS_single={name}')
    if sigline:
	    ax.axhline(y =-np.log10(0.05/len(df)), color = 'red', linestyle = 'dashed', linewidth=1)
    if ~(yrange is None):
        ax.set_ylim(yrange)
    if ~(xrange is None):
        ax.set_xlim(xrange)    
    # Set plot labels
    ax.set_xlabel('NPS combined') 
    ax.set_ylabel(metric.replace('_',' '))
    ax.legend(title='NPS single')
    ax.grid(True)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if save_fig:
        plt.savefig('figures/'+filename,bbox_inches='tight')
    plt.show()