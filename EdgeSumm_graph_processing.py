# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 12:29:55 2018
@author: Wafaa Samy El-Kassas (wafaa.elkassas@gmail.com)
"""
from text_processing import pos_tagging_text, text_refine , get_sent_lemma,get_word_lemma, get_word_synonyms, get_abbreviation_text, cluster_texts, fix_tag
from text_processing import get_jaccard_sim,compute_words_frequency, replaceHyphenated, fix_hyphen, pos_tagging, remove_citation, replace_abbreviation
#------------------------------------------------------------------------------
import re
from nltk import word_tokenize
import networkx as nx     # nx can be seemed as an alias of networkx module
import matplotlib.pyplot as plt
from pprint import pprint
import pandas as pd
from numpy import median
from  math import ceil,floor,sqrt
import collections
import numpy as np
from nltk.corpus import stopwords
#------------------------------------------------------------------------------
def get_sentence_graph(G,tag,s,two_grams,three_grams):
    first_node=""
    second_node=""
    edge=""
    flag=0
    order=0
    tag = fix_tag(tag)
    sentence_nodes=[]
    flag_bracket= False
    for index in range(len(tag)):    #tag: POS tagging of a sentence
        if flag==0:
            gg=0                      
            if gg==0:
                if tag[index][1] in ["NN","NNS","NNP","NNPS"]:
                    #----------------------------------------------------------
                    if tag[index][0] not in sentence_nodes:
                        sentence_nodes.append(get_word_lemma(tag[index][0]))
                    #----------------------------------------------------------
                    if flag_bracket== False:
                        G.add_node(get_word_lemma(tag[index][0]),**{s:tag[index][0]})
                        if first_node=="" and edge=="":
                            first_node= get_word_lemma(tag[index][0])   
                        elif first_node!="" and edge!="":
                            second_node= get_word_lemma(tag[index][0]) 
                            G.add_edge(first_node,second_node,label=edge,sentence=s,order=order)
                            order=order+1
                            first_node = second_node
                            second_node=""
                            edge=""
                        elif first_node=="" and edge!="":
                            second_node= get_word_lemma(tag[index][0]) 
                            G.add_edge('s#',second_node,label=edge,sentence=s,order=order)
                            order=order+1
                            first_node = second_node
                            second_node=""
                            edge=""
                        elif first_node!="" and edge=="":
                            second_node= get_word_lemma(tag[index][0]) 
                            G.add_edge(first_node,second_node,label='',sentence=s,order=order)
                            order=order+1
                            first_node = second_node
                            second_node=""
                    else:
                        if edge =="":
                            edge = tag[index][0]
                        else:
                            edge = edge +" "+ tag[index][0]     
                elif tag[index][1] in [".",":"]:
                    if flag_bracket== False:
                        if first_node!="" and edge!="" and second_node=="":
                            G.add_edge(first_node,'e#',label=edge,sentence=s,order=order)            
                            order=order+1
                        if first_node!="" and edge=="" and second_node=="":
                            G.add_edge(first_node,'e#',label=edge,sentence=s,order=order)            
                            order=order+1 
                        #---------------------
                        if first_node=="" and edge!="" and second_node=="":
                            G.add_edge('s#','e#',label=edge,sentence=s,order=order)            
                            order=order+1 
                        #---------------------
                    else:
                        if edge =="":
                            edge = tag[index][0]
                        else:
                            edge = edge +" "+ tag[index][0]                                 
                if tag[index][1] in ["VB","VBZ","FW","WDT","VBP",":","*","CD","VBN", "JJR", "HYPH","RP","RBR" ,"VBG", "PDT", "VBD", "MD","RB","PRP","EX","TO","IN",",","CC", "WRB", "PRP","PRP$","DT","JJ"]:# and f4==0:
                    
                    if edge =="":
                        edge = tag[index][0]
                    else:
                        edge = edge +" "+ tag[index][0]                
        else:
            flag = flag-1
    return sentence_nodes
#------------------------------------------------------------------------------
def compute_nodes_weights(G,word_frequencies,title,keywords_list,two_grams,three_grams,title_weight,proper_noun_list,keyword_weight,proper_weight,bi_grams_weight):
    nodes = G.nodes()
    all_nodes={}
    distict_nodes=[]
    minimum_value=100000000
    min_1=""
    for node_name in nodes:
        if node_name in word_frequencies.keys():
            all_nodes[node_name]=word_frequencies[node_name]
        else:
            all_nodes[node_name]=0            
 
    graph_nodes_weights = {}
    title_updated1 = get_sent_lemma(title.lower())
    title_updated=word_tokenize(title_updated1)
    biased_list=[]
    for node_name in nodes:
        weight=0
        T=False
        if node_name in word_frequencies.keys() and len(node_name)>1:
            weight = word_frequencies[node_name] 
            for t in title_updated:
                if  t == node_name.lower():
                    T=True
                    weight=weight + title_weight 
                    if node_name not in distict_nodes:
                        distict_nodes.append(node_name)
        #------------------------------------------------------               
            for k in keywords_list:
                k_updated1 = get_sent_lemma(k.lower())
                k_updated=word_tokenize(k_updated1)
                for kk in k_updated:
                    if kk==node_name:
                        T=True
                        weight=weight + keyword_weight
                        if node_name not in distict_nodes:
                            distict_nodes.append(node_name)
        #-----------------------------------------------------                       
            for t in two_grams:
                t_updated1 = get_sent_lemma(t.lower())
                t_updated=word_tokenize(t_updated1)
                for tt in t_updated:
                    if tt.lower()==node_name.lower():
                        T=True
                        weight = weight + bi_grams_weight
                        if node_name not in distict_nodes:
                            distict_nodes.append(node_name)
        #--------------------------------------------------------           
            for t in proper_noun_list:
                t_updated = get_word_lemma(t)
                if t_updated.lower() ==node_name.lower():
                    T=True
                    weight = weight + proper_weight
                    if node_name not in distict_nodes:
                        distict_nodes.append(node_name)
        #----------------------------------------------------               
            for b in biased_list:
                b_updated = get_word_lemma(b.lower())
                if b_updated.find(node_name)>=0:
                    weight = weight + 1
        #----------------------------------------------------
        if len(node_name)<2:
            weight=0
        #----------------------------------------------------
        graph_nodes_weights[node_name] = weight
        if weight<minimum_value and T==True:
            minimum_value=weight
            min_1=node_name
    print("min value ", minimum_value,min_1)
    return [graph_nodes_weights,minimum_value,distict_nodes]
#------------------------------------------------------------------------------
#-----------------------------------traverse the graph-------------------------
def get_candidate_edge_list(G,graph_nodes_weights,two_grams,node_weight,threshold,trial,source_node_max):
    candidate_edge_list = []
    sorted_weights_list = sorted(graph_nodes_weights, key=(lambda key:graph_nodes_weights[key]), reverse=True) 
    numbers = sorted([graph_nodes_weights[key] for key in graph_nodes_weights], reverse=True)
    print("numbers",numbers)
    median_node_weight = median(numbers)
    print("median", median_node_weight)
    visited =[]
    two_grams_lemma =[]
    for t in two_grams:
        two_grams_lemma.append(get_sent_lemma(t.lower()))
    print("#######################################")
    print("sorted", sorted_weights_list)  
    print(G.edges.data())

    print("#######################################")   
    adjusted_node_weight = (median([num for num in numbers if num >= 1]))

    n_list=[num for num in numbers if num >= 1]
    n_avg=(sum(n_list)/len(n_list))
    n_sd=sqrt(np.var(n_list))
    #------------------------------
    n_min= n_avg
    if  adjusted_node_weight<=n_avg:
        n_min= n_avg
    else:
        n_min= adjusted_node_weight
    #------------------------------
    if source_node_max=="3sd":
        n_max=n_avg+(3*n_sd)
    elif source_node_max=="2sd":
        n_max=n_avg+(2*n_sd)
    elif source_node_max=="max":
        n_max=max(n_list)
    print("****&&&*****",n_avg,n_sd,n_max,adjusted_node_weight)
    print("adjusted_node_weight",adjusted_node_weight)
    
    for node in sorted_weights_list:
        if graph_nodes_weights[node] >= (n_min) and graph_nodes_weights[node] <= n_max:  
            if node not in visited:
                visited.append(node)                 
                out_edges=G.out_edges(node,data=True, keys=True)

                max_weight=0  
                weight_sum=0
                weights = []

                out_edge_visited=[]
                out_edges_total=0
                for out in out_edges:
                    if out[1] not in out_edge_visited and graph_nodes_weights[out[1]]>0:
                        out_edge_visited.append(out[1])
                        out_edges_total=out_edges_total+1
                        weights.append(graph_nodes_weights[out[1]])

                        weight_sum = weight_sum + graph_nodes_weights[out[1]]
                        if graph_nodes_weights[out[1]]>= max_weight:
                            max_weight = graph_nodes_weights[out[1]]
                
                if out_edges_total>=1:
                    average_weight = (weight_sum / out_edges_total)
                else:
                    average_weight = (weight_sum )
                
                for out in out_edges:
                    out_weight = graph_nodes_weights[out[1]]

                    threshold_value =0
                    if threshold=="max":
                        threshold_value = max_weight                        
                    if threshold=="avg":
                        threshold_value =    average_weight
                    if threshold=="normal":
                        threshold_value =    n_min 

                    if (out_weight >= threshold_value and out_weight > 0):
                        sent = G[out[0]][out[1]][out[2]]['sentence']
                        tup = (sent,int(sent[3:sent.find("-")]),int(sent[sent.find("sent")+4:]),G[out[0]][out[1]][out[2]]['order'],out[0] , G[out[0]][out[1]][out[2]]['label'] ,out[1])
                        candidate_edge_list.append(tup)

    print("visited",visited)
    return candidate_edge_list
#------------------------------------------------------------------------------
def  get_candidate_summary(G, candidate_edge_list,three_grams,sentence_in_section_dict,trial,edge_weight):
    edges=G.edges(data=True, keys=True)
    edge_data= []
    temp_={}
    for a in edges:
        temp_["label"] = G[a[0]][a[1]][a[2]]["label"]
        temp_["sentence"] =  G[a[0]][a[1]][a[2]]["sentence"]
        temp_["order"] =  G[a[0]][a[1]][a[2]]["order"]

        temp_["start"]=a[0]
        temp_["end"]=a[1]
        temp_["key"]=a[2]
        edge_data.append(temp_)
        temp_={}
    #    print("---------------------------------------------------------------")
    from operator import itemgetter
    candidate_edge_list.sort(key=itemgetter(1,2,3))
    summary_dict = {}
    output = ""    
    visited_sentences=[]
    print("candidate_edge_list",candidate_edge_list)
    if len(candidate_edge_list) > 0:
        min1 = min(candidate_edge_list, key=itemgetter(1))[1]
        max1 = max(candidate_edge_list, key=itemgetter(1))[1]
    else:
        min1=0
        max1=0
    # get average edge count
    edge_count_list={}
    edge_count_list1={}
    count_temp=0
    for index11 in range(min1,max1+1):       
        sentences = [t for t in candidate_edge_list if (t[1] == index11)]   
        for s11 in sentences:
            sent_list11 = [f for f in sentences if (f[0] == s11[0])]
            edge_count_list[s11[0]]=len(sent_list11)
            count_temp=count_temp+len(sent_list11) 
        sorted_edge_list = sorted([edge_count_list[key] for key in edge_count_list], reverse=True)
        adjusted_edge_count=median(sorted_edge_list)
        print("edge weight",edge_weight,count_temp/len(edge_count_list),edge_count_list, adjusted_edge_count,floor(adjusted_edge_count))                    
    #------------------------------------------------------------------------------
        jflag=False
        if adjusted_edge_count<edge_weight:
            edge_weight=adjusted_edge_count
            jflag==True
        if trial < edge_weight and jflag==False:
            edge_weight=trial 
    #------------------------------------------------------------------------------
    for index1 in range(min1,max1+1):
        sentences = [t for t in candidate_edge_list if (t[1] == index1)]        
        for sent in sentences:
            max_order=0
            if sent[0] not in visited_sentences:
                visited_sentences.append(sent[0])
                sent_list = [f for f in sentences if (f[0] == sent[0])]
                sent_edges = [g for g in edge_data if (G[g['start']][g['end']][g['key']]['sentence'] == sent[0])]
                if (len(sent_list) >= edge_weight) or ((edge_count_list[sent[0]]/len(sent_edges))>=0.2): 
                    max_order = max(sent_list, key=itemgetter(3))[3]
                    sent_edges = [g for g in edge_data if (G[g['start']][g['end']][g['key']]['sentence'] == sent[0])]
                    c=0                   
                    sent_edges = sorted(sent_edges, key=itemgetter('order'))
                    output = output + sentence_in_section_dict[sent[0]]
                    output = output.strip()
                    output = remove_citation(output)
                    output = replace_abbreviation(output)
                    output = output.strip()                    
                    summary_dict[sent[0]]=output                   
                    output=""
                    print("sentence ",sent[0], edge_count_list[sent[0]],len(sent_edges),edge_count_list[sent[0]]/len(sent_edges))
                    edge_count_list1[sent[0]]=edge_count_list[sent[0]]/len(sent_edges)                                    
    return [summary_dict,edge_count_list1]
#------------------------------------------------------------------------------
def get_summary(sent_dict,title,keywords_list,two_grams,three_grams, sentence_in_section_dict,trial,word_frequencies,out_file,title_weight,node_weight,edge_weight,threshold,max_words,keyword_weight,proper_weight,bi_grams_weight,source_node_max,output_file,system_no,step,step_bi,step_p,input_topics,topics_weights,syns):
    proper_noun_list=[]
    number_of_params=0
    sentence_nodes_list={}
    G=nx.MultiDiGraph()
    sent_list_for_pos=[]
    sent_code_list=[]
    for sent in sent_dict:       
        xsent = sent_dict[sent]
        xsent = remove_citation(xsent)
        xsent = replaceHyphenated(xsent)
        xsent = replace_abbreviation(xsent)
        sent_list_for_pos.append(xsent)
        sent_code_list.append(sent)
    tags = pos_tagging_text(sent_list_for_pos)
    indx=0
    for tag in tags:
        sent=sent_code_list[indx]
        indx=indx+1 
        sentence_nodes_list[sent]=get_sentence_graph(G,tag,sent,two_grams,three_grams) 
        for t11 in tag:
            if t11[1] in ["NNP","NNPS"]:
                if get_word_lemma(t11[0]).lower() not in proper_noun_list:
                    proper_noun_list.append(get_word_lemma(t11[0]).lower())
    print("proper_noun_list",proper_noun_list)
    #----compute additional weights--------------------------------------------
    words_list={}
    for n in G.nodes:
        if n in word_frequencies:
            if word_frequencies[n]>=1:
                words_list[n]=word_frequencies[n]
    ww=sorted(words_list.values(), reverse=True)
    word_sd=sqrt(np.var(ww))
    word_avg=sum(ww)/len(ww)
    print("words list------avg---median--",ww,sum(ww)/len(ww),median(ww),np.var(ww),word_sd)
    vcc=word_sd/word_avg
    #--------------------------------------------------------------------------
    vc=(abs(word_avg-median(ww)))
    #--------------------------------------------------------------------------
    number_of_params=1
    #--------------------------------------------------------------------------     
    title_weight=(1*vc)/number_of_params
    keyword_weight=(1*vc)/number_of_params
    proper_weight=(1*vc)/number_of_params
    bi_grams_weight=(1*vc)/number_of_params
    #--------------------------------------------------------------------------
    graph_nodes_w = compute_nodes_weights(G,word_frequencies,title,keywords_list,two_grams,three_grams,title_weight,proper_noun_list,keyword_weight,proper_weight,bi_grams_weight)  
    graph_nodes_weights =   graph_nodes_w[0]
    distinct_nodes=graph_nodes_w[2]
    print("distinct_nodes-------",distinct_nodes)
    #--------------------------------------------------------------------------
    candidate_edge_list=get_candidate_edge_list(G, graph_nodes_weights,two_grams,node_weight,threshold,trial,source_node_max) 
    from operator import itemgetter
    candidate_edge_list.sort(key=itemgetter(1,2,3))     
    summary_list = get_candidate_summary(G,candidate_edge_list,three_grams,sent_dict,trial,edge_weight)
    summary = summary_list[0]
    summary_edges = summary_list[1]    
    summary_new_edges=[]
#------------------------------------------------------------------------------    
#------------------ check summary length --------------------------------------
    flag = True
    summary_previous= {} 
    while flag == True:
        sum_count=0
        sentence_nodes_list_new={}
        for s_1 in summary:
            sum_count=sum_count+ len(summary[s_1].split())
        print("1------------",sum_count)       
        if  sum_count<(max_words):
            if summary_previous == {}:
                summary = sent_dict
                summary_edges = summary_list[1]
            flag = False
            break       
        if  sum_count>=(max_words) and sum_count<=(max_words+15): 
            flag = False
            break
        else:            
            trial=trial+1
            print("trial-------------",trial)
            G_new=nx.MultiDiGraph()
            sent_code_list1=[]
            tags1=[]     
            for sent in summary:       
                xsent = summary[sent]
                xsent = remove_citation(xsent)
                xsent = replaceHyphenated(xsent)
                xsent = replace_abbreviation(xsent)
                tags1.append(tags[sent_code_list.index(sent)])
                sent_code_list1.append(sent)
            indx1=0
            for tag in tags1:
                sent=sent_code_list1[indx1]
                indx1=indx1+1               
                sentence_nodes_list_new[sent]=get_sentence_graph(G_new,tag,sent,two_grams,three_grams)
            #------------------------------------------------------------------                      
            graph_nodes_ww = compute_nodes_weights(G_new,word_frequencies,title,keywords_list,two_grams,three_grams,title_weight,proper_noun_list,keyword_weight,proper_weight,bi_grams_weight)             
            graph_nodes_weights_new = graph_nodes_ww[0]             
            #--------------------------------
            nodes_list={}
            for n in G_new.nodes():
                if n in graph_nodes_weights_new:
                    if graph_nodes_weights_new[n]>=1:
                        nodes_list[n]=graph_nodes_weights_new[n]
            print("nodes_list_weights ", nodes_list,sorted(nodes_list.values(), reverse=True),sorted(nodes_list, key=nodes_list.__getitem__, reverse=True))
            bb=sorted(nodes_list.values(), reverse=True)
            node_sd=sqrt(np.var(bb))
            node_avg=sum(bb)/len(bb)
            print("nodes list------avg---median--",bb,sum(bb)/len(bb),median(bb),np.var(bb),sqrt(np.var(bb)))
            #------------------------------------------------------------------------------
            candidate_edge_list=get_candidate_edge_list(G_new, graph_nodes_weights_new,two_grams,node_weight,threshold,trial,source_node_max) 
            from operator import itemgetter
            candidate_edge_list.sort(key=itemgetter(1,2,3))   
            summary_list = get_candidate_summary(G_new,candidate_edge_list,three_grams,summary,trial,edge_weight)    
            summary_new = summary_list[0]
            summary_new_edges = summary_list[1]
            #------------------------------------------------------------------------------                
            sum_count_new=0
            for s_1_new in summary_new:
                sum_count_new=sum_count_new+len(summary_new[s_1_new].split())                   
            if sum_count_new <(max_words):
                flag = False               
                break
            if sum_count_new >= (max_words) and sum_count_new <= (max_words+15):
                flag = False
                summary = summary_new
                summary_edges = summary_new_edges
                graph_nodes_weights=graph_nodes_weights_new
                break
            elif sum_count_new == sum_count:
                summary = summary_new
                summary_edges = summary_new_edges
                graph_nodes_weights = graph_nodes_weights_new
                if threshold=="avg":
                    threshold="normal"
                elif threshold=="normal":
                    threshold="max"
                else:
                    break
            elif sum_count_new > (max_words+15):
                summary = summary_new
                summary_edges = summary_new_edges
                graph_nodes_weights = graph_nodes_weights_new
    #--------------------------------------------------------------------------
    sent_weights_new={}
    sent_weights_final={}
    weight_maxx=1
    for sent in summary:
        s_nodes=sentence_nodes_list[sent]
        sent_weights_new[sent]=0
        for t in s_nodes:
            sent_weights_new[sent]=sent_weights_new[sent]+graph_nodes_weights[t]
        
        if len(s_nodes)>0:
            l_len=len(s_nodes)
        else:
            l_len=1
        
        sent_weights_final[sent]=  sent_weights_new[sent]/l_len 
        if sent_weights_final[sent]>weight_maxx:
            weight_maxx=sent_weights_final[sent]

    sent_list1= sorted(sent_weights_final,key=sent_weights_final.__getitem__,reverse=True)  #summary
    print("------weights-------",sent_weights_final,sent_weights_new)
    sent_list2= [t for t in summary]
   #------------------------------------------------------------------------------------------------------------
    sent_order={}
    y=len(sent_list2)
    for s in sent_list2:
        sent_order[s]=y/len(sent_list2)
        y=y-1
   #---------------------------------------------------------------------------
    summary_order1 = extract_summary(sent_list2,summary,max_words,graph_nodes_weights,sentence_in_section_dict)
    summary_order=summary_order1[0]
    write_summary(summary_order1[1],output_file,"order",system_no)

    summary_weight1 = extract_summary(sent_list1,summary,max_words,graph_nodes_weights,sentence_in_section_dict)
    summary_weight = summary_weight1 [0]
    write_summary(summary_weight1[1],output_file,"weight",system_no)  
    #-----------------------------------------------------------------------------
    c_count=5
    generate_cluster_summary(sent_list1,c_count,summary,max_words,sentence_in_section_dict,output_file,system_no,"cluster000weight")
    #------------------------------------------------------------------------------
    sent_list110={}
    for r in summary:
        sent_list110[r]=sent_order[r]
        print("------order only------",r,sent_list110[r]) 
    sent_list111= sorted(sent_list110,key=sent_list110.__getitem__,reverse=True)
    generate_cluster_summary(sent_list111,c_count,summary,max_words,sentence_in_section_dict,output_file,system_no,"cluster000order")
    #------------------------------------------------------------------------------
    return summary
#------------------------------------------------------------------------------
def extract_summary(sent_list,summary,max_words,graph_nodes_weights,sentence_in_section_dict):
    sum_count1=0
    final_summary={}
    summary1={}
    summary2={}
    sum_count_old=0
    for s in sent_list:
        sum_count1=sum_count1+len(summary[s].split())
        print("-----------------------final count --------",s,len(summary[s].split()),sum_count1,sum_count_old)
        
        if sum_count1 < max_words:
            final_summary[s] = summary[s]
            sum_count_old = sum_count1
        elif sum_count1 >= (max_words) and sum_count1 <= (max_words+15):#20):
            final_summary[s] = summary[s]
            sum_count_old = sum_count1
            break
        elif sum_count1 > (max_words+15):
            sum_count1 = sum_count_old
                                    
    summarylist = collections.OrderedDict(sorted(final_summary.items(), key=lambda t: int(t[0][t[0].find("-sent")+5:])))
    print(summarylist)

    for r in summarylist:
        summary1[r]=summary[r]
        summary2[r]=sentence_in_section_dict[r]
    return [summary1,summary2]
#------------------------------------------------------------------------------
def write_summary(summary,output_file,test_name,test_trial):
    output_summary=""      
    file1= open("D:/summarization_system/html/"+"single"+test_name+str(test_trial)+"_"+output_file+".html","w")     
#--------------------------------------
    file1.write("<html>\n")
    file1.write("<head>\n<title>"+"single"+test_name+str(test_trial)+"_"+output_file+"</title>\n</head>\n")
    file1.write('<body bgcolor="white">\n')
#-------------------------------------
    index1=1
    for s in summary:
        s1 = summary[s] 
        file1.write('<a name="'+str(index1)+'">['+str(index1)+']</a> <a href="#'+str(index1)+'" id='+str(index1)+'>'+(fix_hyphen(s1)).strip()+'</a>')
        index1=index1+1
        file1.write("\n")
        output_summary = output_summary + " " + fix_hyphen(s1)
        output_summary = output_summary.strip()                
#-------------------------------------
    file1.write("</body>\n")
    file1.write("</html>\n")    
#-------------------------------------
    print("summary "+test_name+str(test_trial),len(summary),"--",output_summary) 
    print("----------------------------------------------------------")
    file1.close()
#======================================
    file1= open("D:/summarization_system/txt/"+output_file+"_"+"single"+test_name+str(test_trial)+".txt","w")                   
    for s in summary:
        s1 = summary[s]
        file1.write((fix_hyphen(str(s1))).strip())
        file1.write("\n")
    print("----------------------------------------------------------")
    file1.close()
#--------------------------------------------------------
def generate_cluster_summary(sent_list,c_count,summary,max_words,sentence_in_section_dict,output_file,system_no,system_name):
    cluster_texts1=[]
    cluster_sents1=[]
    cluster_summary={}
    for s in sent_list:
        cluster_texts1.append(summary[s])
        cluster_sents1.append(s)
    if len(cluster_texts1)<c_count:
        ttt=len(cluster_texts1)
    else:
        ttt=c_count
    clusters = cluster_texts(cluster_texts1, ttt)
    pprint(dict(clusters))
    sum_count1=0
    f_visited=[]
    while len(f_visited)< len(summary) and sum_count1 <max_words:
        for c in dict(clusters): 
            print("visited------",f_visited)
            if sum_count1 >=max_words:
                break
            for h in clusters[c]:
                print("sent",h)
                if cluster_sents1[h] not in f_visited:
                    temp_count1=sum_count1+len(summary[cluster_sents1[h]].split())
                    if temp_count1 <max_words:
                        print("-------------------",h,cluster_sents1[h],"-----------------")
                        sum_count1=temp_count1
                        cluster_summary[cluster_sents1[h]] = sentence_in_section_dict[cluster_sents1[h]]
                        f_visited.append(cluster_sents1[h])
                        break
                    if temp_count1 >=max_words and temp_count1 <=(max_words+15):
                        print("-------------------",h,cluster_sents1[h],"-----------------")
                        sum_count1=temp_count1
                        cluster_summary[cluster_sents1[h]] = sentence_in_section_dict[cluster_sents1[h]]
                        f_visited.append(cluster_sents1[h])
                        break
                    if temp_count1 > (max_words+15):
                        f_visited.append(cluster_sents1[h])
          
    cluster_summary1 = collections.OrderedDict(sorted(cluster_summary.items(), key=lambda t: int(t[0][t[0].find("-sent")+5:])))
    write_summary(cluster_summary1,output_file,system_name,system_no)