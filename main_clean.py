from __future__ import print_function
from datetime import datetime as dt
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re, math, timeit
import numpy as np
from data_handler import Data
from tabulate import tabulate
#import time
#==========================================================================================================================#
class TextMining:
    def __init__(self, data):
        self.data = data
        docs = self.data.get_documents()
        self.expected_duration = 0.65 * len(docs)
    
    def start(self, min_count, min_occur, segment_range_ms):
        st = timeit.default_timer()
        documents = self.data.get_documents()
        self.data.set_result('document_count', len(documents))
#        preprocessed = self.preprocess(documents)
        preprocessed = self.data.get_processed_docs()
        filtered = self.filtering(preprocessed, min_count, min_occur)
        self.segment_end = self.segmenting(segment_range_ms)
        start = 0
        for index, end in enumerate(self.segment_end):
            w = self.weighting(filtered[start:end], index)
            self.similarity(w, index)
            print("\n[{}] Segment {} done ({})".format(dt.now(), str(index), str(end-start)))
            start = end + 1
        
        self.data.set_result('text_process_time', timeit.default_timer()-st)
        self.data.set_result('text_process_last_run', dt.now().strftime("%A, %d %B %Y %I:%M:%S %p"))    
    
    def preprocess(self, documents):
        print("[{}] Preprocessing...".format(dt.now()))
        stemmer = StemmerFactory().create_stemmer()
        stopwords = self.data.get_stopword()    
        formal_dict = self.data.get_formalization()  
        formal_pattern = re.compile(r'\b(' + '|'.join(formal_dict.keys()) + r')\b')  
        url_pattern = re.compile(r'((http[s]?|ftp):\/)?\/?([^:\/\s]+)((\/\w+)*\/)([\w\-\.]+[^#?\s]+)(.*)?(#[\w\-]+)?')
        digit_symbol_pattern = re.compile(r'\d+|[^\w\s]')
        user_handler_pattern = re.compile(r'@\w+')
        processed_docs = []
        print("  ",end='')
        runtime = []
        for doc in documents:
            st = timeit.default_timer()
            new_doc = re.sub(url_pattern, "", doc)
            new_doc = re.sub(user_handler_pattern, "", new_doc)
            new_doc = re.sub(digit_symbol_pattern, " ", new_doc)
            new_doc = new_doc.lower()
            new_doc = formal_pattern.sub(lambda x: formal_dict[x.group()], new_doc)
            new_doc = stemmer.stem(new_doc)
            new_doc = new_doc.split()
            new_doc = [word for word in new_doc if word not in stopwords]
            
            print('.', end='')
            processed_docs.append(new_doc)
            runtime.append(timeit.default_timer()-st)
        self.data.save_processed_docs(processed_docs)
        print("\nAverage Preprocessing time : "+str(sum(runtime)/len(runtime)))            
        return processed_docs
        
    def filtering(self, processed_docs, MIN_COUNT, MIN_OCCUR):
        print("\n[{}] Filtering...".format(dt.now()))
        result = [[terms, len(terms) > MIN_COUNT] for terms in processed_docs]
        filtered_docs = [data[0] for data in result if data[1]]
        self.data.set_result('document_count_after_MIN_COUNT', len(filtered_docs))
        
        bag_of_word = list(set([term for docs in filtered_docs for term in docs]))     
        df_dict = {}
        for term in bag_of_word:
            df_dict[term] = 0
            for doc in filtered_docs:
                if term in doc:
                    df_dict[term] += 1 
         
        for idx, data in enumerate(result):
            if data[1]:
                result[idx][1] = False
                for term in data[0]:
                    if df_dict[term] >= MIN_OCCUR:
                        result[idx][1] = True
                        break                  
        filtered_docs = [data[0] for data in result if data[1]]
        self.data.set_result('document_count_after_MIN_OCCUR', len(filtered_docs)) 
        self.data.save_filtered_docs(result)
        
        return filtered_docs
    
    def segmenting(self, segment_range):
        print("[{}] Segmenting...".format(dt.now()))
        docs_timestamp = self.data.get_timestamp(filtered=True)
        lower_time = docs_timestamp[0]
        upper_time = lower_time + segment_range
        end_of_groups = []
        for idx, time in enumerate(docs_timestamp):
            if time > upper_time:
                end_of_groups.append(idx)
                upper_time += segment_range
        
        end_of_groups.append(len(docs_timestamp))
        self.data.set_result('segment_count', len(end_of_groups))
        self.data.set_result('end_of_segments', end_of_groups)
        
        return end_of_groups
                            
    def weighting(self, processed_docs, index):
        print("[{}] Weighting...".format(dt.now()))
        bag_of_word = list(set([term for docs in processed_docs for term in docs]))
        unique_word_count = len(bag_of_word)
        print("  Calculating term frequency...")
        td_freq = np.empty((unique_word_count, len(processed_docs)), dtype=int)
        for i in range(unique_word_count):
            for j in range(len(processed_docs)):
                td_freq[i,j] = processed_docs[j].count(bag_of_word[i])
        
        print("  Calculating document frequency...")
        d_freq = np.zeros(unique_word_count, dtype=int)
        for i in range(unique_word_count):
            for j in range(len(processed_docs)):
                if bag_of_word[i] in processed_docs[j]:
                    d_freq[i] += 1
            
        print("  Calculating TF-IDF...")
        document_weight = np.empty(td_freq.shape)
        for i in range(unique_word_count):
            for j in range(len(processed_docs)):
                document_weight[i,j] = td_freq[i,j]*math.log10(unique_word_count/d_freq[i])
#        self.data.save_td_frequency(bag_of_word, td_freq, d_freq)
        self.data.save_weighting(document_weight, index)
        return document_weight
    
    def similarity(self, document_weight, index):                
        print("[{}] Calculating similarity...".format(dt.now()))
        doc_count = len(document_weight[0])
        similarity = np.empty((doc_count,doc_count))
        magnitude = np.sqrt(np.sum(np.square(document_weight), axis=0))
        for i in range(doc_count):
#            print('.', end='')
            for j in range(i, doc_count):
                dot_product = np.dot(document_weight[:,i], document_weight[:,j])
                similarity[i,j] = dot_product/(magnitude[i]*magnitude[j])
                similarity[j,i] = similarity[i,j]
          
        self.data.save_similarity(similarity, index)
        return similarity         
        
#==========================================================================================================================#
class AffinityPropagation:
    def __init__(self, data):
        self.data = data
        
    def init(self, similarity, pref, damping_f, changed_limit, max_iter):
#        print("Initializing parameter and variable...")
        self.size = len(similarity)
        self.pref = pref
        self.damping_f = damping_f
        self.changed_limit = changed_limit
        self.max_iter = max_iter
        self.availability = np.zeros((self.size, self.size))
        self.responsiblity = np.zeros((self.size, self.size))
        self.similarity = similarity
        for i in range(self.size):
            self.similarity[i,i] = self.pref
                       
    def fit(self, similarity, pref, damping_f, changed_limit, max_iter, index):
        self.iteration = 0
        self.changed = 1
        self.clusters = {}
        self.init(similarity, pref, damping_f, changed_limit, max_iter)
        is_continue = True
        while is_continue:
#            print('.', end='')
            self.update_responsibility_v2()
            self.update_availability_v2()
            new_clusters = self.generate_clusters()
#            if self.iteration == 0:
#                self.data.save_as_csv(self.responsiblity, 'responsiblity iter 0.csv', index)
#                self.data.save_as_csv(self.availability, 'availability iter 0.csv', index)
#                print(new_clusters)
            is_continue = self.is_continue(new_clusters)
#        print("\nFinal iteration : "+str(self.iteration))   
        self.data.set_result('iteration_of_segments', self.iteration, append=True)
        self.data.set_result('cluster_of_segments', len(self.clusters), append=True)
        self.data.set_result('~clusters', self.clusters, append=True)
                
        return self.clusters
            
    def update_responsibility(self):
        print("Calculating responsibility...")
        for i in range(self.size):
            for k in range(self.size):
                sum_as = []
                for kx in range(self.size):
                    if kx != k:
                        sum_as.append(self.availability[i][kx] + self.similarity[i][kx])
                new_responsibility = self.similarity[i][k] - max(sum_as)
                self.responsiblity[i][k] = self.damping_f * self.responsiblity[i][k] + (1-self.damping_f)*new_responsibility
        
    def update_responsibility_v2(self):
        for i in range(self.size):
            sum_as = np.sum([self.availability[i,:], self.similarity[i,:]], axis=0)
            for k in range(self.size):
                max_sum_as = np.max(np.delete(sum_as, k))
                new_responsibility = self.similarity[i,k] - max_sum_as
                self.responsiblity[i,k] = self.damping_f * self.responsiblity[i,k] + (1-self.damping_f)*new_responsibility
                
    def update_availability(self):
        print("Calculating availability...")
        for i in range(self.size):
            for k in range(self.size):
                maxr = []
                if i==k:
                    for ix in range(self.size):
                        if ix != k:
                            maxr.append(max(0, self.responsiblity[ix][k]))
                    new_availability = sum(maxr)
                else:
                    for ix in range(self.size):
                        if ix not in [i, k]:
                            maxr.append(max(0, self.responsiblity[ix][k]))
                    new_availability = min(0, self.responsiblity[k][k] + sum(maxr))
                self.availability[i][k] = self.damping_f * self.availability[i][k] + (1-self.damping_f)*new_availability
                
    def update_availability_v2(self):
        zero_arr = np.zeros(self.size)
        for i in range(self.size):
            for k in range(self.size):
                max_r = np.maximum(zero_arr, self.responsiblity[:,k])
                if i==k:
                    sum_max_r = np.sum(np.delete(max_r, k))
                    new_availability = sum_max_r
                else:
                    sum_max_r = np.sum(np.delete(max_r, [i, k]))
                    new_availability = min(0, self.responsiblity[k,k] + sum_max_r)
                self.availability[i,k] = self.damping_f * self.availability[i,k] + (1-self.damping_f)*new_availability
                
    def generate_clusters(self):
        sums = np.add(self.responsiblity, self.availability)
        exemplars = np.argmax(sums, axis=1)
        clusters = {}
        for idx, exemplar in enumerate(exemplars):
            key = str(exemplar)
            if(key not in clusters):
                member = [idx]
                clusters[key] = member
            else:
                member = clusters[key]
                member.append(idx)
                clusters[key] = member
        return clusters
    
    def is_continue(self, new_cluster):
        self.iteration+=1   
        if new_cluster == self.clusters:
            self.changed += 1
        else:
            self.clusters = new_cluster
            self.changed = 1
        return self.changed <= self.changed_limit and self.iteration < self.max_iter
#==========================================================================================================================#
def get_preference(preference, similarity):
    if isinstance(preference, float) or isinstance(preference, int):
        return preference
    result = 0
    if preference == 'min':
        result = np.amin(similarity)
    elif preference == 'q1':
        result = np.quantile(similarity, 0.25)
    elif preference == 'median':
        result = np.median(similarity)
    elif preference == 'q3':
        result = np.quantile(similarity, 0.75)
    return result

def start(data, min_count=5, min_occur=15, segment_range_ms=6*3600*1000, preference='median', 
          damping_factor=.5, changed_limit=5, max_iteration=300, new_data=True):
#    data = Data(data)
    tm = TextMining(data)
    ap = AffinityPropagation(data)
    if new_data:
        data.reset()
        data.set_result('PARAMETER', {
                'min_count':min_count, 'min_occur':min_occur, 'segment_range_ms':segment_range_ms,
                'preference':preference, 'damping_factor':damping_factor,
                'changed_limit':changed_limit, 'max_iteration':max_iteration }) 
        tm.start(min_count, min_occur, segment_range_ms)  
    
    start = timeit.default_timer() 
    data.reset_cluster()
    data.set_result('PARAMETER', {
                'min_count':min_count, 'min_occur':min_occur, 'segment_range_ms':segment_range_ms,
                'preference':preference, 'damping_factor':damping_factor,
                'changed_limit':changed_limit, 'max_iteration':max_iteration })    
    
    print("[{}] Clustering each segment".format(dt.now()),end='')
    for i in range(data.get_result('segment_count')):
        print('.', end='')
        similarity = data.get_similarity(i)
        segment_preference = get_preference(preference, similarity)
        data.set_result('preference_of_segments', segment_preference, append=True) 
        ap.fit(similarity, segment_preference, damping_factor, changed_limit, max_iteration, i)
    print('')
    data.save_silhouette()
    data.set_result('clustering_time', timeit.default_timer()-start)
    data.set_result('clustering_last_run', dt.now().strftime("%A, %d %B %Y %I:%M:%S %p"))
        

#===========================================================================================================================#  
#data = Data('C:/Users/rzkydr/Documents/Python Scripts/Twitter/search1d')
#start(data, new_data = True)
#data.print_clusters(0)
#data = Data('stream')
#timestamp = data.get_timestamp()
#starting = timestamp[0]
#end = starting + (6*3600*1000)
#end2 = end + (6*3600*1000)
#print(timestamp[0])
#counter = 0
#for idx, stamp in enumerate(timestamp):
#    if stamp <= starting:
#        counter += 1
#        print(idx)
#    starting = stamp
#
#print(counter)
#print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end2/1000)))
#print('finish')
#==================================================================================
#data = Data('stream')
#clusters = data.get_result('~clusters')
#clusters_number = np.zeros(100,int)
#for segment in clusters:
#    for key,val in segment.iteritems():
#        member_count = len(val)
#        clusters_number[member_count] += 1
#print(len(clusters_number))
#      
      
      