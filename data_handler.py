from __future__ import print_function
from evaluation import Evaluation
import csv,  json, os, shutil
import numpy as np
import pandas as pd
import time
  

class Data:                  
    def __init__(self, working_path):
        self.PATH                       = working_path + '/'
        self.FILE_STOPWORD              = 'stopword.txt'
        self.FILE_FORMAL_DICT           = 'formalization-dict.txt'
        self.FILE_TWEET_CSV             = 'tweets_text.csv'
        self.FILE_PREPROCESSED          = 'preprocessed.csv'
        self.FILE_FILTERED_TWEET_CSV    = 'filtered_tweet.csv'
        self.FILE_TD_FREQUENCY          = 'td_frequency.csv'
        self.FILE_WEIGHTING             = 'weighting_result.csv'
        self.FILE_SIMILARITY            = 'similarity.csv'
        self.FILE_RESULT                = 'RESULT.json'
        self.RESULT                     = {}
        if os.path.isfile(self.PATH+self.FILE_RESULT):
            with open(self.PATH+self.FILE_RESULT,'r') as f:
                content = f.read()
                if content:
                    self.RESULT = json.loads(content)
                    clustering_result = ['iteration_of_segments','preference_of_segments','~clusters']
                    for key in clustering_result:
                        if key in self.RESULT:
                            del self.RESULT[key]                        
        else:
            f = open(self.PATH+self.FILE_RESULT,'w+')
                
    def set_result(self, name, value, append=False):
        if append:
            if name in self.RESULT:
                self.RESULT[name] = self.RESULT[name] + [value]  
            else:
                self.RESULT[name] = [value]
        else:        
            self.RESULT[name] = value
        self.update_json()
        
    def update_json(self):
        with open(self.PATH+self.FILE_RESULT,'w') as fp:
            json.dump(self.RESULT, fp, sort_keys=True, indent=4)
            
    def get_result(self, name):
        with open(self.PATH+self.FILE_RESULT,'r') as f:
            try:
                self.RESULT = json.loads(f.read())
            except ValueError, e:
                return None
        if name not in self.RESULT:
            return None
        return self.RESULT[name]
    
    def get_timestamp(self, filtered=False):
        filename = self.PATH+self.FILE_TWEET_CSV
        if filtered:
            filename = self.PATH+self.FILE_FILTERED_TWEET_CSV
            if not os.path.isfile(filename):
                return None
        docs = pd.read_csv(filename)
        return docs.timestamp_ms
            
    def get_documents(self, filtered=False):
        filename = self.PATH+self.FILE_TWEET_CSV
        if filtered:
            filename = self.PATH+self.FILE_FILTERED_TWEET_CSV
        docs = pd.read_csv(filename)
        return docs.text
    
    def get_segment_count(self):
        with open(self.PATH+self.FILE_RESULT, 'r') as f:
            data = json.load(f)
        return data['segment_count']
    
    def get_stopword(self):
        stopwords = []
        with open(self.FILE_STOPWORD) as f:
            stopwords = [line.strip() for line in f.readlines()]
        f.close()    
        return set(stopwords)
    
    def get_formalization(self):
        formal_dict = {}
        with open(self.FILE_FORMAL_DICT) as f:
            for line in f:
                rule = line.strip().split('\t')
                formal_dict[rule[0]] = rule[1]
        f.close()   
        return formal_dict
    
    def save_processed_docs(self, processed_docs):
        with open(self.PATH+self.FILE_PREPROCESSED, 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(processed_docs)
            
    def get_processed_docs(self):
        processed_docs = []
        with open(self.PATH+self.FILE_PREPROCESSED, "r") as f:
            for line in f:
                processed_docs.append([word.rstrip() for word in line.split(",")])
        return processed_docs
    
    def save_td_frequency(self, bag_of_word, term_f, doc_f):
        data = pd.DataFrame(term_f)
        data.insert(0, 'Term', bag_of_word)
        data['Document Frequency'] = doc_f
        data.to_csv(path_or_buf=self.PATH+self.FILE_TD_FREQUENCY, sep=',', index=False)
        
    def save_weighting(self, document_weight, index):
        np.savetxt(self.PATH+str(index)+'_'+self.FILE_WEIGHTING, document_weight, delimiter=',')
    
    def get_weighting(self, index):
        document_weight = np.loadtxt(open(self.PATH+str(index)+'_'+self.FILE_WEIGHTING, 'rb'), delimiter=',')
        return document_weight
        
    def save_similarity(self, similarity, index):
        with open(self.PATH+str(index)+'_'+self.FILE_SIMILARITY, "wb") as f:
            writer = csv.writer(f)
            writer.writerows(similarity)
            
    def get_similarity(self, index):
        similarity = np.loadtxt(open(self.PATH+str(index)+'_'+self.FILE_SIMILARITY, 'rb'), delimiter=',')
        return similarity
    
    def save_filtered_docs(self, result):
        with open(self.PATH+self.FILE_TWEET_CSV,'r') as f1, open(self.PATH+self.FILE_FILTERED_TWEET_CSV,'wb') as f2:
            reader = csv.reader(f1)
            writer = csv.writer(f2)
            writer.writerow(next(reader))
            for idx, row in enumerate(reader):
                if result[idx][1]:
                    writer.writerow(row)
         
    def get_segment_list(self):
        timestamp = self.get_timestamp(filtered=True)
        if timestamp.empty:
            return []
        first_timestamp = timestamp[0]
        last_timestamp = timestamp[len(timestamp)-1]
        segment_range_ms = self.get_result('PARAMETER')['segment_range_ms']
        segment_count = self.get_result('segment_count')
        segment_list = []
        segment_name = time.strftime('(Segmen 0) %Y %B %d, %H:%M-', time.localtime(first_timestamp/1000))
        start = first_timestamp
        for i in range(segment_count-1):
            start += segment_range_ms
            hm = time.strftime('%H:%M', time.localtime(start/1000))
            segment_name += hm
            segment_list.append(segment_name)
            segment_name = time.strftime('(Segmen '+str(i+1)+') %Y %B %d, %H:%M-', time.localtime(start/1000))
        hm = time.strftime('%H:%M', time.localtime(last_timestamp/1000))
        segment_name += hm
        segment_list.append(segment_name)
        return segment_list
    
    def get_summary(self, segment, to, keyword):
        summary = []
        if to is None or to == 0:
            to = segment + 1
        else:
            to += 1

        for i in range(segment, to):
            summary += self.print_clusters(i)
        if keyword is not None and keyword:
            if 'id:' in keyword:
                id = keyword.split(':')[1].split('.')
                return self.print_member(int(id[0]), int(id[1]))
            else:
                return [result for result in summary if keyword in result]
        return summary
    def get_statistic(self):
        statistic = []  
        for key, val in self.RESULT.items():
            if key not in ['~clusters', 'PARAMETER']:
                stats = str(val)
#                if isinstance(val, list):
#                    stats = ', '.join(str(val))
                statistic.append('{}: \n{}\n{}'.format(key,str(stats),''.join(['=' for i in range(25)])))
                
        return statistic
    
    def print_member(self, segment, index):
        cluster = self.get_result('~clusters')[segment]
        documents = self.get_documents(filtered=True)
        end_of_segment = self.get_result('end_of_segments')
        segment_start = end_of_segment[segment-1] if segment > 0 else 0
        member = []
        result = []
        for idx, (key,val) in enumerate(cluster.items()):
            if idx == index:
                member = val
                break
        for doc in member:
            tweet = documents[segment_start+doc]
            result_member = '({}){}\n{}'.format(segment_start+doc,tweet, ''.join(['=' for i in range(100)]))
            result.append(result_member)
        return result
    
    def save_as_csv(self, data, filename, index):
        np.savetxt(self.PATH+str(index)+'_'+filename, data, delimiter=',')
        
    def print_clusters(self, segment):
        cluster = self.get_result('~clusters')[segment]
        documents = self.get_documents(filtered=True)
        filtered_docs = self.get_processed_docs()
        end_of_segment = self.get_result('end_of_segments')
        segment_start = end_of_segment[segment-1] if segment > 0 else 0
        result = []
        for index, (key,val) in enumerate(cluster.items()):
            if len(val) > 10:
                exemplar = documents[segment_start+int(key)]
            #==========
#                member_tweet = [filtered_docs[segment_start+i] for i in val]
#                bag_of_word = list(set([term for docs in member_tweet for term in docs]))     
#                df_dict = {term:0 for term in bag_of_word}
#                for doc in member_tweet:
#                    if term in doc:
#                        df_dict[term] += 1 
#                sorted_df = sorted(df_dict, key=df_dict.get, reverse=True)
#                top_keyword = [sorted_df[i] for i in range(3)]
                top_keyword = ['']
                result_cluster = '({}.{}) Keyword: ({}) Member: {}\n({}){}\n{}'.format(
                        segment, index, ', '.join(top_keyword), len(val),segment_start+int(key), exemplar, ''.join(['=' for i in range(100)]))
                result.append(result_cluster)
        return result
        
    def reset(self):
        directory = os.path.join(os.getcwd(),self.PATH)
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f))]
        self.RESULT = {}
        excluded = [self.FILE_TWEET_CSV, self.FILE_PREPROCESSED]
        for filename in files:
            if filename not in excluded:
                os.remove(os.path.join(directory, filename))
                
    def reset_cluster(self):
        if os.path.isfile(self.PATH+self.FILE_RESULT):
            with open(self.PATH+self.FILE_RESULT,'r') as f:
                content = f.read()
                if content:
                    self.RESULT = json.loads(content)
                    clustering_result = ['iteration_of_segments','preference_of_segments','cluster_of_segments','silhouette','~clusters']
                    for key in clustering_result:
                        if key in self.RESULT:
                            del self.RESULT[key]
        self.update_json()
        
    def save_silhouette(self):
        test = Evaluation()
        clusters = self.get_result('~clusters')
        count = self.get_result('segment_count')
        silhouette_of_segment = np.zeros(count)
        for i in range(count):
            similarity = self.get_similarity(i)
            silhouette = test.silhouette(clusters[i], similarity)
            silhouette_of_segment[i] = silhouette
        result = np.average(silhouette_of_segment)
        self.set_result('silhouette', result)
        
    def copy_log_as(self, name):
        new_dir = self.PATH+'EVALUATION/'
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        newfile = '{}{}_{}'.format(new_dir, name, self.FILE_RESULT)
        shutil.copyfile(self.PATH+self.FILE_RESULT, newfile)