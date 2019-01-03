# -*- coding: utf-8 -*-
"""
Created on Wed Mar 07 15:37:54 2018

@author: rzkydr
"""

import numpy as np
class Evaluation:
    
    """
    Metode utama untuk menghitung ROGUE-N dari ringkasan sistem dan manual
    :param system: String, hasil ringkasan sebuah klaster dari sistem
    :param manual: String, hasil ringkasan sebuah klaster dari pakar
    """
    def rogue_n(self, system, manual):
        sys = [word.lower() for word in system.split()]
        man = [word.lower() for word in manual.split()]
        
        match = [word for word in man if word in sys]
        
        return len(match)/len(man)
    
    """
    Metode utama untuk menghitung silhouette seluruh cluster
    :param clusters: 2D List, setiap list merepresentasikan cluster dan berisi index data
    :param similiarity: 2D List, similiarity antar data berukuran NxN dimana n adalah banyak data
    """
    def silhouette(self, clusters, similiarity):
        self.clusters = clusters
        self.similiarity = similiarity
        silhouette = np.zeros(len(similiarity))
        for i in range(len(similiarity)):
            silhouette[i] = self.get_single_silhouette(i)
        
        return np.average(silhouette)
    """
    Metode untuk menghitung silhouette 1 data point
    :param point: Index data pointx`
    """        
    def get_single_silhouette(self, point):
        inner = -1
        outer = -1
        for key, val in self.clusters.iteritems():
            avg_dist = self.point_to_cluster(point, val)
            if point in val:
                inner = avg_dist
            else:
                if avg_dist > outer:
                    outer = avg_dist
        if inner == outer:
            return 0
        else:
            return (inner - outer) / max(inner, outer)
        
    """
    Metode untuk menghitung rata-rata similiarity antar data dengan data lain pada sebuah cluster
    :param point: Index data point
    :param cluster_child: List, berisi index data yang ada dalam sebuah cluster
    """       
    def point_to_cluster(self, point, cluster_child):
        sum_of_similarity = 0
        counter = 0
        for idx in cluster_child:
            if idx != point:
                sum_of_similarity += self.similiarity[point, idx]
                counter += 1
                
        if counter == 0:
            return 0
        else:
            return sum_of_similarity/counter
    
    
if __name__ == "__main__":
    import main_clean as mc
    from data_handler import Data
    from datetime import datetime as dt
    def evaluation_by(data, name, test_value, min_count, min_occur, segment_range_ms, preference, damping_factor, changed_limit, max_iteration):
        silhouettes = np.zeros(len(test_value))
        print('{}'.format(''.join(['=' for i in range(90)])))
        for index, val in enumerate(test_value):
            if name == 'preference':
                preference = val
            elif name == 'damping_factor':
                damping_factor = val
            elif name == 'changed_limit':
                changed_limit = val
            elif name == 'max_iteration':
                max_iteration = val
#            new_data = (name=='preference' and index == 0)
            
            print('[{}] Testing {} = {}'.format(dt.now(), name, val))
            mc.start(data, new_data = False, min_count=min_count, min_occur=min_occur, segment_range_ms=segment_range_ms, 
                     preference=preference, damping_factor=damping_factor, changed_limit=changed_limit, max_iteration=max_iteration)
            
            result = data.get_result('silhouette')
            silhouettes[index] = result
            print('[{}] Resulting Silhouette = {}'.format(dt.now(), result))
            data.copy_log_as('{}={}'.format(name, val))
        best_index = np.argmax(silhouettes)
        print('Best {} = {}'.format(name, test_value[best_index]))
        return test_value[best_index]
        
    
    def start_evaluation():
        data = Data('search')
        min_count = 5
        min_occur = 82
        segment_range_ms = 6*3600*1000
        preference_test = ['min','q1','median','q3']
        damping_factor_test = np.linspace(0,1,10, endpoint=False)
        changed_limit_test = range(1,11)
        max_iteration_test = range(5, 101, 5)
        
        best_p = evaluation_by(data, 'preference', preference_test, min_count, min_occur, segment_range_ms, None, 0.5, 2, 10)
        best_df = evaluation_by(data, 'damping_factor',damping_factor_test, min_count, min_occur, segment_range_ms, best_p, None, 2, 10)
        best_cl = evaluation_by(data, 'changed_limit', changed_limit_test, min_count, min_occur, segment_range_ms, best_p, best_df, None, 10)
        best_mi = evaluation_by(data, 'max_iteration', max_iteration_test, min_count, min_occur, segment_range_ms, best_p, best_df, best_cl, None)
        
    start_evaluation()
