import Tkinter as tk
import ttk
from tkFileDialog import askopenfilename
from tkFileDialog import askdirectory 
from data_handler import Data
import main_clean as mc

class SummarizerApp():
    def __init__(self, root):
        self.path = None
        self.min_count = tk.IntVar()
        self.min_occur = tk.IntVar()
        self.segment_range = tk.DoubleVar()
        self.preference = tk.StringVar()
        self.damping_factor = tk.DoubleVar()
        self.max_iteration = tk.IntVar()
        self.new_data = tk.IntVar()
        self.from_segment = tk.IntVar()
        self.to_segment = tk.IntVar()
        self.search_keyword = tk.StringVar()     
        #==============================================================================
        
        frame = ttk.Frame(root, width=1000, height=600)
        frame.pack()
        
        path_button = ttk.Button(frame, text='Open Folder', command=self.action_open_path)
        path_button.grid(row=0, column=0, padx=5, pady=5,sticky='nsew')
        
        self.create_input(frame, 'Min Count', self.min_count, 25, 10, 0, 1)
        self.create_input(frame, 'Min Occur', self.min_occur, 25, 10, 1, 1)
        self.create_input(frame, 'Segment Range (hour)', self.segment_range, 25, 10, 2, 1)
        self.create_input(frame, 'Preference (median or 0-1)', self.preference, 25, 10, 0, 2)
        self.create_input(frame, 'Damping Factor (0-1)', self.damping_factor, 25, 10, 1, 2)
        self.create_input(frame, 'Max Iteration', self.max_iteration, 25, 10, 2, 2)
        
        start_frame = tk.Frame(frame)
        start_frame.grid(row=0,column=3,rowspan=3,sticky='nsew')
        
        restart_checkbox = ttk.Checkbutton(start_frame , text='Process new data', variable=self.new_data)
        restart_checkbox.grid(row=0,column=0,sticky='nw', padx=5, pady=5)
        start_button = ttk.Button(start_frame, text='Start', command=self.action_start)
        start_button.grid(row=1,column=0,sticky='nsew', padx=5, pady=5)
#        expected_label = ttk.Label(start_frame, text='Expected Time : 10h14m32s', width=38,anchor='w')
#        expected_label.grid(row=2,column=0)
#        status_label = ttk.Label(start_frame, text='Status : Clustering (1/4) segment', width=38,anchor='w')
#        status_label.grid(row=3,column=0)
        #==============================================================================
        
        cluster_frame = ttk.LabelFrame(frame, text='Summaries')
        cluster_frame.grid(row=5,column=0,columnspan=3, padx=5, pady=5,sticky='nsew')
        
        self.selected_segment = tk.StringVar()
        self.segment_option = ttk.OptionMenu(cluster_frame, self.selected_segment, 'Select a segment...')
        self.segment_option.configure(width=40)
        self.segment_option.grid(row=0,column=0,sticky='ew')
        
        self.create_input(cluster_frame, 'From', self.from_segment, 5, 5, 0, 1)
        self.create_input(cluster_frame, 'To', self.to_segment, 5, 5, 0, 2)
        self.create_input(cluster_frame, 'Search', self.search_keyword, 8, 20, 0, 3)
        filter_button = ttk.Button(cluster_frame, text='Filter', command=self.action_filter)
        filter_button.grid(row=0, column=4, padx=5, pady=5,sticky='nsew')
        
        summary_frame = ttk.Frame(cluster_frame)
        summary_frame.grid(row=1,column=0,columnspan=5, padx=5, pady=5,sticky='nsew')
        self.summary_text = tk.Text(summary_frame, width=100 ,wrap="word")
        summary_sb = ttk.Scrollbar(summary_frame, orient="vertical")
        self.summary_text.configure(yscrollcommand=summary_sb.set)
        self.summary_text.configure(font=("Calibri", 12))
        summary_sb.configure(command=self.summary_text.yview)
        self.summary_text.grid(row=0,column=0,sticky="nsew")
        summary_sb.grid(row=0,column=1,sticky="nsew")
        
        statistic_frame = ttk.LabelFrame(frame, text='Statistics')
        statistic_frame.grid(row=5,column=3, padx=5, pady=5,sticky='nsew')
        self.statistic_text = tk.Text(statistic_frame,width=25, wrap="word")
        scrollbar = ttk.Scrollbar(statistic_frame, orient="vertical")
        self.statistic_text.configure(yscrollcommand=scrollbar.set)
        self.statistic_text.configure(font=("Calibri", 12))
        scrollbar.configure(command=self.statistic_text.yview)
        self.statistic_text.grid(row=0,column=0,sticky="nsew")
        scrollbar.grid(row=0,column=1,sticky="nsew")

    def action_filter(self):
        s1 = self.from_segment.get()
        s2 = self.to_segment.get()
        keyword = self.search_keyword.get()
        if None in [s1,s2,keyword] and '' in [s1,s2,keyword]:
            return
        self.load_summary(s1, to_segment=s2, keyword=keyword) 
        
    def action_open_path(self):
        self.path = askdirectory()
        if not self.path:
            return
        data = Data(self.path)
        param = data.get_result('PARAMETER')
        if param is not None:
            self.min_count.set(param['min_count'])
            self.min_occur.set(param['min_occur'])
            self.segment_range.set(param['segment_range_ms']/(3600*1000))
            self.preference.set(param['preference'])
            self.damping_factor.set(param['damping_factor'])
            self.max_iteration.set(param['max_iteration'])
        
            segment_list = data.get_segment_list()
            if len(segment_list) > 0:
                option = self.segment_option['menu']
                option.delete(0,'end')
                for index, string in enumerate(segment_list):
                    option.add_command(label=string, command=lambda idx=index, val=string: self.load_summary(idx, name=val) )
            
            self.statistic_text.delete('1.0','end')
            statistic = data.get_statistic()
            for text in statistic:
                self.statistic_text.insert('insert', text + '\n')
        
    def load_summary(self, segment, name=None, to_segment=0, keyword=None):
        if not self.path:
            return
        if name is not None:
            self.selected_segment.set(name)
        data = Data(self.path)
        self.summary_text.delete('1.0','end')
        summary = data.get_summary(segment, to_segment, keyword)
        for text in summary:
            self.summary_text.insert('insert', text + '\n')
        
    def action_start(self):
        data = Data(self.path)
        pref = self.preference.get()
        pref = pref if pref == 'median' or 'max' else float(pref)
        param = [self.min_count.get(), self.min_occur.get(),
                 self.segment_range.get(), pref, 
                 self.damping_factor.get(), self.max_iteration.get()]
        if None not in param:
            mc.start(data, min_count=param[0], min_occur=param[1],
                     segment_range_ms=param[2]*3600*1000, preference=param[3], 
                     damping_factor=param[4], max_iteration=param[5],
                     new_data=self.new_data.get())
            segment_list = data.get_segment_list()
            if len(segment_list) > 0:
                option = self.segment_option['menu']
                option.delete(0,'end')
                for index, string in enumerate(segment_list):
                    option.add_command(label=string, command=lambda idx=index, val=string: self.load_summary(idx, name=val) )
            
            self.statistic_text.delete('1.0','end')
            statistic = data.get_statistic()
            for text in statistic:
                self.statistic_text.insert('insert', text + '\n')
                
    def create_input(self,master, text, var, width1, width2, row, column):
        sub_frame = ttk.Frame(master)
        sub_frame.grid(row=row,column=column,padx=5,pady=5,ipady=2,sticky='nsew')
        ttk.Label(sub_frame,text=text,width=width1,anchor='w').grid(row=0,column=0)
        ttk.Entry(sub_frame,textvariable=var,width=width2).grid(row=0,column=1,sticky='e')

root = tk.Tk()
root.title('Summarization')
app = SummarizerApp(root)
root.mainloop()

