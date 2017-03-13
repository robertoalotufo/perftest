import sys
import pickle
import time
import os
import inspect
import numpy as np
from glob import glob
#import posix_time as pt
    
class Tester():
    def __init__(self, assert_fnc, base_namespace, base_page, runs=3):
        self._assert = assert_fnc
        self._base_path = self.__get_path(base_namespace, base_page)
        self._runs = runs
        self._test_cases_file = '{base_path}/test_cases.pkl'.format(base_path=self._base_path)
        self._train_cases_file = '{base_path}/train_cases.pkl'.format(base_path=self._base_path)
        self._local_results = []
        self.__load_cases()
        
    def __save(self, data, filename, make_backup=False):
        with open(filename, 'wb') as f:            
            pickle.dump(data, f)

    def __load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data
        
    def __get_path(self, namespace, page):
        # path = os.path.join(aws_config.media_root, aws_config.attachment_media_prefix, namespace, page)
        path = os.path.abspath('.')
        if not os.path.isdir(path):
            os.makedirs(path)
        return path
        
    def remove_pages(self, pages):
        for namespace, page in pages:
            try:
                user_results_file = '{base_path}/{namespace}.{page}.results.pkl'.format(base_path=self._base_path, namespace=namespace, page=page)
                os.remove(user_results_file)
            except:
                pass
        
    def __load_cases(self):
        try:
            self._train_cases = self.__load(self._train_cases_file)
        except:
            self._train_cases = []
            
        try:
            self._test_cases = self.__load(self._test_cases_file)
        except:
            self._test_cases = []
        
    def reset_cases(self):
        self._test_cases  = []
        self._train_cases = []
            
    def add_test_case(self, case_input, case_output, weight=1, show_in_results_table=False, label='undefined'):
        self._test_cases.append((case_input, case_output, weight, show_in_results_table, u'{}'.format(label)))
        self.__save(self._test_cases, self._test_cases_file)
        
    def add_train_case(self, case_input, case_output):
        self._train_cases.append((case_input, case_output))
        self.__save(self._train_cases, self._train_cases_file)
        
    def get_input(self, case_idx=0):
        s = self._train_cases[case_idx][0]
        if type(s) == np.ndarray:
            return s.copy()
        else:
            return s
        
    def get_output(self, case_idx=0):
        return self._train_cases[case_idx][1]
        
    def run_test(self, fnc, username=None, fncname=None):
        fnc_path = inspect.getsourcefile(fnc)
        if fnc is None:
            raise Exception('Your function must be defined as module.')
        
        #### U_PACKAGE, U_MODULE
        #u_package, u_module = tuple(fnc_path.replace('.','/').split('/')[-3:-1])
        #user_results_file = '{base_path}/{namespace}.{page}.results.pkl'.format(base_path=self._base_path, namespace=u_package, page=u_module)
        user_results_file = 'myresults'
        
        #### USERNAME
        #username = u'{user}'.format(user=u_module.split('_')[0] if username is None else unicode(username, 'utf-8'))
        #if (username == 'activity') or (username == 'library'):
        #   username = u'{user}'.format(user=u_module.split('_')[1])
        username = 'roberto'
        
        
        fncname = u'{fnc}'.format(fnc='{fnc}()'.format(fnc=fnc.__name__) if fncname is None else u'{fnc}'.format(fnc=fncname))
        
        
        #results = {'namespace': u_package, 'page': u_module}
        results = {}
        results['username'] = username
        results['fncname'] = fncname
        results['partials'] = []
        results['lapsed_time'] = 0
        results['score'] = 0
        weight_sum = 0
        for case in self._test_cases:
            min_lapsed_time = float('inf')
            weight_sum += case[2]
            score = 1.
            for i in range(self._runs):
                time_marker = time.time()
                #time_marker = pt.get_cpu_time()
                try:
                    score_a = self._assert(fnc, case[0], case[1])
                    score = min(score,score_a)   # bug para testar se não alterou imagem de entrada
                except:
                    score = 0
                    min_lapsed_time = 0
                    break
                lapsed_time = (time.time() - time_marker)*1000
                #lapsed_time = (pt.get_cpu_time() - time_marker)*1000.
                min_lapsed_time = min(min_lapsed_time, lapsed_time)
            results['lapsed_time'] += min_lapsed_time
            results['score'] += (score*case[2])
            if case[3]:
                results['partials'].append((score, min_lapsed_time))
        results['score']=1.0*results['score']/weight_sum
        self._local_results.append(results)
        self.__save(self._local_results, user_results_file)
        
    def __prepare_data__(self, results, partial_count, count, fnclink=None):
        table = []
        sizes = []
        scores = []
        lapsed_times = []
        for result in results:
            count += 1
            table_line = []
            column_sizes = []
            table_line.append('{:d}'.format(count))
            column_sizes.append(len(table_line[-1]))
            table_line.append(result['username'])
            column_sizes.append(len(table_line[-1]))
            if fnclink is None:
                table_line.append(result['fncname'])
            else:
                table_line.append(u'`{link} {fnc}`'.format(link=fnclink, fnc=result['fncname']))
            column_sizes.append(len(table_line[-1]))
            for i in range(partial_count):
                try:
                    score, lapsed_time = result['partials'][i]
                    score = '{:.0%}'.format(score)
                    lapsed_time = '{:.3f}'.format(lapsed_time)
                except:
                    score, lapsed_time = '---', '---'
                table_line.append(lapsed_time)
                column_sizes.append(len(table_line[-1]))
                table_line.append(score)
                column_sizes.append(len(table_line[-1]))
            score, lapsed_time = result['score'], result['lapsed_time']
            scores.append(score)
            lapsed_times.append(lapsed_time)
            score = '{:.0%}'.format(score)
            lapsed_time = '{:.3f}'.format(lapsed_time)
            table_line.append(lapsed_time)
            column_sizes.append(len(table_line[-1]))
            table_line.append(score)
            column_sizes.append(len(table_line[-1]))
            table.append(table_line)
            sizes.append(column_sizes)
        return (table, sizes, scores, lapsed_times)
        
    def __print_results_table(self, table, column_sizes, partial_labels, sequence, fnc_ranking, user_ranking):
        partial_labels.append(u'Todas as execuções')
        lblScore = u'**Acertos**'
        lblTime = u'**Tempo (ms)**'
        # Building the header
        column_sizes[0] = 24
        column_sizes[1] = max(column_sizes[1], 9)
        column_sizes[2] = max(column_sizes[2], 14)
        header_top = '+{2:-<{idx_len}}+{0:-<{autor_len}}+{1:-<{fnc_len}}+'.format('', '', '', idx_len=column_sizes[0], autor_len=column_sizes[1], fnc_len=column_sizes[2])
        header_div = '|{2:<{idx_len}}|{0:<{autor_len}}|{0:<{fnc_len}}+'.format('', '', '', idx_len=column_sizes[0], autor_len=column_sizes[1], fnc_len=column_sizes[2])
        table_div = '+{2:-<{idx_len}}+{0:-<{autor_len}}+{1:-<{fnc_len}}+'.format('', '', '', idx_len=column_sizes[0], autor_len=column_sizes[1], fnc_len=column_sizes[2])
        header1 = '|{2:<{idx_len}}|{0:<{autor_len}}|{1:<{fnc_len}}|'.format('', '', '', idx_len=column_sizes[0], autor_len=column_sizes[1], fnc_len=column_sizes[2])
        header2 = '|{0:<{idx_len}}|{autor:<{autor_len}}|{fnc:<{fnc_len}}|'.format(u'**Ranking Autor-Função**', idx_len=column_sizes[0], autor='**Autor**', autor_len=column_sizes[1], fnc=u'**Função**', fnc_len=column_sizes[2])
        for i, sector_label in enumerate(partial_labels):
            sector_label = u'**{}**'.format(sector_label)
            a, b = column_sizes[3+i*2:5+i*2]
            a = max(a, 14)
            b = max(b, 11)
            sector_len = max(a+b, len(sector_label))+1
            a+=(sector_len-a-b-1)
            column_sizes[3+i*2:5+i*2] = [a,b]
            header_top+='{0:-<{sector_len}}+'.format('', sector_len=sector_len)
            z = '{0:-<{score_len}}+{1:-<{time_len}}+'.format('', '', score_len=a, time_len=b)
            header_div+=z
            table_div+=z
            header1+='{sector_label:<{sector_len}}|'.format(sector_label=sector_label, sector_len=sector_len)
            header2+='{time:<{time_len}}|{score:<{score_len}}|'.format(score=lblScore, score_len=b, time=lblTime, time_len=a)
        print(header_top)
        print(header1)
        print(header_div)
        print(header2)
        print(table_div)
        #count = 1
        for i in sequence:
            data = table[i]
            line = '|{0: <{idx_len}}|{autor:<{autor_len}}|{fnc:<{fnc_len}}|'.format('{} - {}'.format(user_ranking[i]+1, fnc_ranking[i]+1), idx_len=column_sizes[0], autor=data[1], autor_len=column_sizes[1], fnc=data[2], fnc_len=column_sizes[2])
            #count += 1
            for j in range(len(partial_labels)):
                a, b = column_sizes[3+j*2:5+j*2]
                line+='{score:<{score_len}}|{time:<{time_len}}|'.format(score=data[3+j*2], score_len=a, time=data[4+j*2], time_len=b)
            print(line)
            print(table_div)
        
    def __show_results_table__(self, local_only=True):
        num_pages, num_functions = 0,0
        partial_labels = []
        for case in self._test_cases:
            if case[3]:
                partial_labels.append(case[4])
            
        count = 0
        if local_only:
            table, column_sizes, scores, lapsed_times = self.__prepare_data(self._local_results, len(partial_labels), count)
        else:
            idx = []
            result_files = glob(self._base_path+'/*.results.pkl')
            table, column_sizes, scores, lapsed_times = [], [], [], []
            for partial_result_file in result_files:
                try:
                    partial_results = self.__load(partial_result_file)
                    t = self.__prepare_data(partial_results, len(partial_labels), count, '{}:{}'.format(partial_results[0]['namespace'], partial_results[0]['page']))
                    count += len(partial_results)
                    idx.append(len(partial_results))
                    table.extend(t[0])
                    column_sizes.extend(t[1])
                    scores.extend(t[2])
                    lapsed_times.extend(t[3])
                    num_pages += 1
                    num_functions += len(partial_results)
                except:
                    pass
            
        if len(table)==0:
            print(u'Nenhum resultado encontrado.')
            return
            
        column_sizes = np.array(column_sizes).max(axis=0)
        scores = 1.0/np.array(scores)
            
        fnc_ranking = np.argsort(np.lexsort((lapsed_times, scores)))
            
        if not local_only:
            user_ranking = []
            x = 0
            for i in idx:
                user_ranking.append(np.ones(i, dtype='int')*fnc_ranking[x:x+i].min())
                x+=i
            user_ranking = np.hstack(user_ranking)
            print('Tabela de resultados consolidados de %d funções distribuídas em %d páginas distintas.'%(num_functions, num_pages))
            print()
        else:
            user_ranking = np.ones(fnc_ranking.shape, dtype='int')*fnc_ranking.min()
            
        sequence = np.lexsort((fnc_ranking, user_ranking))
        tf = np.zeros(user_ranking.max()+1, dtype='int')
        ranks = np.unique(user_ranking)
        tf[ranks] = np.arange(ranks.size)
        user_ranking = tf[user_ranking]
        self.__print_results_table(table, column_sizes, partial_labels, sequence, fnc_ranking, user_ranking.astype('int'))
        
    def __prepare_data(self, results, partial_count, count, fnclink=None):
        table = []
        scores = []
        lapsed_times = []
        for result in results:
            count += 1
            table_line = []
            table_line.append('"{}"'.format(result['username']))
            if fnclink is None:
                table_line.append('"{}"'.format(result['fncname']))
            else:
                table_line.append(u'"`{link} {fnc}`"'.format(link=fnclink, fnc=result['fncname']))
            for i in range(partial_count):
                try:
                    score, lapsed_time = result['partials'][i]
                    score = '{:.0%}'.format(score)
                    lapsed_time = '{:.3f}'.format(lapsed_time)
                except:
                    score, lapsed_time = '---', '---'
                table_line.append(lapsed_time)
                table_line.append(score)
            score, lapsed_time = result['score'], result['lapsed_time']
            scores.append(score)
            lapsed_times.append(lapsed_time)
            score = '{:.0%}'.format(score)
            lapsed_time = '{:.3f}'.format(lapsed_time)
            table_line.append(lapsed_time)
            table_line.append(score)
            table.append(table_line)
        return (table, scores, lapsed_times)
        
    def show_results_table(self, local_only=True, rank='user'):
        num_pages, num_functions = 0,0
        partial_labels = []
        table_header = ['   "Ranking do autor"','"Ranking da função"','"Autor"','"Função"']
        for case in self._test_cases:
            if case[3]:
                partial_labels.append(case[4])
                table_header.append('"{} (tempo em ms)"'.format(case[4]))
                table_header.append('"{} (pontuação)"'.format(case[4]))
        table_header.extend(['"Tempo total (ms)"','"Pontuação total"'])
            
        count = 0
        if local_only:
            table, scores, lapsed_times = self.__prepare_data(self._local_results, len(partial_labels), count)
        else:
            idx = []
            result_files = glob(self._base_path+'/*.results.pkl')
            table, scores, lapsed_times = [], [], []
            for partial_result_file in result_files:
                try:
                    partial_results = self.__load(partial_result_file)
                    t = self.__prepare_data(partial_results, len(partial_labels), count, '{}:{}'.format(partial_results[0]['namespace'], partial_results[0]['page']))
                    count += len(partial_results)
                    idx.append(len(partial_results))
                    table.extend(t[0])
                    scores.extend(t[1])
                    lapsed_times.extend(t[2])
                    num_pages += 1
                    num_functions += len(partial_results)
                except:
                    pass
            
        n = len(table)
        if n==0:
            print(u'Nenhum resultado encontrado.')
            return
            
        table = np.array(table)
        scores = 1.0/np.array(scores)
            
        fnc_ranking = np.argsort(np.lexsort((lapsed_times, scores)))
            
        if not local_only:
            user_ranking = []
            x = 0
            for i in idx:
                user_ranking.append(np.ones(i, dtype='int')*fnc_ranking[x:x+i].min())
                x+=i
            user_ranking = np.hstack(user_ranking)
        else:
            user_ranking = np.ones(fnc_ranking.shape, dtype='int')*fnc_ranking.min()
        if rank == 'user':
            sequence = np.lexsort((fnc_ranking, user_ranking))
        else:
            sequence = np.lexsort((user_ranking, fnc_ranking))
            
        tf = np.zeros(user_ranking.max()+1, dtype='int')
        ranks = np.unique(user_ranking)
        tf[ranks] = np.arange(ranks.size)
        user_ranking = tf[user_ranking]
            
        fnc_ranking += 1
        user_ranking += 1
        table = np.hstack((np.char.mod('   %d', user_ranking[sequence]).reshape(n,1), fnc_ranking[sequence].reshape(n,1), table[sequence]))
        #table_header = np.array(table_header)
        #print table_header
        #table_header.shape = (1,) + table_header.shape
        #print table_header.shape, table.shape
        table = np.vstack((table_header, table))
        print('.. csv-table:: Tabela de resultados consolidados de %d funções distribuídas em %d páginas distintas.'%(num_functions, num_pages))
        print('   :header-rows: 1')
        print()
        np.savetxt(sys.stdout, table, delimiter=",", fmt='%s')
        if not local_only:
            np.savetxt('{}/results.csv'.format(self._base_path), table, delimiter=",", fmt='%s')
            #print
            #print '`{}/results.csv Baixar arquivo CSV`'.format(self._base_path).replace('/awmedia/www', '')

