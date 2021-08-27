import sys
import os
import numpy as np
import logging
import glob
import signal
import time
import math
import re
import colorful as cf
from settings import log, bar_format, Settings
from shutil import copyfile 
from distutils.dir_util import copy_tree
from tqdm import tqdm
from shutil import rmtree
from subprocess import Popen, PIPE, STDOUT
from mutator import Mutator

class Fuzzer(Settings):
    def __init__(self):
        super().__init__()
        for a_dir in self.dirs.values():
            os.makedirs(a_dir, exist_ok=True)
        keys = ['time', 'path', 'edge', 'timeout', 'crash', 'coverage', 'executed']
        values = [0] * len(keys)
        self.stats = dict(zip(keys, values))

        in_files = sorted(glob.glob('%s/id:*' % self.dirs['in']))
        self.prev_id = int(in_files[-1].split(':')[1].split(',')[0])

    def run_verifier(self, in_dir, ops, skip=0):

        report_file = self.files['report']
        map_file = self.files['map']
        target = self.targets['fast']
        trace_dir = self.dirs['trace']
        verifier = self.files['verifier']
        timeout = self.props['timeout']
        max_file = self.props['max_file']
        one_line = 1000

        prog = [verifier, '-i', in_dir, '-d', str(timeout), '-s', str(max_file)]
        for op in ops:
            if op == 'e':
                prog += ['-e', str(skip)]
            if op == 'm':
                prog += ['-m', map_file]
            if op == 'r':
                prog += ['-r', report_file]
            if op == 't':
                prog += ['-t', trace_dir]
            if op == 'u':
                prog += ['-u', in_dir]

        target = target.strip().split(' ')
        prog += target
        p = Popen(prog, stdout = PIPE, stderr = STDOUT)
        num_found = 0
        for line_number in range(0, 4): 
            line = p.stdout.readline().decode('utf-8')
            if 'Spinning up' in line and line_number == 0:
                continue
            elif 'All right' in line and line_number == 1:
                continue
            elif 'Scanning' in line and line_number == 2:
                continue
            elif 'Found' in line and line_number == 3:
                num_found = int(re.search('Found\s*(\d+)', line).group(1))
                log.info('found {}'.format(num_found))
            else:
                log.error('error: {}'.format(line))
                p.kill()
                sys.exit(1)
        remaining_line = int(num_found / one_line)
        for i in tqdm(range(0, remaining_line), bar_format=bar_format):
            p.stdout.readline()
        os.waitpid(p.pid, 0)

        return num_found

    def move_testcases(self):
        crash_dir = self.dirs['crash']
        topk_dir = self.dirs['topk']
        timeout_dir = self.dirs['timeout']
        report_file = self.files['report']
        files = open(report_file, 'r').read().strip()
        if len(files):
            files = files.split('\n')
            for src in files:
                if int(src[0]) == 0:
                    dest_dir = topk_dir
                elif int(src[0]) == 1:
                    dest_dir = timeout_dir
                elif int(src[0]) == 2:
                    dest_dir = crash_dir
                self.prev_id += 1
                src, dest = src[4:], '{}/{}'.format(dest_dir, src.split('/')[-1])
                dest = re.sub('id:(\d+),src', 'id:%06d,src' % self.prev_id, dest)
                copyfile(src, dest)

    def collect_stats(self, num_found):
        report_file = self.files['report']
        map_file = self.files['map']
        in_byte = np.frombuffer(open(map_file, 'rb').read(), dtype=np.uint8, count=(1 << 16))
        coverage = in_byte[in_byte != 255]

        files = open(report_file, 'r').read().strip()
        files = files.split('\n') if len(files) else []
        run_result = [0, 0, 0, 0]
        cov_result = [0, 0, 0, 0]
        for src in files:
            run_result[int(src[0])] += 1
            cov_result[int(src[2])] += 1
        self.stats['time'] = int(time.time())
        self.stats['path'] += (cov_result[1] + cov_result[2])
        self.stats['edge'] += cov_result[2]
        self.stats['timeout'] += run_result[1]
        self.stats['crash'] += run_result[2]
        self.stats['coverage'] = coverage.shape[0]
        self.stats['executed'] += num_found

        log.info('path(+{}), edge(+{}), timeout(+{}), crash(+{})'.format(
            cf.bold_white(cov_result[1] + cov_result[2]),
            cf.bold_green(cov_result[2]),
            cf.bold_yellow(run_result[1]),
            cf.bold_red(run_result[2])
        ))
        line = ','.join(map(str, self.stats.values()))
        plot_fd = open(self.files['plot'], 'a+')
        plot_fd.write('%s\n' % line)
        plot_fd.close()

        return cov_result

    def run_havoc(self, trace_files):
        tmp_dir = self.dirs['tmp']
        topk_dir = self.dirs['topk']

        rmtree(tmp_dir)
        os.mkdir(tmp_dir)

        in_files = np.array(list(map(lambda x: '%s/%s' % (topk_dir, x.split('/')[-1]), trace_files)))
        for in_file in in_files:
            dest = '{}/{}'.format(tmp_dir, in_file.split('/')[-1])
            copyfile(in_file, dest)
        return self.run_verifier(tmp_dir, 'rmu')

    def logspace(self, mi, ma):
        ranges = []
        idx = 0
        diff = ma - mi
        while True:
            fr = 0 if not idx else 2 ** idx
            to = 2 ** (idx + 1)
            if to > diff: to = diff
            if fr >= diff: break
            ranges.append((mi + fr, mi + to))
            idx += 1
        return ranges

    def start(self):
        trace_dir = self.dirs['trace']
        in_dir = self.dirs['in']
        topk_dir = self.dirs['topk']
        tmp_dir = self.dirs['tmp']
        bitmap = self.props['bitmap']

        num_found = self.run_verifier(in_dir, 'rm')
        self.move_testcases()
        self.collect_stats(num_found)

        m = Mutator()
        train_selector = {
            'coverage': m.train_with_coverage_bitmap,
            'justmiss': m.train_with_justmiss_bitmap,
            'neuzz': m.train_with_neuzz_bitmap,
        }

        log.info('bitmap({})'.format(cf.bold_white(bitmap)))
        skip = 0 

        while True:
            skip += self.run_verifier(topk_dir, 'te', skip)
            trace_files = np.array(sorted(glob.glob('%s/id:*' % trace_dir)))

            fast_files = m.filter_by_missed_edges(trace_files)
            slow_files = np.setdiff1d(trace_files, fast_files)

            sources = list(map(lambda x : x.split(':')[1].split(',')[0], fast_files))
            grads, in_bytes, in_sizes = train_selector[bitmap](fast_files)

            # det
            for mut_range in self.logspace(0, in_sizes.max()):
                if mut_range[0] < in_sizes.max():
                    m.mutate(grads, in_bytes, in_sizes, sources, mut_range)
                    num_found = self.run_verifier(tmp_dir, 'rm')
                    self.move_testcases()
                    self.collect_stats(num_found)
            # havoc

            num_found = self.run_havoc(fast_files)
            self.move_testcases()
            num_edges = self.collect_stats(num_found)[2]

            for _ in range(round(np.log2(num_edges + 1))):
                num_found = self.run_havoc(fast_files)
                self.move_testcases()
                self.collect_stats(num_found)
