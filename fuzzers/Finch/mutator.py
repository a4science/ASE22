import numpy as np
import glob 
import os
import torch
import time
import math
import random
import sys
from shutil import rmtree
from net import Net, DistLoss
from tqdm import tqdm
from settings import log, bar_format, Settings

class Mutator(Settings):

    def __init__(self):
        super().__init__()

    def compute_grads(self, in_norms, loss_norms, in_shape, loss_shape):

        bitmap = self.props['bitmap']
        epoches = self.props['epoches']
        gpu = self.props['gpu']
        device = torch.device('cuda:%s' % gpu)
        net = Net(in_norms[0].shape[1], loss_norms[0].shape[1]).double().to(device)
        loss_fn = DistLoss() if bitmap == 'justmiss' else torch.nn.BCELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

        log.info('train {} => {}'.format(in_shape, loss_shape))

        for epoch in tqdm(range(epoches), bar_format=bar_format):
            for in_norm, loss_norm in zip(in_norms, loss_norms):
                xs = torch.tensor(in_norm, device=device)
                ys = torch.tensor(loss_norm, device=device)
                y_pred = net(xs)
                loss = loss_fn(y_pred, ys)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        grads = np.zeros(in_shape)
        fr = 0
        for in_norm, loss_norm in zip(in_norms, loss_norms):
            xs = torch.tensor(in_norm, requires_grad = True, device=device)
            ys = torch.tensor(loss_norm, device=device)
            y_pred = net(xs)
            loss = loss_fn(y_pred, ys)
            optimizer.zero_grad()
            loss.backward()
            grad = xs.grad.cpu().numpy()
            grads[fr : fr + grad.shape[0]] = grad
            fr += grad.shape[0]

        return grads 

    def filter_by_missed_edges(self, trace_files):
        map_file = self.files['map']
        trackers = np.zeros((1 << 16, 20000), dtype=np.uint8)
        max_u32 = (1 << 32) - 1

        visited_map = np.zeros(1 << 16, dtype=np.uint8)
        visible_map = np.frombuffer(open(map_file, 'rb').read(), dtype=np.uint32, offset=(1 << 17))

        for idx, trace_file in enumerate(trace_files):
            # read coverage and distances
            tracebit = np.frombuffer(open(trace_file, 'rb').read(), dtype=np.uint8, count=(1 << 16))
            visited_map = visited_map | tracebit
            distance = np.frombuffer(open(trace_file, 'rb').read(), dtype=np.uint32, offset=(1 << 16))
            # save pos of trace_file
            edges = np.where((distance == visible_map) & (distance != max_u32))[0]
            trackers[edges, idx] = 1

        visited_edges = np.where(visited_map != 0)[0]
        visible_edges = np.where(visible_map != max_u32)[0]
        missed_edges = np.setdiff1d(visible_edges, visited_edges)

        # find minimal set of trace files touching missed branches
        min_indices = []
        virgin_map = np.zeros(missed_edges.shape, dtype=np.uint8)
        tracker_map = trackers[missed_edges].sum(axis = 0)
        for idx in tracker_map.argsort()[::-1]:
            if not tracker_map[idx]: break
            tmp = virgin_map.sum()
            virgin_map |= trackers[missed_edges,idx]
            if virgin_map.sum() > tmp: min_indices.append(idx)
        return trace_files[min_indices]

    def train_with_justmiss_bitmap(self, trace_files):
        batch_size = self.props['batch_size']
        topk_dir = self.dirs['topk']
        map_file = self.files['map']
        max_u32 = (1 << 32) - 1

        in_files = np.array(list(map(lambda x: '%s/%s' % (topk_dir, x.split('/')[-1]), trace_files)))
        in_sizes = np.array(list(map(lambda x: os.path.getsize(x), in_files)))
        in_bytes = np.zeros((in_files.shape[0], in_sizes.max()), dtype=np.uint8)

        tracebits = np.zeros((len(trace_files), 1 << 16), dtype=np.uint8)
        distances = np.zeros((len(trace_files), 1 << 16), dtype=np.uint32)

        visited_map = np.zeros(1 << 16, dtype=np.uint8)
        visible_map = np.full(1 << 16, max_u32, dtype=np.uint32)

        for idx, (in_file, trace_file) in enumerate(zip(in_files, trace_files)):
            # read coverage and distances
            tracebits[idx] = np.frombuffer(open(trace_file, 'rb').read(), dtype=np.uint8, count=(1 << 16))
            visited_map = visited_map | tracebits[idx]
            distances[idx] = np.frombuffer(open(trace_file, 'rb').read(), dtype=np.uint32, offset=(1 << 16))
            visible_map = visible_map & distances[idx]
            # read test inputs
            in_byte = np.frombuffer(open(in_file, 'rb').read(), dtype=np.uint8)
            in_bytes[idx][:in_byte.shape[0]] = in_byte

        # compute visited and missed edges
        visited_edges = np.where(visited_map != 0)[0]
        visible_edges = np.where(visible_map != max_u32)[0]
        missed_edges = np.setdiff1d(visible_edges, visited_edges)
        missed_distances = distances[:,missed_edges]

        # TODO: should normalize in a different way
        # normalize data
        in_norms = np.array_split(in_bytes / 255.0, math.ceil(in_bytes.shape[0] / batch_size))
        loss_norms = np.array_split(missed_distances / max_u32, math.ceil(missed_distances.shape[0] / batch_size))

        return self.compute_grads(in_norms, loss_norms, in_bytes.shape, missed_distances.shape), in_bytes, in_sizes

    def train_with_neuzz_bitmap(self, trace_files):
        batch_size = self.props['batch_size']
        topk_dir = self.dirs['topk']
        map_file = self.files['map']
        max_u32 = (1 << 32) - 1

        in_files = np.array(list(map(lambda x: '%s/%s' % (topk_dir, x.split('/')[-1]), trace_files)))
        in_sizes = np.array(list(map(lambda x: os.path.getsize(x), in_files)))
        in_bytes = np.zeros((in_files.shape[0], in_sizes.max()), dtype=np.uint8)

        tracebits = np.zeros((len(trace_files), 1 << 16), dtype=np.uint8)
        distances = np.zeros((len(trace_files), 1 << 16), dtype=np.uint32)

        visited_map = np.zeros(1 << 16, dtype=np.uint8)

        for idx, (in_file, trace_file) in enumerate(zip(in_files, trace_files)):
            # read coverage and distances
            tracebits[idx] = np.frombuffer(open(trace_file, 'rb').read(), dtype=np.uint8, count=(1 << 16))
            visited_map = visited_map | tracebits[idx]
            # read test inputs
            in_byte = np.frombuffer(open(in_file, 'rb').read(), dtype=np.uint8)
            in_bytes[idx][:in_byte.shape[0]] = in_byte

        # compute visited edges
        visited_edges = np.where(visited_map != 0)[0]

        # normalize data
        visited_tracebits = tracebits[:, visited_edges]
        visited_tracebits[visited_tracebits > 1] = 1
        in_norms = np.array_split(in_bytes / 255.0, math.ceil(in_bytes.shape[0] / batch_size))
        loss_norms = np.array_split(visited_tracebits / 1.0, math.ceil(visited_tracebits.shape[0] / batch_size))

        return self.compute_grads(in_norms, loss_norms, in_bytes.shape, visited_tracebits.shape), in_bytes, in_sizes

    def train_with_coverage_bitmap(self, trace_files):

        batch_size = self.props['batch_size']
        topk_dir = self.dirs['topk']
        map_file = self.files['map']
        max_u32 = (1 << 32) - 1

        in_files = np.array(list(map(lambda x: '%s/%s' % (topk_dir, x.split('/')[-1]), trace_files)))
        in_sizes = np.array(list(map(lambda x: os.path.getsize(x), in_files)))
        in_bytes = np.zeros((in_files.shape[0], in_sizes.max()), dtype=np.uint8)

        tracebits = np.zeros((len(trace_files), 1 << 16), dtype=np.uint8)
        distances = np.zeros((len(trace_files), 1 << 16), dtype=np.uint32)

        visited_map = np.zeros(1 << 16, dtype=np.uint8)
        visible_map = np.full(1 << 16, max_u32, dtype=np.uint32)

        for idx, (in_file, trace_file) in enumerate(zip(in_files, trace_files)):
            # read coverage and distances
            tracebits[idx] = np.frombuffer(open(trace_file, 'rb').read(), dtype=np.uint8, count=(1 << 16))
            visited_map = visited_map | tracebits[idx]
            distances[idx] = np.frombuffer(open(trace_file, 'rb').read(), dtype=np.uint32, offset=(1 << 16))
            visible_map = visible_map & distances[idx]
            # read test inputs
            in_byte = np.frombuffer(open(in_file, 'rb').read(), dtype=np.uint8)
            in_bytes[idx][:in_byte.shape[0]] = in_byte

        # compute visited edges
        visited_edges = np.where(visited_map != 0)[0]
        visible_edges = np.where(visible_map != max_u32)[0]
        visited_edges = np.intersect1d(visible_edges, visited_edges)

        # normalize data
        visited_tracebits = tracebits[:, visited_edges]
        visited_tracebits[visited_tracebits > 1] = 1
        in_norms = np.array_split(in_bytes / 255.0, math.ceil(in_bytes.shape[0] / batch_size))
        loss_norms = np.array_split(visited_tracebits / 1.0, math.ceil(visited_tracebits.shape[0] / batch_size))

        return self.compute_grads(in_norms, loss_norms, in_bytes.shape, visited_tracebits.shape), in_bytes, in_sizes

    def mutate(self, grads, in_bytes, in_sizes, sources, mut_range):

        total_created = 0
        tmp_dir = self.dirs['tmp']
        rmtree(tmp_dir)
        os.mkdir(tmp_dir)

        triplet = list(zip(grads, in_bytes, in_sizes, sources))
        log.info('mutate [%d:%d]' % (mut_range[0], mut_range[1]))

        for idx in tqdm(range(len(triplet)), bar_format=bar_format):
            grad, in_byte, in_size, source = triplet[idx]
            topk = np.abs(grad[:in_size]).argsort()[::-1][mut_range[0]: mut_range[1]]
            sign = np.sign(grad[topk])
            tmp = in_byte[topk].copy()

            # compute up and down steps
            steps = [0, 0, 0, 0]
            positive_locs = np.where(sign == 1)[0]
            negative_locs = np.where(sign == -1)[0]
            if len(positive_locs):
                steps[0] = 255 - tmp[positive_locs].min()
                steps[2] = tmp[positive_locs].max()
            if len(negative_locs):
                steps[1] = tmp[negative_locs].max()
                steps[3] = 255 - tmp[negative_locs].min()
            up_step = max(steps[0], steps[1])
            low_step = max(steps[2], steps[3])

            # go-up
            for step in range(up_step):
                val = tmp + (step + 1) * sign
                val[val > 255] = 255
                val[val < 0] = 0
                in_byte[topk] = val.astype(np.uint8)
                in_byte[:in_size].tofile('%s/id:%08d,src:%s,op:grad,step:+%d,range:[%d:%d]' % (
                    tmp_dir, total_created, source, step + 1, mut_range[0], mut_range[1]
                ))
                total_created += 1

            # go-down
            for step in range(low_step):
                val = tmp - (step + 1) * sign
                val[val > 255] = 255
                val[val < 0] = 0
                in_byte[topk] = val.astype(np.uint8)
                in_byte[:in_size].tofile('%s/id:%08d,src:%s,op:grad,step:-%d,range:[%d:%d]' % (
                    tmp_dir, total_created, source, step + 1, mut_range[0], mut_range[1]
                ))
                total_created += 1

            # reset
            in_byte[topk] = tmp
