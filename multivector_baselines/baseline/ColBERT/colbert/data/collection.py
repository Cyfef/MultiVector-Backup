
# Could be .tsv or .json. The latter always allows more customization via optional parameters.
# I think it could be worth doing some kind of parallel reads too, if the file exceeds 1 GiBs.
# Just need to use a datastructure that shares things across processes without too much pickling.
# I think multiprocessing.Manager can do that!

import os
import itertools

from colbert.evaluation.loaders import load_collection
from colbert.infra.run import Run
from pathlib import Path
import torch
class Collection:
    def __init__(self, path=None, data=None):
        self.path = path
        self.has_embedding = False
        self.data = data or self._load_file(path)
        

    def __iter__(self):
        # TODO: If __data isn't there, stream from disk!
        return self.data.__iter__()

    def __getitem__(self, item):
        # TODO: Load from disk the first time this is called. Unless self.data is already not None.
        return self.data[item]

    def __len__(self):
        # TODO: Load here too. Basically, let's make data a property function and, on first call, either load or get __data.
        return len(self.data)

    def _load_file(self, path):
        self.path = path
        if Path(path).is_dir():
            self.has_embedding = True
            return self.load_precomputed_embeddings(path)
        else:
            return self._load_tsv(path) if path.endswith('.tsv') else self._load_jsonl(path)

    def load_precomputed_embeddings(self, embeddings_folder):
        embedding_file_count = 0
        for filename in os.listdir(embeddings_folder):
            if filename.endswith('.pt') and filename.startswith('embeddings.'):
                embedding_file_count += 1
        
        data_ls = []
        doclen_ls = []
        # embedding_file_count=10
        for idx in range(embedding_file_count):
            embedding_path = os.path.join(embeddings_folder, f'embeddings.{idx}.pt')
            if os.path.exists(embedding_path):
                embedding, doclens = torch.load(embedding_path, map_location='cpu')
                offset=0
                for i in range(len(doclens)):
                    data_ls.append(embedding[offset:offset + doclens[i]])
                    offset += doclens[i]
                doclen_ls.extend(doclens)
                # data_ls.append((embedding, doclens))
        # print("doclen_ls::", doclen_ls)
        self.doclen_ls = doclen_ls
        return data_ls
    
    def load_precomputed_embeddings2(self, embeddings_folder):
        doc_count = torch.load(os.path.join(embeddings_folder, 'doc_count'), map_location='cpu')
        return torch.arange(doc_count)
        
        
    def _load_tsv(self, path):
        return load_collection(path)

    def _load_jsonl(self, path):
        raise NotImplementedError()

    def provenance(self):
        return self.path
    
    def toDict(self):
        return {'provenance': self.provenance()}

    def save(self, new_path):
        assert new_path.endswith('.tsv'), "TODO: Support .json[l] too."
        assert not os.path.exists(new_path), new_path

        with Run().open(new_path, 'w') as f:
            # TODO: expects content to always be a string here; no separate title!
            for pid, content in enumerate(self.data):
                content = f'{pid}\t{content}\n'
                f.write(content)
            
            return f.name

    def enumerate(self, rank):
        if not self.has_embedding:
            for _, offset, passages in self.enumerate_batches(rank=rank):
                for idx, passage in enumerate(passages):
                    yield (offset + idx, passage)
        else:
            for _, offset, data in self.enumerate_batches2(rank=rank):
                for idx in range(len(data)):
                    # Assuming data is a list of tuples (embedding, doclens)
                    # where embedding is a tensor and doclens is a list of lengths.
                    # Here we yield the index and the corresponding passage.
                    yield (offset+idx, data[idx])  # Assuming data is a list of passages

    def enumerate_batches(self, rank, chunksize=None):
        assert rank is not None, "TODO: Add support for the rank=None case."

        chunksize = chunksize or self.get_chunksize()

        offset = 0
        iterator = iter(self)

        for chunk_idx, owner in enumerate(itertools.cycle(range(Run().nranks))):
            L = [line for _, line in zip(range(chunksize), iterator)]

            if len(L) > 0 and owner == rank:
                yield (chunk_idx, offset, L)

            offset += len(L)

            if len(L) < chunksize:
                return
            
    def enumerate_batches2(self, rank, chunksize=None):
        assert rank is not None, "TODO: Add support for the rank=None case."
        
        offset = 0
        # for idx, data in iter(self.data):
        #     yield (idx, offset, data)  # Assuming data is a list of passages
        #     offset += len(data[0])

        chunksize = chunksize or self.get_chunksize()

        offset = 0
        iterator = iter(self)

        for chunk_idx, owner in enumerate(itertools.cycle(range(Run().nranks))):
            L = [line for _, line in zip(range(chunksize), iterator)]

            if len(L) > 0 and owner == rank:
                yield (chunk_idx, offset, L)

            offset += len(L)

            if len(L) < chunksize:
                return
    
    def get_chunksize(self):
        # return min(25_000, 1 + len(self) // Run().nranks)  # 25k is great, 10k allows things to reside on GPU??
        return min(2500, 1 + len(self) // Run().nranks)  # 25k is great, 10k allows things to reside on GPU??

    @classmethod
    def cast(cls, obj):
        if type(obj) is str:
            return cls(path=obj)

        if type(obj) is list:
            return cls(data=obj)

        if type(obj) is cls:
            return obj

        assert False, f"obj has type {type(obj)} which is not compatible with cast()"


# TODO: Look up path in some global [per-thread or thread-safe] list.
