import multiprocessing as mp
from uuid import uuid4
from queue import Queue

class Domain:

    def param_dict(self):
        example  = [
            ("param_name0", min, max),  
            ("param_name1", min, max),  
        ]
        return 

    def run(self, worker_id, run_id, **kwargs):
        pass

class Executor:

    """Adds executor params to argpasr config parser"""
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('-p', '--parallel',type=int, default=1, help='level on parallization')
        parser.add_argument('-w', '--worker_base_id', type=int, default=0, help='offset of workers')
        return parser

    def __init__(self, config, domain):
        self.tasks_in_parallel = config.parallel
        self.worker_base_id = config.worker_base_id
        self.domain = domain

        self.tokens = Queue()
        for x in range(0+self.worker_base_id,self.tasks_in_parallel + self.worker_base_id):
          self.tokens.put(x)

    """Asynchronoulsy executes domain.run with the given **kwargs, may block if poolsize is maxed out.
    @Returns a future
    """
    def submit_task(self, **kwargs):
        worker_id = self.tokens.get(block=True)
        return self.pool.apply_async(
            self.domain.run,
            kwds={**kwargs, "worker_id": worker_id, "run_id": str(uuid4()), },
            callback= lambda x: self.tokens.put(worker_id) ,
            error_callback=lambda x: self.tokens.put(worker_id),
        )

    def finalize(self,):
        self.pool.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()

    def __enter__(self):
        self.pool = mp.Pool(self.tasks_in_parallel)
        return self



