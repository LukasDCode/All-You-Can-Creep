import multiprocessing as mp
from uuid import uuid4
from queue import Queue
import time

from ..utils.color import bcolors

class Runner:
    def run(
        self,
        worker_id : int,
        run_id : str,
        agent_class,
        **kwargs):
        pass



DEFAULT_PARALLEL=1
DEFAULT_BASE_ID=0

class Executor:

    """Adds executor params to argpasr config parser"""
    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('-p', '--parallel', type=int, default=DEFAULT_PARALLEL, help='level on parallization')
        parser.add_argument('-w', '--worker_base_id', type=int, default=DEFAULT_BASE_ID, help='offset of workers')
        return parser

    def __init__(
        self,
        runner,
        parallel=DEFAULT_PARALLEL,
        worker_base_id=DEFAULT_BASE_ID,
        **kwargs,
        ):
        self.tasks_in_parallel = parallel
        self.worker_base_id = worker_base_id
        self.worker_failed_counter = 0
        self.runner = runner

        self.tokens = Queue()
        for x in range(0+self.worker_base_id,self.tasks_in_parallel + self.worker_base_id):
          self.tokens.put(x)

    def free_token_with_delay(self, worker_id, delay=5.):
        print(f"{bcolors.WARNING}Delaying free token {worker_id}{bcolors.ENDC}")
        time.sleep(delay)
        self.tokens.put(worker_id)
    def error_callback(self,error, worker_id, run_id, delay=5.):
        print(f"{bcolors.FAIL}worker {worker_id} with run {run_id} failed: {error} {bcolors.ENDC}")
        self.worker_failed_counter += 1
        self.free_token_with_delay(self.worker_base_id + self.tasks_in_parallel + self.worker_failed_counter, delay)

    """Asynchronoulsy executes runner.run with the given **kwargs, may block if poolsize is maxed out.
    @Returns a future
    """
    def submit_task(self, run_id=str(uuid4()), **kwargs):
        worker_id = self.tokens.get(block=True)
        print(f"{bcolors.WARNING}Reserved worker id: {worker_id}{bcolors.ENDC}")
        return self.pool.apply_async(
            self.runner.run,
            kwds={**kwargs, "worker_id": worker_id, "run_id": run_id, },
            callback= lambda x: {self.free_token_with_delay(worker_id)},
            error_callback=lambda x: {self.error_callback(x, worker_id,run_id)},
        )

    def finalize(self,):
        self.pool.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()

    def __enter__(self):
        self.pool = mp.Pool(self.tasks_in_parallel)
        return self



