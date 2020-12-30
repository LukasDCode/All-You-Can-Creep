import multiprocessing as mp
from queue import Queue

class Domain:

    def param_dict(self):
        example  = [
            ("nme", min, max),  
            ("name", min, max),  
        ]
        return 

    def create_task_runner(self, is_slurm, params):
        return self.create_slurm_runner(params) if is_slurm else self.create_local_runner(params=params)
        # returns TaskRunner
    
    def create_slurm_runner(self, params):
        return SlurmRunner(self, params)

    def create_local_runner(self, params):
        return DirectRunner(self, params)

    def run(self,worker_id, params):
        """Executes the run directly in memory
        returns rewards [float]
        """
        pass

    def python_run_command(self,params):
        """Specifies the command to be run by slurm within the repository"""
        pass

    def python_run_parse_log(self, logfilestring):
        """Parses the slurm log to yield the requested values
        returns rewards [float]
        """
        pass
        

class TaskRunner:

    def run(self, worker_id):
        """executes one run with specific parameters
        returns rewards [float] 
        """
        pass

class DirectRunner(TaskRunner):

    def __init__(self, env_spec, params):
        self.env_spec = env_spec
        self.params = params
    
    def run(self, worker_id):
        return self.env_spec.run(worker_id, self.params)


class SlurmRunner(TaskRunner):

    def __init__(self, env_spec, params):
        self.params = params
        self.env_spec = env_spec
        
    def run(self, worker_id):
        command = self.env_spec.python_run_command
        # TODO
        # start slurm job on remote machine
        # poll slurm job on remote machine
        # parse 

class Executor:

    """Adds executor params to argpasr config parser"""
    @staticmethod
    def add_parser_args(parser):
        return parser.add_argument('-p', '--parallel',type=int, default=1, help='level on parallization')


    def __init__(self, config, domain, on_slurm=False):
        self.tasks_in_parallel = config.parallel
        self.pool = mp.Pool(self.tasks_in_parallel)
        self.domain = domain
        self.on_slurm = on_slurm

        self.tokens = Queue()
        for x in range(0,self.tasks_in_parallel):
          self.tokens.put(x)

    def submit_task(self, params):
        worker_id = self.tokens.get(block=True)
        return self.pool.apply_async(
            self.domain.create_task_runner(is_slurm=self.on_slurm, params=params).run,
            kwds={"worker_id":worker_id},
            callback= lambda x: self.tokens.put(worker_id) ,
            error_callback=lambda x: self.tokens.put(worker_id),
        )

    def finalize(self,):
        self.pool.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()

    def __enter__(self):
        return self



