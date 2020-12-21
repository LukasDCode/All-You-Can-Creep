import multiprocessing as mp

class Domain:

    def param_dict(self):
        example  = [
            ("nme", min, max),  
            ("name", min, max),  
        ]
        return 

    def create_task_runner(self, is_slurm, params):
        return self.create_slurm_runner(params) if is_slurm else self.create_local_runner(params)
        # returns TaskRunner
    
    def create_slurm_runner(self, params):
        return SlurmRunner(self, params)

    def create_local_runner(self, params):
        return DirectRunner(self, params)

    def run(self,params):
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

    def run(self):
        """executes one run with specific parameters
        returns rewards [float] 
        """
        pass

class DirectRunner(TaskRunner):

    def __init__(self, env_spec, params):
        self.env_spec = env_spec
        self.params = params
    
    def run(self):
        return self.env_spec.run(self.params)


class SlurmRunner(TaskRunner):

    def __init__(self, env_spec, params):
        self.params = params
        self.env_spec = env_spec
        
    def run(self):
        command = self.env_spec.python_run_command
        # TODO
        # start slurm job on remote machine
        # poll slurm job on remote machine
        # parse 

class Executor:

    def __init__(self, on_slurm, tasks_in_parallel, domain):
        self.tasks_in_parallel = tasks_in_parallel
        self.pool = mp.Pool(tasks_in_parallel)
        self.domain = domain
        self.on_slurm = on_slurm

    def submit_task(self, params):
        return self.pool.apply_async(
            self.domain.create_task_runner(is_slurm=self.on_slurm, params=params).run
        )

    def finalize(self,):
        self.pool.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()

    def __enter__(self):
        return self



