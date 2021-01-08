# Tracking params, metrics and artifacts with mlflow

https://mlflow.org/docs/latest/tracking.html#


## Configuring mlflow
To run runs with mlflow tracking the following env variables need to be set.
Get the information from asp@bschuess.dev
`MLFLOW_TRACKING_URI=https://bschuess.dev/asp/mlflow/#/`
`MLFLOW_TRACKING_USERNAME=username`
`MLFLOW_TRACKING_PASSWORD=password`
The latter two contain sensitive information and should:
- not appear in the job name in slurm
- not be visible in the command history 
- we could write them into start.sh -> and make this script only visible to the user 

Furthermore you need sftp access to mlflow-ftp@bschuess.dev
Send you public rsa key to asp@bschuess.dev

## Using mlflow
```
import mlflow
mlflow.set_experiment("/my-experiment")
with mlflow.start_run():
    mlflow.set_tags("a2c",)
    mlflow.log_param("a", 1)
    mlflow.log_metric("b", 2)
    mlflow.log_artifact("filepath")

```


