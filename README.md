# All You Can Creep

[14.12.2020 Presentation KickOff](https://docs.google.com/presentation/d/1Xw14hQdzAnOwRLO7TBfw0uGq7R6cbRojAYQVoDM4pPU/edit#slide=id.g7871c53ed9_0_0)

# Links

[Examples](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md)

[Installation](https://github.com/Unity-Technologies/ml-agents/blob/release_10_docs/docs/Installation.md)

[ML Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Python-API.md)

[Gym Wrapper](https://github.com/Unity-Technologies/ml-agents/blob/master/gym-unity/README.md)

[Environment Exec](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Executable.md)

# Worm Environment

## Observation Space

\| Index         \| Purpose            \| Body Part  \| Note \|
\| ------------- \|:------------------:\| ----------:\| --- \|
\| 0             \| distance to front  \| whole Worm \| measuring distance to the next object straight ahead of the worm, max is 10 \|
\| 1             \| speed to target    \| whole Worm \| - \|
\| 2             \| rotation of orientation cube \| whole Worm \| in degree divided by 180 \|
\| 3             \| "from-to-rotation" \| whole Worm \| - \|
\| 4             \| distance to target \| whole Worm \| - \|
\| ------------- \|:------------------:\| ----------:\|---:\|
\| 5             \| touching ground    \| body segment or joint \| boolean if this body part is touching the ground or not \|
\| 6             \| speed of body part \| body segment or joint\| - \|
\| 7             \| ankle speed        \| body segment or joint \| - \|
\| 8             \| inverse transform direction \| body segment \| - \|
\| 9             \| local rotation \| body segment \| - \|
\| 10             \| ankle of joint \| joint \| - \|

Indices 5 to 10 are repeated for every body part or joint respectively.
Further info can be taken from the "WormAgent.cs" located in "/ml-agents/Project/Assets/ML-Agents/Examples/Worm/Scripts/WormAgent.cs"
