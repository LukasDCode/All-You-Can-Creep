import time
import numpy as np
import torch as T
from math import trunc
from multiprocessing import Pool
"""

Reward_arr: [0. 0. 0. ... 0. 0. 0.]
Dones_arr: [False False False ... False False  True]
Values: [-0.15783533 -0.33105567 -0.27574265 ... -0.21850853 -0.18411195
 -0.17393553]

 Alle haben Länge 10.000 !!!

"""

gamma=0.99
gae_lambda=0.95
device = T.device('cpu')

# get the starting time to measure how long the reward-calculation takes
start_time = time.perf_counter_ns()

reward_arr = [ 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
advantage = np.zeros(len(reward_arr), dtype=np.float32)
dones_arr = [False, False, False, False, True, False, False, False, True, True]
values = [-0.15783533, -0.33105567, -0.27574265, -0.15783533, -0.18411195, -0.33105567, -0.27574265, -0.21850853, -0.18411195, -0.17393553]

term_arr_old = []
for t in range(len(reward_arr)-1): # länge 10.000 
    discount = 1
    a_t = 0
    for k in range(t, len(reward_arr)-1):
        a_t += discount*(reward_arr[k] + gamma*values[k+1]*\
                (1-int(dones_arr[k])) - values[k])
        term_arr_old.append(discount*(reward_arr[k] + gamma*values[k+1]*(1-int(dones_arr[k])) - values[k]))
        discount *= gamma*gae_lambda
    advantage[t] = a_t
#advantage = T.tensor(advantage).to(device)

old_reward_arr = reward_arr
old_adv_arr = advantage
old_dones_arr = dones_arr
old_values = values

end_time = time.perf_counter_ns()
actual_time = end_time-start_time
time_minutes = int(actual_time/(60*1e+9))
time_seconds = trunc((actual_time/1e+9)%60)
time_nanosecs = actual_time%1e+9
print("Calculating the rewards took", time_minutes, "Minutes and", time_seconds, "Seconds", actual_time, "Nanoseconds")

# get the starting time to measure how long the reward-calculation takes
start_time = time.perf_counter_ns()

reward_arr = [ 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
advantage = np.zeros(len(reward_arr), dtype=np.float32)
dones_arr = [False, False, False, False, True, False, False, False, True, True]
values = [-0.15783533, -0.33105567, -0.27574265, -0.15783533, -0.18411195, -0.33105567, -0.27574265, -0.21850853, -0.18411195, -0.17393553]

#discount_array
discount = 1
discount_arr = [1]
for i in range(len(reward_arr)-2):
    discount *= gamma*gae_lambda
    discount_arr.append(discount)

print("discount_arr :",discount_arr)
    
#term_array
term_arr = []
for i in range(len(reward_arr)-1):
    term_arr.append(reward_arr[i] + gamma*values[i+1]*(1-int(dones_arr[i])) - values[i])
    
print("Länge Term Array:", len(term_arr))
print("Länge Discount Array:", len(discount_arr))
# print("Länge Advantage Array:", len(adv_arr))

#advantage_array
adv_arr = []
for k in range(len(discount_arr)):
    if len(adv_arr) == 0:
        adv_arr.append(discount_arr[-(k+1)] * term_arr[-(k+1)])
        print("FIRST RUN")
    elif k == len(discount_arr)-1:
        adv_arr.append(discount_arr[-(k+1)] * term_arr[-(k+1)])
        print("LAST RUN")
    else:
        adv_arr.append(discount_arr[-(k+1)] * term_arr[-(k+1)] + adv_arr[k-1])
        print("SOMETHING ELSE")
# adv_arr = adv_arr.reverse()
    
end_time = time.perf_counter_ns()
actual_time = end_time-start_time
time_minutes = int(actual_time/(60*1e+9))
time_seconds = trunc((actual_time/1e+9)%60)
time_nanosecs = actual_time%1e+9
print("Calculating the rewards took", time_minutes, "Minutes and", time_seconds, "Seconds", actual_time, "Nanoseconds")

print("AFTER OUR REWARD CALCULATION")
#print("Reward_arr:    ", reward_arr) # --> Array of number
#print("Old Reward_arr:", old_reward_arr) # --> Array of number
#print("Dones_arr:    ", dones_arr) # --> Array of booleans
#print("Old Dones_arr:", old_dones_arr) # --> Array of booleans
#print("Values:    ", values) # --> Array of numbers
#print("Old Values:", old_values) # --> Array of numbers
print("Advantage:    ", adv_arr)
print("Old Advantage:", old_adv_arr)

print("term_arr_old:     ", term_arr_old)
print("term_arr:         ", term_arr)

term_arr_old_calc = []
offset = 0
for i in range(len(term_arr)):
    intermediate_step = 0
    for j in range(i+1):
        calc = j+i+offset
        intermediate_step += term_arr_old[calc]
    offset+=i
    term_arr_old_calc.append(intermediate_step)


print("term_arr_old_calc:", term_arr_old_calc)

# [0, 1,   2,     3,       4]
# [0, 1,2, 3,4,5, 6,7,8,9, 10,11,12,13,14]
 


#t_0    xxxx d=1               a_t_0 = 1*TERM[0] + g*l*1 * TERM[1] + g*l*g*l*1 * TERM[2] + g*l*g*l*g*l*1 * TERM[3]
#t_1     xxx d=g*l*1           a_t_1 = g*l*1 * TERM[1] + g*l*g*l*1 * TERM[2] + g*l*g*l*g*l*1 * TERM[3]
#t_2      xx d=g*l*g*l*1       a_t_2 = g*l*g*l*1 * TERM[2] + g*l*g*l*g*l*1 * TERM[3]
#t_3       x d=g*l*g*l*g*l*1   a_t_3 = g*l*g*l*g*l*1 * TERM[3]

# adv [a_t_0,  a_t_1,  a_t_2,  a_t_3]



'''
discounted_returns = []
R = 0
for reward in reversed(rewards):
    R = reward + gamma*R
    discounted_returns.append(R)
discounted_returns.reverse()
'''



'''
discounted_returns = []
R = 0
for reward in reversed(rewards):
    R = reward + gamma*R
    discounted_returns.append(R)
discounted_returns.reverse()
'''






