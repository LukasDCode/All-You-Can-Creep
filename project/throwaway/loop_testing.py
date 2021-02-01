import numpy as np

'''
for t in range(len(reward_arr)-1):                                  #t = 0 1 ... 8
    discount = 1
    a_t = 0
    for k in range(t, len(reward_arr)-1):                           #k = 
                                                                    # k_0 = 0 1 ... 8
                                                                    # k_1 = 1 ... 8

        a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                (1-int(dones_arr[k])) - values[k])
        discount *= self.gamma*self.gae_lambda
    advantage[t] = a_t
    
print(advantage)

for i in range(len(reward_arr)-2):
    d[i+1] = d[i] * gamma * gae_lambda

print(d)
'''

reward_arr = [ 1.0, 2.0, 3.0, 4.0]#, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
advantage = np.zeros(len(reward_arr), dtype=np.float32)
dones_arr = [False, True, False, False]#, False, True, False, False, False, False]
values = [-0.15783533, -0.33105567, -0.27574265, -0.15783533]#, -0.18411195, -0.33105567, -0.27574265, -0.21850853, -0.18411195, -0.17393553]

gamma=0.99
gae_lambda=0.95


'''########################################################################################################################################
a = np.zeros(len(reward_arr), dtype=np.float32)

for t in range(len(reward_arr)-1):
    discount = 1
    a_t = 0
    for k in range(t, len(reward_arr)-1):
        a_t += reward_arr[k] + gamma*values[k+1]*(1-int(dones_arr[k])) - values[k]
    advantage[t] = a_t

print(advantage)

for i in range(len(reward_arr)-1):

    # a[-(i+2)] = a[-(i+1)] + d[-(i+1)] * ( reward_arr[-(i+2)] + gamma*values[-(i+1)]*(1-int(dones_arr[-(i+2)])) - values[-(i+2)] )
    a[-(i+2)] = a[-(i+1)] + reward_arr[-(i+2)] + gamma*values[-(i+1)]*(1-int(dones_arr[-(i+2)])) - values[-(i+2)]

print(a)
########################################################################################################################################

d = np.ones(len(reward_arr), dtype=np.float32)

for i in range(1,len(reward_arr)-1):

    d[i] = d[i-1] * gamma * gae_lambda

#print(d)
#[1.         0.9405     0.88454026 0.83191013 0.78241146 0.73585796
# 0.6920744  0.650896   0.6121677  1.        ]

t: 0
  k: 0
    a_t     : 0
    discount: 1

    a_t += 1 *(reward_arr[ 0 ] + gamma*values[ 1 ]*(1-int(dones_arr[ 0 ])) - values[ 0 ]))

  k: 1
    a_t     : 0.8300902166999999
    discount: 0.9405

    a_t += 0.9405 *(reward_arr[ 1 ] + gamma*values[ 2 ]*(1-int(dones_arr[ 1 ])) - values[ 1 ]))

  k: 2
    a_t     : 3.022448074335
    discount: 0.88454025

    a_t += 0.88454025 *(reward_arr[ 2 ] + gamma*values[ 3 ]*(1-int(dones_arr[ 2 ])) - values[ 2 ]))

advantage[ 0 ] = 5.7817587116672

t: 1
  k: 1
    a_t     : 0
    discount: 1

    a_t += 1 *(reward_arr[ 1 ] + gamma*values[ 2 ]*(1-int(dones_arr[ 1 ])) - values[ 1 ]))

  k: 2
    a_t     : 2.33105567
    discount: 0.9405

    a_t += 0.9405 *(reward_arr[ 2 ] + gamma*values[ 3 ]*(1-int(dones_arr[ 2 ])) - values[ 2 ]))

advantage[ 1 ] = 5.26493194573865

t: 2
  k: 2
    a_t     : 0
    discount: 1

    a_t += 1 *(reward_arr[ 2 ] + gamma*values[ 3 ]*(1-int(dones_arr[ 2 ])) - values[ 2 ]))

advantage[ 2 ] = 3.1194856733

[5.781759  5.264932  3.1194856 0.       ]
[6.1847897 5.4505415 3.1194856 0.       ]

########################################################################################################################################'''

for t in range(len(reward_arr)-1): 
    print()                                 #t = 0 1 ... 8
    print("t:",t)

    discount = 1
    a_t = 0

    for k in range(t, len(reward_arr)-1):                           # k = 
        print("  k:",k)                                             # k_0 = 0 1 ... 8
                                                                    # k_1 = 1 ... 8
        print("    a_t     :",a_t)
        print("    discount:",discount)
        print()
        print("    a_t +=",discount,"*(reward_arr[",k,"] + gamma*values[",k+1,"]*(1-int(dones_arr[",k,"])) - values[",k,"]))")
        print()

        a_t += discount*(reward_arr[k] + gamma*values[k+1]*(1-int(dones_arr[k])) - values[k])
        discount *= gamma*gae_lambda
    
    print("advantage[",t,"] =", a_t)
    advantage[t] = a_t

print()    
print(advantage)

a = np.zeros(len(reward_arr), dtype=np.float32)

for i in range(len(reward_arr)-1):

    a[-(i+2)] = a[-(i+1)] * gamma * gae_lambda + reward_arr[-(i+2)] + gamma*values[-(i+1)]*(1-int(dones_arr[-(i+2)])) - values[-(i+2)]

print(a)

########################################################################################################################################

reward_arr = [ 1.0, 2.0, 3.0, 4.0]#, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
advantage = np.zeros(len(reward_arr), dtype=np.float32)
dones_arr = [False, True, False, False]#, False, True, False, False, False, False]
values = [-0.15783533, -0.33105567, -0.27574265, -0.15783533]#, -0.18411195, -0.33105567, -0.27574265, -0.21850853, -0.18411195, -0.17393553]







    




