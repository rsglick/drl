import numpy as np
import matplotlib.pyplot as plt


#
# Equations of Motion
#  Euler Integration
# 
def euler( state, ctrl, dt ):
    pos_old = state[0].copy()
    vel_old = state[1].copy()
    acc_old = state[2].copy()        

    acc_new = acc_old + dt * (ctrl - acc_old)
    vel_new = vel_old + dt * acc_new
    pos_new = pos_old + dt * vel_new

    # return in state array
    new_state = np.empty_like(state)
    new_state[0] = pos_new.copy()
    new_state[1] = vel_new.copy()
    new_state[2] = acc_new.copy()

    return new_state

#
# Equations of Motion
#  RK4 Integration
# 
def rk4( state, ctrl, dt ):
    dt_half = 0.5 * dt

    k1 = np.zeros(1)
    k2 = np.zeros_like(k1)
    k3 = np.zeros_like(k1)
    k4 = np.zeros_like(k1)
    l1 = np.zeros_like(k1)
    l2 = np.zeros_like(k1)
    l3 = np.zeros_like(k1)
    l4 = np.zeros_like(k1)

    pos_old = state[0].copy()
    vel_old = state[1].copy()
    acc_old = state[2].copy()

    pos_new = np.zeros_like(pos_old)
    vel_new = np.zeros_like(vel_old)
    acc_new = np.zeros_like(vel_old)
    acc_tmp = np.zeros_like(k1)


    # Acceleration
    acc_new =  state[2] + dt * (ctrl - state[2]) # Accleration

    # Half k1 and l1
    k1 = dt_half * acc_new
    l1 = dt_half * vel_old

    # Half k2 and l2
    pos_new = pos_old + l1 
    vel_new = vel_old + k1
    acc_tmp = acc_new.copy()

    k2 = dt_half * acc_tmp
    l2 = dt_half * (vel_old + k1)

    # Half k3 and l3
    pos_new = pos_old + l2 
    vel_new = vel_old + k2
    acc_tmp = acc_new.copy()

    k3 = dt * acc_tmp
    l1 = dt * (vel_old + k2)

    # Half k4 and l4
    pos_new = pos_old + l3 
    vel_new = vel_old + k3
    acc_tmp = acc_new.copy()
    
    k4 = dt_half * acc_tmp
    l4 = dt_half * (vel_old + k3)

    # Complete RK
    vel_new = vel_old + ( k1 + 2.0 * k2 + k3 + k4 ) / 3.0 
    pos_new = pos_old + ( l1 + 2.0 * l2 + l3 + l4 ) / 3.0

    # return in state array
    new_state = np.empty_like(state)
    new_state[0] = pos_new.copy()
    new_state[1] = vel_new.copy()
    new_state[2] = acc_new.copy()

    return new_state

#
# Quickly test integration
#
def test_integration_1d(tfinal=10, n_axis=1):
    np.random.seed(0)

    tstart   = 0
    dt       = 0.01
    time     = np.arange(tstart, tfinal, dt)

    # Random Uniform Initial State
    pos = np.random.uniform(-1, 1, n_axis)
    vel = np.zeros_like(pos)
    acc = np.zeros_like(pos)
    initial_state = np.array( [pos, vel, acc] )
    print(f"initial_state = {initial_state}")
    
    tmp = []
    for i in range(len(time)):
        tmp.append(np.empty( (3,n_axis) ) )

    state_rk4    = tmp
    state_rk4[0] = initial_state.copy()
    state_eul    = tmp
    state_eul[0] = initial_state.copy()

    # Apply control command to entire array
    #ctrl_cmd = 5.0
    ctrl_cmd = np.ones( (n_axis,) )* 5.0
    tmp = []
    for i in range(len(time)):
        tmp.append( np.ones( (1,n_axis) ) )
    ctrl  = [i * ctrl_cmd for i in tmp]

    # Perform Integration
    for i, t in enumerate(time):
        if i == len(time)-1:
            continue
        state_rk4[i+1] = rk4(   state_rk4[i], ctrl[i], dt )
        state_eul[i+1] = euler( state_eul[i], ctrl[i], dt )

    state_rk4_pos_x = np.array([ state_rk4[:][i][0][0]  for i in range(len(time)) ])
    state_rk4_vel_x = np.array([ state_rk4[:][i][1][0]  for i in range(len(time)) ])
    state_rk4_acc_x = np.array([ state_rk4[:][i][2][0]  for i in range(len(time)) ])

    state_eul_pos_x = np.array([ state_eul[:][i][0][0]  for i in range(len(time)) ])
    state_eul_vel_x = np.array([ state_eul[:][i][1][0]  for i in range(len(time)) ])
    state_eul_acc_x = np.array([ state_eul[:][i][2][0]  for i in range(len(time)) ])

    ctrl_x = np.array([ ctrl[i][0][0]  for i in range(len(time)) ]   )

    # Plot to confirm working
    fig, ax = plt.subplots(3,1,sharex=True)
    fig.suptitle("1D")
    ax[0].set_title("X")
    ax[0].plot( time, state_rk4_pos_x, label="rk4")
    ax[0].plot( time, state_eul_pos_x, '--', label="eul")
    ax[0].set_ylabel("Position")
    ax[1].plot( time, state_rk4_vel_x, label="rk4")
    ax[1].plot( time, state_eul_vel_x, '--', label="eul")
    ax[1].set_ylabel("Velocity")
    ax[2].plot( time, state_rk4_acc_x, label="rk4")
    ax[2].plot( time, state_eul_acc_x, '--', label="eul")
    ax[2].plot( time, ctrl_x, '--')
    ax[2].set_ylabel("Acceleration")
    ax[2].set_xlabel("Time")
    ax[2].legend()
    #fig.savefig("test_integration_1d.png")




def test_integration_2d(tfinal=10, n_axis=2):
    np.random.seed(0)

    tstart   = 0
    dt       = 0.01
    time     = np.arange(tstart, tfinal, dt)

    # Random Uniform Initial State
    pos = np.random.uniform(-1, 1, n_axis)
    vel = np.zeros_like(pos)
    acc = np.zeros_like(pos)
    initial_state = np.array( [pos, vel, acc] )
    print(f"initial_state = {initial_state}")
    
    tmp = []
    for i in range(len(time)):
        tmp.append(np.empty( (3,n_axis) ) )

    state_rk4    = tmp
    state_rk4[0] = initial_state.copy()
    state_eul    = tmp
    state_eul[0] = initial_state.copy()

    # Apply control command to entire array
    #ctrl_cmd = 5.0
    ctrl_cmd = np.ones( (n_axis,) )* 5.0
    tmp = []
    for i in range(len(time)):
        tmp.append( np.ones( (1,n_axis) ) )
    ctrl  = [i * ctrl_cmd for i in tmp]

    # Perform Integration
    for i, t in enumerate(time):
        if i == len(time)-1:
            continue
        state_rk4[i+1] = rk4(   state_rk4[i], ctrl[i], dt )
        state_eul[i+1] = euler( state_eul[i], ctrl[i], dt )

    state_rk4_pos_x = np.array([ state_rk4[:][i][0][0]  for i in range(len(time)) ])
    state_rk4_pos_y = np.array([ state_rk4[:][i][0][1]  for i in range(len(time)) ])
    state_rk4_vel_x = np.array([ state_rk4[:][i][1][0]  for i in range(len(time)) ])
    state_rk4_vel_y = np.array([ state_rk4[:][i][1][1]  for i in range(len(time)) ])
    state_rk4_acc_x = np.array([ state_rk4[:][i][2][0]  for i in range(len(time)) ])
    state_rk4_acc_y = np.array([ state_rk4[:][i][2][1]  for i in range(len(time)) ])

    state_eul_pos_x = np.array([ state_eul[:][i][0][0]  for i in range(len(time)) ])
    state_eul_pos_y = np.array([ state_eul[:][i][0][1]  for i in range(len(time)) ])
    state_eul_vel_x = np.array([ state_eul[:][i][1][0]  for i in range(len(time)) ])
    state_eul_vel_y = np.array([ state_eul[:][i][1][1]  for i in range(len(time)) ])
    state_eul_acc_x = np.array([ state_eul[:][i][2][0]  for i in range(len(time)) ])
    state_eul_acc_y = np.array([ state_eul[:][i][2][1]  for i in range(len(time)) ])

    ctrl_x = np.array([ ctrl[i][0][0]  for i in range(len(time)) ]   )
    ctrl_y = np.array([ ctrl[i][0][1]  for i in range(len(time)) ]   )

    # Plot to confirm working
    fig, ax = plt.subplots(3,2,sharex=True)
    fig.suptitle("2D")
    ax[0,0].set_title("X")
    ax[0,0].plot( time, state_rk4_pos_x, label="rk4")
    ax[0,0].plot( time, state_eul_pos_x, '--', label="eul")
    ax[0,0].set_ylabel("Position")
    ax[1,0].plot( time, state_rk4_vel_x, label="rk4")
    ax[1,0].plot( time, state_eul_vel_x, '--', label="eul")
    ax[1,0].set_ylabel("Velocity")
    ax[2,0].plot( time, state_rk4_acc_x, label="rk4")
    ax[2,0].plot( time, state_eul_acc_x, '--', label="eul")
    ax[2,0].plot( time, ctrl_x, '--')
    ax[2,0].set_ylabel("Acceleration")
    ax[2,0].set_xlabel("Time")
    ax[2,0].legend()

    ax[0,1].set_title("Y")
    ax[0,1].plot( time, state_rk4_pos_y, label="rk4")
    ax[0,1].plot( time, state_eul_pos_y, '--', label="eul")
    #ax[0,1].set_ylabel("Position")
    ax[1,1].plot( time, state_rk4_vel_y, label="rk4")
    ax[1,1].plot( time, state_eul_vel_y, '--', label="eul")
    #ax[1,1].set_ylabel("Velocity")
    ax[2,1].plot( time, state_rk4_acc_y, label="rk4")
    ax[2,1].plot( time, state_eul_acc_y, '--', label="eul")
    ax[2,1].plot( time, ctrl_y, '--')
    #ax[2,1].set_ylabel("Acceleration")
    ax[2,1].set_xlabel("Time")
    ax[2,1].legend()
    #fig.savefig("test_integration_2d.png")

def test_integration_3d(tfinal=10, n_axis=3):
    np.random.seed(0)

    tstart   = 0
    dt       = 0.01
    time     = np.arange(tstart, tfinal, dt)

    # Random Uniform Initial State
    pos = np.random.uniform(-1, 1, n_axis)
    vel = np.zeros_like(pos)
    acc = np.zeros_like(pos)
    initial_state = np.array( [pos, vel, acc] )
    print(f"initial_state = {initial_state}")
    
    tmp = []
    for i in range(len(time)):
        tmp.append(np.empty( (3,n_axis) ) )

    state_rk4    = tmp
    state_rk4[0] = initial_state.copy()
    state_eul    = tmp
    state_eul[0] = initial_state.copy()

    # Apply control command to entire array
    #ctrl_cmd = 5.0
    ctrl_cmd = np.ones( (n_axis,) )* 5.0
    tmp = []
    for i in range(len(time)):
        tmp.append( np.ones( (1,n_axis) ) )
    ctrl  = [i * ctrl_cmd for i in tmp]

    # Perform Integration
    for i, t in enumerate(time):
        if i == len(time)-1:
            continue
        state_rk4[i+1] = rk4(   state_rk4[i], ctrl[i], dt )
        state_eul[i+1] = euler( state_eul[i], ctrl[i], dt )

    state_rk4_pos_x = np.array([ state_rk4[:][i][0][0]  for i in range(len(time)) ])
    state_rk4_pos_y = np.array([ state_rk4[:][i][0][1]  for i in range(len(time)) ])
    state_rk4_pos_z = np.array([ state_rk4[:][i][0][2]  for i in range(len(time)) ])
    state_rk4_vel_x = np.array([ state_rk4[:][i][1][0]  for i in range(len(time)) ])
    state_rk4_vel_y = np.array([ state_rk4[:][i][1][1]  for i in range(len(time)) ])
    state_rk4_vel_z = np.array([ state_rk4[:][i][1][2]  for i in range(len(time)) ])
    state_rk4_acc_x = np.array([ state_rk4[:][i][2][0]  for i in range(len(time)) ])
    state_rk4_acc_y = np.array([ state_rk4[:][i][2][1]  for i in range(len(time)) ])
    state_rk4_acc_z = np.array([ state_rk4[:][i][2][2]  for i in range(len(time)) ])

    state_eul_pos_x = np.array([ state_eul[:][i][0][0]  for i in range(len(time)) ])
    state_eul_pos_y = np.array([ state_eul[:][i][0][1]  for i in range(len(time)) ])
    state_eul_pos_z = np.array([ state_eul[:][i][0][2]  for i in range(len(time)) ])
    state_eul_vel_x = np.array([ state_eul[:][i][1][0]  for i in range(len(time)) ])
    state_eul_vel_y = np.array([ state_eul[:][i][1][1]  for i in range(len(time)) ])
    state_eul_vel_z = np.array([ state_eul[:][i][1][2]  for i in range(len(time)) ])
    state_eul_acc_x = np.array([ state_eul[:][i][2][0]  for i in range(len(time)) ])
    state_eul_acc_y = np.array([ state_eul[:][i][2][1]  for i in range(len(time)) ])
    state_eul_acc_z = np.array([ state_eul[:][i][2][2]  for i in range(len(time)) ])

    ctrl_x = np.array([ ctrl[i][0][0]  for i in range(len(time)) ]   )
    ctrl_y = np.array([ ctrl[i][0][1]  for i in range(len(time)) ]   )
    ctrl_z = np.array([ ctrl[i][0][2]  for i in range(len(time)) ]   )

    # Plot to confirm working
    fig, ax = plt.subplots(3,3,sharex=True)
    fig.suptitle("3D")
    ax[0,0].set_title("X")
    ax[0,0].plot( time, state_rk4_pos_x, label="rk4")
    ax[0,0].plot( time, state_eul_pos_x, '--', label="eul")
    ax[0,0].set_ylabel("Position")
    ax[1,0].plot( time, state_rk4_vel_x, label="rk4")
    ax[1,0].plot( time, state_eul_vel_x, '--', label="eul")
    ax[1,0].set_ylabel("Velocity")
    ax[2,0].plot( time, state_rk4_acc_x, label="rk4")
    ax[2,0].plot( time, state_eul_acc_x, '--', label="eul")
    ax[2,0].plot( time, ctrl_x, '--')
    ax[2,0].set_ylabel("Acceleration")
    ax[2,0].legend()

    ax[0,1].set_title("Y")
    ax[0,1].plot( time, state_rk4_pos_y, label="rk4")
    ax[0,1].plot( time, state_eul_pos_y, '--', label="eul")
    ax[1,1].plot( time, state_rk4_vel_y, label="rk4")
    ax[1,1].plot( time, state_eul_vel_y, '--', label="eul")
    ax[2,1].plot( time, state_rk4_acc_y, label="rk4")
    ax[2,1].plot( time, state_eul_acc_y, '--', label="eul")
    ax[2,1].plot( time, ctrl_y, '--')
    ax[2,1].set_xlabel("Time")
    ax[2,1].legend()

    ax[0,2].set_title("Z")
    ax[0,2].plot( time, state_rk4_pos_z, label="rk4")
    ax[0,2].plot( time, state_eul_pos_z, '--', label="eul")
    ax[1,2].plot( time, state_rk4_vel_z, label="rk4")
    ax[1,2].plot( time, state_eul_vel_z, '--', label="eul")
    ax[2,2].plot( time, state_rk4_acc_z, label="rk4")
    ax[2,2].plot( time, state_eul_acc_z, '--', label="eul")
    ax[2,2].plot( time, ctrl_z, '--')
    ax[2,2].legend()
    #fig.savefig("test_integration_3d.png")

