"""
System Identification (SI)
https://arxiv.org/abs/1702.02453

Examples of two types:
1. Off-line SI
2. On-line SI
"""
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)  # add parent path
from gym import spaces
from sim2real_policies.utils.rl_utils import load, load_model
from sim2real_policies.utils.choose_env import choose_env
from sim2real_policies.utils.policy_networks import DPG_PolicyNetwork, RandomPolicy
from sim2real_policies.sys_id.common.nets import *
from sim2real_policies.sys_id.common.operations import plot, plot_train_test
from sim2real_policies.sys_id.common.utils import query_key_params, query_params
from torch.utils.tensorboard import SummaryWriter


############### Offline SI joint by joint ################

class OSINetwork_single(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, dropout=0.0):
        """ OSI network for single joint """
        super(OSINetwork_single, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, int(hidden_dim/4))
        self.dropout2 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(int(hidden_dim/4), output_dim)

    def forward(self, input):
        if len(input.shape) < 2:
            input = torch.FloatTensor(np.expand_dims(input, 0))
        x = F.tanh(self.linear1(input))
        x = self.dropout1(x)
        x = F.tanh(self.linear2(x))
        x = self.dropout2(x)
        x = F.tanh(self.linear3(x))
        return x.squeeze(0)

def separate_data(x, y, dim):
    # data processing to separate it for each joint
    input_list = [x[:, i::dim] for i in range(dim)]  # list of input for each joint
    label_list = [y[:, i] for i in range(dim)] # list of label for each joint
    return input_list, label_list

def OfflineSIupdate_separate(input, label, test_input=None, test_label=None, \
    epoch=500, lr=1e-3, save_path='./model/osi', dim_joint=7):
    """ Update the system identification (SI) with offline dataset for each joint separately """
    writer = SummaryWriter()

    criterion = nn.MSELoss()
    data_dim = len(input[0])
    label_dim = len(label[0])
    osi_model_list=[]
    params_list = []
    for i in range(dim_joint):
        osi_model = OSINetwork_single(input_dim = int(data_dim/dim_joint), output_dim = int(label_dim/dim_joint))
        osi_model_list.append(osi_model)
        params_list += osi_model.parameters()

    optimizer = optim.Adam(params_list, lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)  # gamma: decay for each step
    input_list, label_list = separate_data(input, label, dim_joint)

    for i in range(epoch):
        joints_loss = []
        for j in range(dim_joint):
            input = input_list[j]
            label = label_list[j]
            osi_model = osi_model_list[j]
            input = torch.Tensor(input)
            label = torch.Tensor(label).view(-1, 1)
            predict = osi_model(input)
            single_loss = criterion(predict, label)
            joints_loss.append(single_loss)
        # epoch_loss += np.sum(joints_loss)
        optimizer.zero_grad()
        epoch_loss = torch.mean(torch.stack(joints_loss))
        epoch_loss.backward()
        optimizer.step()
        print('Epoch: {} | Loss: {}'.format(i, epoch_loss.detach()))
        scheduler.step()

        if i%5==0:
            writer.add_scalars('Train Loss/Separate', {'joint_0':joints_loss[0],
                'joint_1':joints_loss[1],
                'joint_2':joints_loss[2], 
                'joint_3':joints_loss[3],
                'joint_4':joints_loss[4],
                'joint_5':joints_loss[5],
                'joint_6':joints_loss[6]},
                i)
            for j in range(dim_joint):
                torch.save(osi_model_list[j].state_dict(), save_path+'{}'.format(j))
            if test_input is not None:
                test_loss = OfflineSIeval_separate(test_input, test_label)
                writer.add_scalars('Loss/All', {'train_loss: ': epoch_loss.detach(), 'test_loss: ': test_loss }, i)
    # save model
    for j in range(dim_joint):
        torch.save(osi_model_list[j].state_dict(), save_path+'{}'.format(j))

def OfflineSIeval_separate(input, label, save_path='./model/osi', dim_joint=7):
    """ Evaluate the trained system identification (SI) model with test dataset for each joint separately """
    criterion = nn.MSELoss()
    data_dim = len(input[0])
    label_dim = len(label[0])
    osi_model_list=[]
    for i in range(dim_joint):
        osi_model = OSINetwork_single(input_dim = int(data_dim/dim_joint), output_dim = int(label_dim/dim_joint))
        osi_model.load_state_dict(torch.load(save_path+'{}'.format(i)))
        osi_model_list.append(osi_model)
    input_list, label_list = separate_data(input, label, dim_joint)
    loss = []
    for j in range(dim_joint):
        input = input_list[j]
        label = label_list[j]
        osi_model = osi_model_list[j]
        input = torch.Tensor(input)
        label = torch.Tensor(label).view(-1, 1)
        predict = osi_model(input)
        loss.append(criterion(predict, label).detach().cpu().numpy())
    return np.mean(loss)


############### Offline SI all together ################

class OSINetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, dropout=0.1):
        """ same OSI network structure as: https://arxiv.org/abs/1702.02453 """
        super(OSINetwork, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.dropout2 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(int(hidden_dim/2), int(hidden_dim/4))
        self.dropout3 = nn.Dropout(dropout)
        self.linear4 = nn.Linear(int(hidden_dim/4), output_dim)

    def forward(self, input):
        if len(input.shape) < 2:
            input = torch.FloatTensor(np.expand_dims(input, 0))
        x = F.tanh(self.linear1(input))
        x = self.dropout1(x)
        x = F.tanh(self.linear2(x))
        x = self.dropout2(x)
        x = F.tanh(self.linear3(x))
        x = self.dropout3(x)
        x = F.tanh(self.linear4(x))  # if x is normalized
        return x.squeeze(0)

def OfflineSIupdate(input, label, test_input=None, test_label=None, epoch=500, lr=1e-3, save_path='./model/osi'):
    """ Update the system identification (SI) with offline dataset """
    writer = SummaryWriter()

    criterion = nn.MSELoss(reduce=False)  # not reduce for plotting in scalar separately
    data_dim = len(input[0])
    label_dim = len(label[0])
    osi_model = OSINetwork(input_dim = data_dim, output_dim = label_dim)
    optimizer = optim.Adam(osi_model.parameters(), lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)  # gamma: decay for each step
    input = torch.Tensor(input)
    label = torch.Tensor(label)
    loss_list=[]
    test_loss_list=[]

    for i in range(epoch):
        predict = osi_model(input)
        loss = criterion(predict, label)
        loss = loss.mean(dim=0)
        for j in range(loss.shape[0]):
            writer.add_scalar('Loss/Train index{}'.format(j), loss[j].detach().cpu().numpy(), i)
        loss = torch.mean(loss)
        optimizer.zero_grad()
        loss.backward()
        print('Epoch: {} | Loss: {}'.format(i, loss))
        optimizer.step()
        scheduler.step()

        if i%10==0:
            torch.save(osi_model.state_dict(), save_path)
            loss_list.append(loss.detach().cpu().numpy())  # if not detach to cpu, cause memory leakage!
            if test_input is not None:
                test_loss = OfflineSIeval(osi_model, test_input, test_label)
                test_loss_list.append(test_loss)
                writer.add_scalars('Loss/Total', {'train_loss: ': loss.detach(), 'test_loss: ': test_loss }, i)
            #     plot_train_test(loss_list, test_loss_list)
            # else:
            #     plot(loss_list)
    
    torch.save(osi_model.state_dict(), save_path)

def OfflineSIeval(model, input, label, save_path='./osi'):
    """ Evaluate the trained system identification (SI) model with test dataset """
    criterion = nn.MSELoss()
    data_dim = len(input[0])
    label_dim = len(label[0])
    input = torch.Tensor(input)
    label = torch.Tensor(label)
    if model is None:
        model = OSINetwork(input_dim = data_dim, output_dim = label_dim)
        model.load_state_dict(torch.load(save_path))
    predict = model(input)
    print('label: ',label)
    print('predict: ', predict)
    loss = criterion(predict, label)
    return loss.detach().cpu().numpy()  # if not detach to cpu, cause memory leakage!



def OfflineSI(env_name='SawyerReach', params_dim=37, length=3):
    """ 
    offline system identification: collect trajectories and train the SI model
    """    
    params, raw_history = offline_history_collection(env_name=env_name, params_dim = params_dim) 
    label, data = generate_data(params, raw_history, length=length)
    OfflineSIupdate(data, label)
    

############### Online SI (OSI) ################

def OnlineSIupdate(OSImodel, input, label, epoch=1, lr=1e-3, save_path='./osi'):
    """ Update the system identification (SI) with online data collection """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(OSImodel.parameters(), lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)  # gamma: decay for each step
    input = torch.Tensor(input)
    label = torch.Tensor(label)
    total_loss = 0

    for i in range(epoch):
        predict = OSImodel(input)
        # print(predict.shape, label.shape)
        loss = criterion(predict, label)
        optimizer.zero_grad()
        loss.backward()
        # print('Train the SI model, Epoch: {} | Loss: {}'.format(i, loss))
        optimizer.step()
        scheduler.step()
        total_loss+=loss.detach().cpu().numpy()

    torch.save(OSImodel.state_dict(), save_path)
    return OSImodel, total_loss

def offline_history_collection(env_name, itr=30, max_steps=30, policy=None, params_dim=37, SIspace='end', selected_joints=[0]):
    """ collect random simulation parameters and trajetories with given policy """
    env, environment_params, environment_wrappers, environment_wrapper_arguments = choose_env(env_name)
    action_space = env.action_space
    state_space = env.observation_space
    if policy is None:  # load off-line policy is no policy
        policy=DPG_PolicyNetwork(state_space, action_space, 512).cuda()
    
    history=[]
    params_list=[]
    history=[]
    for epi in range(itr):
        state = env.reset()
        # params = query_params(env)
        params = query_key_params(env)
        epi_traj = []
        params_list.append(params)
        for step in range(max_steps):
            action = policy.get_action(state)
            next_state, _, _, info = env.step(action)
            if SIspace == 'end':
                epi_traj.append(np.concatenate((state, action)))
            elif SIspace == 'joint':
                epi_traj.append(np.concatenate((env._joint_positions[selected_joints], info['joint_velocities'][selected_joints])))

            state = next_state
        history.append(np.array(epi_traj))
    print("Finished collecting data.")
    return params_list, history


def online_history_collection(OSImodel, env_name='SawyerReach', proj_net=None, policy=None, length=3, \
    itr=30, max_steps=30, params_dim = 37, hidden_dim=512, PRED_PARAM=False, SIspace='end', selected_joints=[0]):
    """ collect random simulation parameters and trajetories with universal policy 
    https://arxiv.org/abs/1702.02453 (Preparing for the Unknown: Learning a Universal Policy with Online System Identification)
    """
    env, environment_params, environment_wrappers, environment_wrapper_arguments = choose_env(env_name)
    action_space = env.action_space
    ini_state_space = env.observation_space
    state_space = spaces.Box(-np.inf, np.inf, shape=(ini_state_space.shape[0]+params_dim, ))  # add the dynamics param dim

    if policy is None:  # load off-line policy if no policy
        policy=TD3_PolicyNetwork(state_space, action_space, hidden_dim).cuda()

    params_list=[]
    history=[]
    for eps in range(itr):  # K
        state = env.reset()
        # params = query_params(env)
        params = query_key_params(env)
        epi_traj = []
        params_list.append(params)

        # N is 1 in this implementation, as each env.reset() will have different parameter set

        for step in range(max_steps):  # T
            if len(epi_traj)>=length and PRED_PARAM:
                osi_input = stack_data(epi_traj, length)  # stack (s,a) to have same length as in the model input
                pre_params = OSImodel(osi_input).detach().numpy()
            else:
                pre_params = params

            if proj_net is not None:  # projected to low dimensions
                pre_params = proj_net.get_context(pre_params)
            params_state = np.concatenate((pre_params, state))   # use predicted parameters instead of true values for training, according to the paper
            action = policy.get_action(params_state)
            next_state, _, _, info = env.step(action)
            if SIspace == 'end':
                epi_traj.append(np.concatenate((state, action)))
            elif SIspace == 'joint':
                epi_traj.append(np.concatenate((env._joint_positions[selected_joints], info['joint_velocities'][selected_joints])))

            state = next_state
        history.append(np.array(epi_traj))
    print("Finished collecting data of {} trajectories.".format(itr))
    return params_list, history

def stack_data(traj, length):
    traj = np.array(traj)
    return traj[-length:, :].reshape(-1)

def generate_data(params, raw_history, length=3):
    """ 
    generate training dataset with raw history trajectories;
    length is the number of (state, action, next_state) pairs, there are l state-action pairs in length l sequence
    """
    assert len(params) == len(raw_history)
    label=[]
    data=[]
    for param, epi in zip(params, raw_history):
        for i in range(0, len(epi)-length):
            data.append(epi[i:i+length].reshape(-1))   # [s,a,s,a] for length=2
            label.append(param)
    assert len(label)==len(data)

    return label, data

def evaluate_model(model, label, data):
    """
    evaluate the trained model with evaluation dataset
    """
    data = torch.Tensor(data)
    label = torch.Tensor(label)
    criterion = nn.MSELoss()
    predict = model(data)
    loss = criterion(predict, label)
    return loss


def OnlineSI(env_name='SawyerReach', state_dim=6, action_dim=3, length=3, params_dim=4, context_dim=3,\
     itr = 100, Projection=True, SIspace='joint', selected_joints=[0,1,3,5]): # SIspace: 'joint' or 'end'
    """
    online system identification: train the SI model with collected trajectories, 
    then train SI with the exploration universal policy conditioned on SI predicted params
    """
    if SIspace == 'end':
        data_dim = length*(state_dim+action_dim)
        save_path='osi_end'
    elif SIspace == 'joint':
        joint_dim = len(selected_joints)
        data_dim = length*(joint_dim+joint_dim)  # for joint space, 'state' is joint position, 'action' is joint velocity
        save_path='osi_joint'
    osi_model = OSINetwork(input_dim = data_dim, output_dim = params_dim)

    # evaluation dataset, collected offline
    params, raw_history = offline_history_collection(env_name, itr=50, max_steps=30, \
        policy=RandomPolicy(action_dim = action_dim), params_dim=params_dim, SIspace=SIspace, selected_joints=selected_joints)
    eval_label, eval_data = generate_data(params, raw_history, length=length)

    if Projection:
        proj_net = load_model(path = '../../../../data/pup_td3/model/pup_td3_projection', input_dim=params_dim, output_dim=context_dim)
        policy=load(path = '../../../../data/pup_td3/model/pup_td3', alg='TD3', state_dim = state_dim+context_dim, action_dim = action_dim)
        
    else:
        proj_net = None
        try:
            policy=load(path = '../../../../data/up_td3/model/up_td3', alg='TD3', state_dim = state_dim+params_dim, action_dim = action_dim)
            print('Load previously saved policy.')
        except:
            policy=RandomPolicy(action_dim = action_dim)
            print('Load random policy.')

    # update with true dynamics parameters from simulator
    print('Started OSI training stage I.'+'\n'+'--------------------------------------------------')
    params, raw_history = online_history_collection(osi_model, env_name, length=length, \
        proj_net = proj_net, policy = policy, params_dim = params_dim, PRED_PARAM=False, SIspace=SIspace, selected_joints=selected_joints) 
    label, data = generate_data(params, raw_history, length)
    osi_model = OnlineSIupdate(osi_model, data, label, epoch=50, save_path=save_path)
    print('Finished OSI training stage I.')
    print('Started OSI training stage II.'+'\n'+'--------------------------------------------------')
    # update with predicted dynamics parameters from simulator
    loss_list=[]
    lr = 1e-3
    decay=0.8
    for _ in range(itr):  # while not converge
        params, raw_history = online_history_collection(osi_model, env_name, length=length, \
            proj_net = proj_net, policy = policy, params_dim = params_dim, PRED_PARAM=True, SIspace=SIspace, selected_joints=selected_joints) 
        label, data = generate_data(params, raw_history, length)
        osi_model = OnlineSIupdate(osi_model, data, label, epoch=50, save_path=save_path, lr=lr)
        lr*=decay
        loss = evaluate_model(osi_model, eval_label, eval_data)
        loss_list.append(loss)
        if SIspace == 'end':
            plot(loss_list, name='osi_end')
        elif SIspace == 'joint':
            plot(loss_list, name='osi_joint')
    print('Finished OSI training stage II.')

def test(env_name='SawyerReach', state_dim=6, action_dim=3, length=3, params_dim=4, path='./osi', SIspace='end', selected_joints=[0]):
    env, environment_params, environment_wrappers, environment_wrapper_arguments = choose_env(env_name)
    if SIspace == 'end':
        data_dim = length*(state_dim+action_dim)
        save_path='osi_end'
    elif SIspace == 'joint':
        joint_dim = len(selected_joints)
        data_dim = length*(joint_dim+joint_dim)  # for joint space, 'state' is joint position, 'action' is joint velocity
        save_path='osi_joint'    
    osi_model = OSINetwork(input_dim = data_dim, output_dim = params_dim)

    osi_model.load_state_dict(torch.load(save_path))
    osi_model.eval()
    policy = RandomPolicy(action_dim = action_dim)

    for eps in range(10): 
        state = env.reset()
        # params = query_params(env)
        params = query_key_params(env)
        epi_traj = []
        print('true params: ', params)

        for step in range(30): 
            if len(epi_traj)>=length:
                osi_input = stack_data(epi_traj, length)  # stack (s,a) to have same length as in the model input
                pre_params = osi_model(osi_input).detach().numpy()
                print('predicted params: ', pre_params)

            action = policy.get_action(state)
            next_state, _, _, info = env.step(action)
            if SIspace == 'end':
                epi_traj.append(np.concatenate((state, action)))
            elif SIspace == 'joint':
                epi_traj.append(np.concatenate((env._joint_positions[selected_joints], info['joint_velocities'][selected_joints])))

            state = next_state


if __name__ == '__main__':
    state_dim=6
    action_dim=3
    length=8
    params_dim=1
    SIspace='joint'  # if 'joint', the simplest cases to predict the joint coefficients with joint positions and velocities
    selected_joints=[0]   # [0,1,3,5], predict selected joints only
    
    # OfflineSI()

    OnlineSI(state_dim=state_dim, action_dim=action_dim, length=length, params_dim=params_dim, Projection=False, SIspace=SIspace, selected_joints=selected_joints)

    test(state_dim=state_dim, action_dim=action_dim, length=length, params_dim=params_dim, SIspace=SIspace, selected_joints=selected_joints)