"""
System Identification (SI)
https://arxiv.org/abs/1702.02453

Examples of two types:
1. Off-line SI: in sim2real_policies.sys_id.common.utils
2. On-line SI
"""

from sim2real_policies.sys_id.common.operations import *
from sim2real_policies.sys_id.common.utils import *
from sim2real_policies.utils.rl_utils import load, load_model
from sim2real_policies.utils.choose_env import choose_env


class OSI(object):
    """
    The class of online system identification
    Args:
        Projection (bool): whether exists a projection module for reducing the dimension of state
        CAT_INTERNAL (bool): whether concatenate the interal state to the external observation
        context_dim (int): the integral compressed dimension for the projcection module
    """
    def __init__(self, env_name='SawyerReach', length=3, context_dim=3, Projection=True, CAT_INTERNAL=False):
        self.cat_internal = CAT_INTERNAL
        env, environment_params, environment_wrappers, environment_wrapper_arguments = choose_env(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        print('Env name: ', env_name)
        print('Dimension of env state: ', state_dim)
        print('Dimension of env action: ', action_dim)
        self.params_dim = env.get_randomised_parameter_dimensions()
        print('Dimension of randomised parameters: ', self.params_dim)
        data_dim = length*(state_dim+action_dim)
        if CAT_INTERNAL:
            internal_state_dim = env.get_internal_state_dimension()
            print('Dimension of internal state: ', internal_state_dim)
            data_dim = length*(state_dim+action_dim+internal_state_dim)
        else:
            data_dim = length*(state_dim+action_dim)
        self.osi_model = OSINetork(input_dim = data_dim, output_dim = self.params_dim)
        self.env_name = env_name
        self.length = length  # trajectory length for prediction

        if Projection:
            self.proj_net = load_model(path = '../../../../data/pup_td3/model/pup_td3_projection', input_dim=self.params_dim, output_dim=context_dim)
            self.policy=load(path = '../../../../data/pup_td3/model/pup_td3', alg='TD3', state_dim = state_dim+context_dim, action_dim = action_dim)
            self.save_path = '../../../../../data/pup_td3/model/osi'
            
        else:
            self.proj_net = None
            self.policy=load(path = '../../../../data/up_td3/model/up_td3', alg='TD3', state_dim = state_dim+self.params_dim, action_dim = action_dim)
            self.save_path = '../../../../../data/up_td3/model/osi'

    def predict(self, traj):
        traj_input = stack_data(traj, self.length)
        print(traj_input)
        output = self.osi_model(traj_input).detach().numpy()
        print('out: ', output)
        return output

    def load_model(self):
        self.osi_model.load_state_dict(torch.load(self.save_path, map_location='cuda:0'))
        self.osi_model.eval()

    def osi_train(self, itr = 20):   
        # update with true dynamics parameters from simulator
        print('Started OSI training stage I.'+'\n'+'--------------------------------------------------')
        params, raw_history = self.online_history_collection(itr=10, PRED_PARAM=False, CAT_INTERNAL=self.cat_internal) 
        label, data = self.generate_data(params, raw_history)
        self.osi_update(data, label, epoch=5)
        print('Finished OSI training stage I.')
        print('Started OSI training stage II.'+'\n'+'--------------------------------------------------')
        # update with predicted dynamics parameters from simulator
        losses = []
        for _ in range(itr):  # while not converge
            params, raw_history = self.online_history_collection(PRED_PARAM=True, CAT_INTERNAL = self.cat_internal) 
            label, data = self.generate_data(params, raw_history)
            loss = self.osi_update(data, label, epoch=5)
            losses.append(loss)
            plot(losses, name='osi_train')
        print('Finished OSI training stage II.')



    def generate_data(self, params, raw_history):
        """ 
        generate training dataset with raw history trajectories;
        length is the number of (state, action, next_state) pairs, there are l state-action pairs in length l sequence
        """
        assert len(params) == len(raw_history)
        label=[]
        data=[]
        for param, epi in zip(params, raw_history):
            for i in range(0, len(epi)-self.length):
                data.append(epi[i:i+self.length].reshape(-1))   # [s,a,s,a] for length=2
                label.append(param)
        assert len(label)==len(data)

        return label, data


    def online_history_collection(self, itr=30, max_steps=30, PRED_PARAM=False, CAT_INTERNAL=False):
        """ collect random simulation parameters and trajetories with universal policy 
        https://arxiv.org/abs/1702.02453 (Preparing for the Unknown: Learning a Universal Policy with Online System Identification)
        """
        env, environment_params, environment_wrappers, environment_wrapper_arguments = choose_env(self.env_name)
        action_space = env.action_space
        ini_state_space = env.observation_space
        state_space = spaces.Box(-np.inf, np.inf, shape=(ini_state_space.shape[0]+self.params_dim, ))  # add the dynamics param dim

        # a random policy
        data_collection_policy=DPG_PolicyNetwork(state_space, action_space, hidden_dim=512).cuda()

        params_list=[]
        history=[]
        for eps in range(itr):  # K
            state = env.reset()
            params = query_params(env, randomised_only=True)
            epi_traj = []
            params_list.append(params)

            # N is 1 in this implementation, as each env.reset() will have different parameter set

            for step in range(max_steps):  # T
                if CAT_INTERNAL:
                    internal_state = env.get_internal_state()
                    full_state = np.concatenate([state, internal_state])
                else:
                    full_state = state
                if len(epi_traj)>=self.length and PRED_PARAM:
                    osi_input = stack_data(epi_traj, self.length)  # stack (s,a) to have same length as in the model input
                    pre_params = self.osi_model(osi_input).detach().numpy()
                else:
                    pre_params = params

                if self.proj_net is not None:  # projected to low dimensions
                    pre_params = self.proj_net.get_context(pre_params)
                else:
                    pass
                    # print('No projection network!')
                params_state = np.concatenate((pre_params, state))   # use predicted parameters instead of true values for training, according to the paper
                action = data_collection_policy.get_action(params_state)
                epi_traj.append(np.concatenate((full_state, action)))

                next_state, _, _, _ = env.step(action)
                state = next_state
            history.append(np.array(epi_traj))
        print("Finished collecting data of {} trajectories.".format(itr))
        return params_list, history


    def osi_update(self, input, label, epoch=1, lr=1e-1):
        """ Update the system identification (SI) with online data collection """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.osi_model.parameters(), lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)  # gamma: decay for each step
        input = torch.Tensor(input)
        label = torch.Tensor(label)

        for i in range(epoch):
            predict = self.osi_model(input)
            loss = criterion(predict, label)
            optimizer.zero_grad()
            loss.backward()
            print('Train the SI model, Epoch: {} | Loss: {}'.format(i, loss))
            optimizer.step()
            scheduler.step()

        torch.save(self.osi_model.state_dict(), self.save_path)

        return loss.detach().cpu().numpy()

class OSINetork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, dropout=0.1):
        """ same OSI network structure as: https://arxiv.org/abs/1702.02453 """
        super(OSINetork, self).__init__()
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
        x = self.linear4(x)
        return x.squeeze(0)

def stack_data(traj, length):
    traj = np.array(traj)
    return traj[-length:, :].reshape(-1)

if __name__ == '__main__':
    ENV_NAME =['SawyerReach', 'SawyerPush', 'SawyerSlide'][0]

    osi = OSI(env_name = ENV_NAME, length=3, context_dim=3, Projection=False, CAT_INTERNAL=True)
    osi.osi_train()