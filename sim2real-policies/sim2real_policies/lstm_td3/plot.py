# high-dimensional reacher
import numpy as np
import matplotlib.pyplot as plt

def smooth(y, radius=2, mode='two_sided'):
    if len(y) < 2*radius+1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius+1)
        return np.convolve(y, convkernel, mode='same') / \
               np.convolve(np.ones_like(y), convkernel, mode='same')
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel,mode='full') / \
              np.convolve(np.ones_like(y), convkernel, mode='full')
        return out[:-radius+1]
    
def moving_sum(y, window=2):
    c = y.cumsum()
    c[window:] = c[window:] - c[:-window]
    return c/float(window)
    


def plot(x, data, color, label):
    y_m=np.mean(data, axis=0)
    y_std=np.std(data, axis=0)
    y_upper=y_m+y_std
    y_lower=y_m-y_std
    plt.fill_between(
    x, list(y_lower), list(y_upper), interpolate=True, facecolor=color, linewidth=0.0, alpha=0.3
)
    plt.plot(x, list(y_m), color=color, label=label)
    

file_pre = './'
y=np.load(file_pre+'eval_rewards.npy')
s=np.load(file_pre+'eval_success.npy')

plt.figure(figsize=(8,6))
fig, axs = plt.subplots(2)
x=np.arange(len(y))
axs[0].plot(x, smooth(y), label = 'SawyerPush', color='b')

axs[0].set_ylabel('Reward')
# plt.ylim(0)
axs[0].legend( loc=2)
axs[0].grid()
axs[1].set_xlabel('Episodes')
axs[1].set_ylabel('Average Success Rate')
axs[1].plot(x, moving_sum(s), label = 'SawyerPush', color='b')
axs[1].grid()


plt.savefig('reward.pdf')
plt.show()
