import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_log(filename):
    log = pickle.load(open(filename, 'rb'))

    df = pd.DataFrame({'iter': log['log']['iter'], 'error': log['log']['error']})
    sns.lineplot(x='iter', y='error', drawstyle='steps', data=df)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Objective value evolution')

    # point the best solution
    plt.scatter(x = log['opt']['iter'], y = log['opt']['error'], color='orange', label='Best solution')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    plot_log('log_error.pkl')
    

