import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)


def plot_number_of_sessions():
    figure = plt.figure()
    figure.set_size_inches(12, 8)

    ax = figure.add_subplot(1, 1, 1)
    ax.set_ylabel('# Sessions')
    ax.set_xlabel('Time Delta (minutes)')
    number_of_sessions = [35809, 27682, 24861, 23469, 22575, 20627, 19936, 19474, 19222, 19036, 18996, 18959, 18944, 18936, 18927, 18921]
    time_deltas = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    plt.plot(time_deltas, number_of_sessions)
    plt.savefig('data/plots/sessionlength_vs_timedelta.png', dpi=400)


def plot_average_session_length():
    figure = plt.figure()
    figure.set_size_inches(12, 8)

    ax = figure.add_subplot(1, 1, 1)
    ax.set_ylabel('Session Length (page-views)')
    ax.set_xlabel('Time Delta (minutes)')
    average_length = [6.0922673071, 7.88086120945, 8.77510960943, 9.29558140526, 9.66369878184, 10.576331992, 10.9429173355, 11.2025264455, 11.3493913224, 11.4602857743,
                      11.4844177722, 11.506830529, 11.515941723, 11.5208069286, 11.526285201, 11.529940278]
    time_deltas = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    plt.plot(time_deltas, average_length)
    plt.savefig('data/plots/average_session_length.png', dpi=400)


def plot_session_distribution():
    session_df = pd.read_csv('logs/big_session_df_new.csv', sep='|', header=0)
    session_distr = session_df['Session-Nr'].value_counts()

    figure = plt.figure()
    figure.set_size_inches(16, 12)

    ax = figure.add_subplot(1, 1, 1)
    ax.set_ylabel('Session Length')
    ax.set_xlabel('Sessions (sorted by Length)')

    median = session_distr.values[len(session_distr.values)/2]
    plt.title('Median: %d' % median)
    plt.tight_layout()

    left = [x for x in xrange(0, len(session_distr.values), 1)]
    ax.bar(left, session_distr.values, width=1, color='b', linewidth=0)

    plt.savefig('data/plots/session_distribution.png', dpi=400)


def plot_session_histogram():
    session_df = pd.read_csv('logs/big_session_df_new.csv', sep='|', header=0)
    session_distr = session_df['Session-Nr'].value_counts()
    session_occurances = session_distr.value_counts().sort_index()

    print session_occurances

    figure = plt.figure()
    figure.set_size_inches(16, 12)

    ax = figure.add_subplot(1, 1, 1)
    ax.set_ylabel('# Sessions')
    ax.set_xlabel('Session Length')

    #plt.tight_layout()

    left = [x for x in xrange(1, 31, 1)]
    ax.bar(left, session_occurances.values[:30], width=0.5, color='b', linewidth=0)

    plt.savefig('data/plots/session_histogram.png', dpi=400)


def main():
    #plot_number_of_sessions()
    #plot_average_session_length()
    #plot_session_distribution()
    plot_session_histogram()

if __name__ == '__main__':
    main()
