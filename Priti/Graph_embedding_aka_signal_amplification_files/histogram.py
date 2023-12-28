import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def save_and_show_plot(figure, filename):
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_events(events, counts, label):
    x = np.arange(len(events))
    bar_width = 0.7

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot bars
    bars = ax.bar(x, counts, width=bar_width, alpha=0.7, color='blue', edgecolor='black', linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(events, rotation=45, ha='right')
    ax.set_xlabel('Event')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Bar Chart of {label} Events')

    # Add events to the legend
    #ax.legend(bars, events, title='Events')

    save_and_show_plot(plt, f'{label.lower()}_events.pdf')

def plot_events_with_higher_variance(total_eventcount1, total_eventcount2, label1, label2):
    events_common = set(total_eventcount1.keys()).intersection(total_eventcount2.keys())

    # Calculate the absolute difference in counts for each event
    variance1 = {event: total_eventcount1[event] - total_eventcount2.get(event, 0) for event in events_common}
    variance2 = {event: total_eventcount2[event] - total_eventcount1.get(event, 0) for event in events_common}

    # Find events with higher variance in benign compared to malware
    high_variance_benign = {event: count1 for event, count1 in variance1.items() if count1 > variance2.get(event, 0)}

    # Find events with higher variance in malware compared to benign
    high_variance_malware = {event: count2 for event, count2 in variance2.items() if count2 > variance1.get(event, 0)}

    # Plot events with higher variance in benign
    if high_variance_benign:
        events_benign, counts_benign = zip(*high_variance_benign.items())
        plot_events(events_benign, counts_benign, f'{label1}_High_Variance_train_case2_Compared_to_{label2}')

    # Plot events with higher variance in malware
    if high_variance_malware:
        events_malware, counts_malware = zip(*high_variance_malware.items())
        plot_events(events_malware, counts_malware, f'{label2}_High_Variance_train_case2_Compared_to_{label1}')


def malware_benign_event_total_count_plot(malware_event_count_dict,benign_event_count_dict):
    events = list(malware_event_count_dict.keys())
    malware_counts = list(malware_event_count_dict.values())
    benign_counts = list(benign_event_count_dict.values())

    # Bar width
    bar_width = 0.5

    # Set up figure and axes
    fig, ax = plt.subplots()

    # Bar positions
    event_positions = np.arange(len(events))

    # Create bars
    malware_bars = ax.barh(event_positions - bar_width / 2, malware_counts, bar_width, label='Malware')
    benign_bars = ax.barh(event_positions + bar_width / 2, benign_counts, bar_width, label='Benign')

    # Set labels and title
    ax.set_ylabel('Events')
    ax.set_xlabel('Count')
    ax.set_title('Event Counts for Malware and Benign Samples')
    ax.set_yticks(event_positions)
    ax.set_yticklabels(events, fontsize=4) 
    #ax.set_yticklabels(events)

    for malware_bar, benign_bar in zip(malware_bars, benign_bars):
        ax.text(malware_bar.get_width(), malware_bar.get_y() + malware_bar.get_height() / 2,
                f'{malware_bar.get_width():,.0f}', ha='left', va='center', fontsize=2, color='black')

        ax.text(benign_bar.get_width(), benign_bar.get_y() + benign_bar.get_height() / 2,
            f'{benign_bar.get_width():,.0f}', ha='left', va='center', fontsize=2, color='black')

    # Adjust layout to prevent clipping of tick-labels
    plt.tight_layout()

    ax.legend()

    # Save plot to PDF
    pdf_output_file_path = "/home/pwakodi1/tabby/Graph_embedding_aka_signal_amplification_files/Analysis_results/Case1_Train_malware_benign_event_counts_plot.pdf"
    plt.savefig(pdf_output_file_path, bbox_inches='tight')  # Use bbox_inches='tight' to prevent label cropping

    # Show plot
    plt.show()

def plot_4events(Mtrain1, Btrain1, Mtest1,Btest1):
    group_size = 6
    keys = list(Mtrain1.keys())
    num_groups = len(keys) // group_size

    # Bar width
    bar_width = 0.2

    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size

        current_keys = keys[start_idx:end_idx]
        values1 = [Mtrain1[key] for key in current_keys]
        values2 = [Btrain1[key] for key in current_keys]
        values3 = [Mtest1[key] for key in current_keys]
        values4 = [Btest1[key] for key in current_keys]

        # Set up figure and axes
        fig, ax = plt.subplots()

        # Bar positions
        positions = np.arange(len(current_keys))

        # Create bars
        bars1 = ax.bar(positions - 1.5 * bar_width, values1, bar_width, label='Malwaretrain2')
        bars2 = ax.bar(positions - 0.5 * bar_width, values2, bar_width, label='Benigntrain2')
        bars3 = ax.bar(positions + 0.5 * bar_width, values3, bar_width, label='Malwaretest2')
        bars4 = ax.bar(positions + 1.5 * bar_width, values4, bar_width, label='Benigntest2')

        # Set labels and title
        ax.set_xlabel('Events')
        ax.set_ylabel('Count')
        ax.set_title(f'Comparison of Event Counts (Group {i+1})')
        ax.set_xticks(np.arange(len(current_keys)))
        ax.set_xticklabels(current_keys, fontsize=5)

        # Annotate bar values on top of each bar
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f'{bar.get_height():,.0f}',
                        ha='center',
                        va='bottom',
                        fontsize=5,
                        color='black')

        ax.legend()

        # Save plot to PDF
        pdf_output_file_path = f"/home/pwakodi1/tabby/Graph_embedding_aka_signal_amplification_files/Analysis_results/case2_event_counts_comparison_plot_group_{i+1}.pdf"
        plt.savefig(pdf_output_file_path, bbox_inches='tight')

        # Close the current figure
        plt.close()

        print(f"Plot saved to {pdf_output_file_path}")

total_eventcount_benign_test_case1 = {
    "0.0": 2222,
    "Cleanup": 73,
    "Close": 73,
    "Create": 73,
    "CreateNewFile": 69,
    "DeletePath": 70,
    "DirEnum": 72,
    "FSCTL": 70,
    "NameCreate": 73,
    "QueryInformation": 73,
    "QuerySecurity": 73,
    "Read": 73,
    "Write": 69,
    "SetDelete": 70,
    "CpuBasePriorityChange": 73,
    "ImageLoad": 73,
    "ImageUnload": 73,
    "ProcessStart/Start": 73,
    "ThreadStart/Start": 73,
    "ThreadStop/Stop": 73,
    "EventID(1)": 73,
    "EventID(2)": 73,
    "EventID(4)": 73,
    "EventID(7)": 73,
    "EventID(8)": 73,
    "EventID(9)": 73,
    "EventID(11)": 73,
    "EventID(13)": 73,
    "EventID(14)": 73,
    "KERNEL_NETWORK_TASK_TCPIP/Datasent.": 66,
    "KERNEL_NETWORK_TASK_TCPIP/Datareceived.": 51,
    "File": 73,
    "Reg": 73,
    "Net": 69,
    "Proc": 73,
    "Thread": 73,
    "KERNEL_NETWORK_TASK_TCPIP/Dataretransmitted.": 18,
    "EventID(5)": 11,
    "Flush": 11,
    "SetInformation": 12,
    "CpuPriorityChange": 13,
    "ThreadWorkOnBehalfUpdate": 6,
    "Unseen": 11,
    "ProcessStop/Stop": 7,
    "KERNEL_NETWORK_TASK_TCPIP/Connectionattempted.": 2,
    "KERNEL_NETWORK_TASK_TCPIP/Disconnectissued.": 1
}
total_eventcount_malware_test_case1 = {
    "0.0": 1945,
    "Cleanup": 62,
    "Close": 62,
    "Create": 62,
    "DirEnum": 62,
    "NameCreate": 61,
    "QueryInformation": 62,
    "QuerySecurity": 62,
    "Read": 62,
    "CpuBasePriorityChange": 62,
    "ImageLoad": 62,
    "ImageUnload": 62,
    "ProcessStart/Start": 62,
    "ThreadStart/Start": 62,
    "ThreadStop/Stop": 62,
    "KERNEL_NETWORK_TASK_TCPIP/Datasent.": 58,
    "File": 62,
    "Net": 60,
    "Proc": 62,
    "Thread": 62,
    "CreateNewFile": 60,
    "DeletePath": 60,
    "FSCTL": 60,
    "Write": 60,
    "SetDelete": 60,
    "EventID(1)": 61,
    "EventID(2)": 61,
    "EventID(4)": 61,
    "EventID(7)": 61,
    "EventID(8)": 61,
    "EventID(9)": 61,
    "EventID(11)": 61,
    "EventID(13)": 61,
    "EventID(14)": 61,
    "KERNEL_NETWORK_TASK_TCPIP/Datareceived.": 52,
    "Reg": 61,
    "EventID(15)": 1,
    "KERNEL_NETWORK_TASK_TCPIP/Dataretransmitted.": 8,
    "ProcessStop/Stop": 1,
    "KERNEL_NETWORK_TASK_TCPIP/Disconnectissued.": 3,
    "KERNEL_NETWORK_TASK_UDPIP/DatareceivedoverUDPprotocol.": 1
}

total_eventcount_benign_test_case2 = {
    "0.0": 5140,
    "Cleanup": 167,
    "Close": 167,
    "Create": 167,
    "CreateNewFile": 157,
    "DeletePath": 159,
    "DirEnum": 166,
    "FSCTL": 158,
    "NameCreate": 165,
    "QueryInformation": 167,
    "QuerySecurity": 167,
    "Read": 167,
    "Write": 157,
    "SetDelete": 159,
    "CpuBasePriorityChange": 167,
    "ImageLoad": 167,
    "ImageUnload": 167,
    "ProcessStart/Start": 167,
    "ThreadStart/Start": 167,
    "ThreadStop/Stop": 167,
    "EventID(1)": 167,
    "EventID(2)": 167,
    "EventID(4)": 167,
    "EventID(7)": 167,
    "EventID(8)": 167,
    "EventID(9)": 167,
    "EventID(11)": 167,
    "EventID(13)": 167,
    "EventID(14)": 167,
    "KERNEL_NETWORK_TASK_TCPIP/Datasent.": 146,
    "KERNEL_NETWORK_TASK_TCPIP/Datareceived.": 114,
    "File": 167,
    "Reg": 167,
    "Net": 152,
    "Proc": 167,
    "Thread": 167,
    "KERNEL_NETWORK_TASK_TCPIP/Dataretransmitted.": 30,
    "EventID(5)": 31,
    "Flush": 15,
    "SetInformation": 16,
    "CpuPriorityChange": 19,
    "ThreadWorkOnBehalfUpdate": 8,
    "Unseen": 15,
    "ProcessStop/Stop": 34,
    "KERNEL_NETWORK_TASK_TCPIP/Connectionattempted.": 2,
    "KERNEL_NETWORK_TASK_TCPIP/Disconnectissued.": 2,
    "KERNEL_NETWORK_TASK_UDPIP/DatareceivedoverUDPprotocol.": 1,
    "KERNEL_NETWORK_TASK_UDPIP/DatasentoverUDPprotocol.": 1
}
total_eventcount_malware_test_case2 = {
    "0.0": 4942,
    "Cleanup": 158,
    "Close": 158,
    "Create": 158,
    "DirEnum": 158,
    "NameCreate": 157,
    "QueryInformation": 158,
    "QuerySecurity": 158,
    "Read": 158,
    "CpuBasePriorityChange": 158,
    "ImageLoad": 158,
    "ImageUnload": 158,
    "ProcessStart/Start": 158,
    "ThreadStart/Start": 158,
    "ThreadStop/Stop": 158,
    "KERNEL_NETWORK_TASK_TCPIP/Datasent.": 143,
    "File": 158,
    "Net": 149,
    "Proc": 158,
    "Thread": 158,
    "CreateNewFile": 154,
    "DeletePath": 154,
    "FSCTL": 154,
    "Write": 154,
    "SetDelete": 154,
    "EventID(1)": 154,
    "EventID(2)": 154,
    "EventID(4)": 154,
    "EventID(7)": 154,
    "EventID(8)": 154,
    "EventID(9)": 154,
    "EventID(11)": 154,
    "EventID(13)": 154,
    "EventID(14)": 154,
    "KERNEL_NETWORK_TASK_TCPIP/Datareceived.": 125,
    "Reg": 154,
    "EventID(15)": 1,
    "KERNEL_NETWORK_TASK_TCPIP/Dataretransmitted.": 25,
    "ProcessStop/Stop": 19,
    "KERNEL_NETWORK_TASK_TCPIP/Disconnectissued.": 8,
    "KERNEL_NETWORK_TASK_UDPIP/DatareceivedoverUDPprotocol.": 3,
    "Unseen": 1,
    "SetInformation": 3,
    "EventID(5)": 8,
    "EventID(6)": 1,
    "KERNEL_NETWORK_TASK_TCPIP/Connectionattempted.": 1,
    "PagePriorityChange": 1,
    "CpuPriorityChange": 2,
    "KERNEL_NETWORK_TASK_UDPIP/DatasentoverUDPprotocol.": 1
}

total_eventcount_benign_train_case1 =  {
   "0.0": 9053,
   "Cleanup": 287,
   "Close": 287,
   "Create": 287,
   "CreateNewFile": 273,
   "DeletePath": 272,
   "DirEnum": 286,
   "FSCTL": 273,
   "NameCreate": 284,
   "QueryInformation": 287,
   "QuerySecurity": 287,
   "Read": 287,
   "Write": 273,
   "SetDelete": 272,
   "CpuBasePriorityChange": 288,
   "ImageLoad": 288,
   "ImageUnload": 288,
   "ProcessStart/Start": 288,
   "ThreadStart/Start": 288,
   "ThreadStop/Stop": 288,
   "EventID(1)": 288,
   "EventID(2)": 288,
   "EventID(4)": 288,
   "EventID(7)": 288,
   "EventID(8)": 288,
   "EventID(9)": 288,
   "EventID(11)": 288,
   "EventID(13)": 288,
   "EventID(14)": 287,
   "KERNEL_NETWORK_TASK_TCPIP/Datasent.": 232,
   "File": 287,
   "Reg": 288,
   "Net": 243,
   "Proc": 288,
   "Thread": 288,
   "KERNEL_NETWORK_TASK_TCPIP/Datareceived.": 193,
   "KERNEL_NETWORK_TASK_TCPIP/Connectionattempted.": 3,
   "Flush": 16,
   "SetInformation": 18,
   "CpuPriorityChange": 21,
   "ProcessStop/Stop": 12,
   "EventID(5)": 24,
   "KERNEL_NETWORK_TASK_TCPIP/Dataretransmitted.": 39,
   "Unseen": 17,
   "ThreadWorkOnBehalfUpdate": 3,
   "KERNEL_NETWORK_TASK_TCPIP/Disconnectissued.": 4,
   "KERNEL_NETWORK_TASK_TCPIP/Protocolcopieddataonbehalfofuser.": 1,
   "KERNEL_NETWORK_TASK_UDPIP/DatareceivedoverUDPprotocol.": 2,
   "KERNEL_NETWORK_TASK_UDPIP/DatasentoverUDPprotocol.": 1,
   "PagePriorityChange": 1
}
total_eventcount_malware_train_case1 = {
   "0.0": 7842,
   "Cleanup": 248,
   "Close": 248,
   "Create": 248,
   "CreateNewFile": 243,
   "DeletePath": 244,
   "DirEnum": 248,
   "FSCTL": 242,
   "NameCreate": 247,
   "QueryInformation": 248,
   "QuerySecurity": 248,
   "Read": 248,
   "Write": 243,
   "SetDelete": 244,
   "CpuBasePriorityChange": 248,
   "ImageLoad": 248,
   "ImageUnload": 248,
   "ProcessStart/Start": 248,
   "ThreadStart/Start": 248,
   "ThreadStop/Stop": 248,
   "EventID(1)": 240,
   "EventID(2)": 241,
   "EventID(4)": 241,
   "EventID(7)": 241,
   "EventID(8)": 241,
   "EventID(9)": 241,
   "EventID(11)": 241,
   "EventID(13)": 241,
   "EventID(14)": 239,
   "KERNEL_NETWORK_TASK_TCPIP/Datasent.": 223,
   "KERNEL_NETWORK_TASK_TCPIP/Datareceived.": 187,
   "File": 248,
   "Reg": 241,
   "Net": 231,
   "Proc": 248,
   "Thread": 248,
   "KERNEL_NETWORK_TASK_TCPIP/Dataretransmitted.": 23,
   "ProcessStop/Stop": 9,
   "QueryEA": 1,
   "PagePriorityChange": 1,
   "CpuPriorityChange": 2,
   "KERNEL_NETWORK_TASK_TCPIP/Disconnectissued.": 3,
   "KERNEL_NETWORK_TASK_UDPIP/DatareceivedoverUDPprotocol.": 2,
   "EventID(15)": 1,
   "EventID(5)": 2,
   "KERNEL_NETWORK_TASK_UDPIP/DatasentoverUDPprotocol.": 1,
   "KERNEL_NETWORK_TASK_TCPIP/Connectionattempted.": 2
}

total_eventcount_benign = {
    "0.0": 20737,
    "Cleanup": 659,
    "Close": 659,
    "Create": 659,
    "CreateNewFile": 631,
    "DeletePath": 630,
    "DirEnum": 657,
    "FSCTL": 631,
    "NameCreate": 651,
    "QueryInformation": 659,
    "QuerySecurity": 657,
    "Read": 657,
    "Write": 631,
    "SetDelete": 630,
    "CpuBasePriorityChange": 662,
    "ImageLoad": 662,
    "ImageUnload": 662,
    "ProcessStart/Start": 662,
    "ThreadStart/Start": 662,
    "ThreadStop/Stop": 662,
    "EventID(1)": 659,
    "EventID(2)": 659,
    "EventID(4)": 659,
    "EventID(7)": 659,
    "EventID(8)": 659,
    "EventID(9)": 658,
    "EventID(11)": 659,
    "EventID(13)": 659,
    "EventID(14)": 655,
    "KERNEL_NETWORK_TASK_TCPIP/Datasent.": 503,
    "File": 659,
    "Reg": 659,
    "Net": 521,
    "Proc": 662,
    "Thread": 662,
    "KERNEL_NETWORK_TASK_TCPIP/Datareceived.": 412,
    "KERNEL_NETWORK_TASK_TCPIP/Connectionattempted.": 8,
    "Flush": 30,
    "SetInformation": 34,
    "CpuPriorityChange": 45,
    "ProcessStop/Stop": 145,
    "EventID(5)": 152,
    "KERNEL_NETWORK_TASK_TCPIP/Dataretransmitted.": 89,
    "Unseen": 30,
    "ThreadWorkOnBehalfUpdate": 9,
    "KERNEL_NETWORK_TASK_TCPIP/Disconnectissued.": 13,
    "KERNEL_NETWORK_TASK_TCPIP/Protocolcopieddataonbehalfofuser.": 1,
    "KERNEL_NETWORK_TASK_UDPIP/DatareceivedoverUDPprotocol.": 7,
    "KERNEL_NETWORK_TASK_UDPIP/DatasentoverUDPprotocol.": 1,
    "PagePriorityChange": 2,
    "IoPriorityChange": 1,
    "EventID(6)": 1
}
total_eventcount_malware = {
    "0.0": 19790,
    "Cleanup": 628,
    "Close": 628,
    "Create": 628,
    "CreateNewFile": 605,
    "DeletePath": 607,
    "DirEnum": 628,
    "FSCTL": 602,
    "NameCreate": 623,
    "QueryInformation": 628,
    "QuerySecurity": 628,
    "Read": 628,
    "Write": 605,
    "SetDelete": 607,
    "CpuBasePriorityChange": 628,
    "ImageLoad": 628,
    "ImageUnload": 628,
    "ProcessStart/Start": 628,
    "ThreadStart/Start": 628,
    "ThreadStop/Stop": 628,
    "EventID(1)": 605,
    "EventID(2)": 606,
    "EventID(4)": 606,
    "EventID(7)": 606,
    "EventID(8)": 606,
    "EventID(9)": 606,
    "EventID(11)": 606,
    "EventID(13)": 606,
    "EventID(14)": 604,
    "KERNEL_NETWORK_TASK_TCPIP/Datasent.": 571,
    "KERNEL_NETWORK_TASK_TCPIP/Datareceived.": 481,
    "File": 628,
    "Reg": 606,
    "Net": 585,
    "Proc": 628,
    "Thread": 628,
    "KERNEL_NETWORK_TASK_TCPIP/Dataretransmitted.": 93,
    "ProcessStop/Stop": 65,
    "QueryEA": 1,
    "PagePriorityChange": 4,
    "CpuPriorityChange": 14,
    "KERNEL_NETWORK_TASK_TCPIP/Disconnectissued.": 13,
    "KERNEL_NETWORK_TASK_UDPIP/DatareceivedoverUDPprotocol.": 14,
    "EventID(15)": 1,
    "EventID(5)": 35,
    "KERNEL_NETWORK_TASK_UDPIP/DatasentoverUDPprotocol.": 13,
    "KERNEL_NETWORK_TASK_TCPIP/Connectionattempted.": 6,
    "SetInformation": 3,
    "EventID(6)": 2,
    "Unseen": 3
}

malware_event_count_dict_train_case1 = {
 " Cleanup ": 198981,
 " Close ": 194554,
 " Create ": 245758,
 " CreateNewFile ": 553,
 " DeletePath ": 518,
 " DirEnum ": 31657,
 " DirNotify ": 0,
 " Flush ": 0,
 " FSCTL ": 2441,
 " NameCreate ": 12955,
 " NameDelete ": 0,
 " OperationEnd ": 0,
 " QueryInformation ": 173064,
 " QueryEA ": 10,
 " QuerySecurity ": 2248,
 " Read ": 59564,
 " Write ": 3531,
 " SetDelete ": 518,
 " SetInformation ": 0,
 " PagePriorityChange ": 2,
 " IoPriorityChange ": 0,
 " CpuBasePriorityChange ": 249,
 " CpuPriorityChange ": 8,
 " ImageLoad ": 19928,
 " ImageUnload ": 2360,
 " ProcessStop/Stop ": 14,
 " ProcessStart/Start ": 262,
 " ProcessFreeze/Start ": 0,
 " ThreadStart/Start ": 4613,
 " ThreadStop/Stop ": 2782,
 " ThreadWorkOnBehalfUpdate ": 0,
 " JobStart/Start ": 0,
 " JobTerminate/Stop ": 0,
 " Rename ": 0,
 " Renamepath ": 0,
 " Thisgroupofeventstrackstheperformanceofflushinghives ": 0,
 " EventID(1) ": 52817,
 " EventID(2) ": 501421,
 " EventID(3) ": 0,
 " EventID(4) ": 750009,
 " EventID(5) ": 5,
 " EventID(6) ": 0,
 " EventID(7) ": 334043,
 " EventID(8) ": 101112,
 " EventID(9) ": 91366,
 " EventID(10) ": 0,
 " EventID(11) ": 11682,
 " EventID(13) ": 320332,
 " EventID(14) ": 649,
 " EventID(15) ": 81,
 " KERNEL_NETWORK_TASK_TCPIP/Datasent. ": 1175,
 " KERNEL_NETWORK_TASK_TCPIP/Datareceived. ": 342,
 " KERNEL_NETWORK_TASK_TCPIP/Connectionattempted. ": 2,
 " KERNEL_NETWORK_TASK_TCPIP/Disconnectissued. ": 3,
 " KERNEL_NETWORK_TASK_TCPIP/Dataretransmitted. ": 29,
 " KERNEL_NETWORK_TASK_TCPIP/connectionaccepted. ": 0,
 " KERNEL_NETWORK_TASK_TCPIP/Protocolcopieddataonbehalfofuser. ": 0,
 " KERNEL_NETWORK_TASK_UDPIP/DatareceivedoverUDPprotocol. ": 2,
 " KERNEL_NETWORK_TASK_UDPIP/DatasentoverUDPprotocol. ": 1,
 " Unseen ": 0,

}


benign_event_count_dict_train_case1 = {
" Cleanup ": 243712,
" Close ": 237887,
" Create ": 300673,
" CreateNewFile ": 1003,
" DeletePath ": 555,
" DirEnum ": 41460,
" DirNotify ": 0,
" Flush ": 16,
" FSCTL ": 3387,
" NameCreate ": 16750,
" NameDelete ": 0,
" OperationEnd ": 0,
" QueryInformation ": 216669,
" QueryEA ": 0,
" QuerySecurity ": 2802,
" Read ": 61677,
" Write ": 3158,
" SetDelete ": 555,
" SetInformation ": 127,
" PagePriorityChange ": 300,
" IoPriorityChange ": 0,
" CpuBasePriorityChange ": 313,
" CpuPriorityChange ": 322,
" ImageLoad ": 26261,
" ImageUnload ": 4292,
" ProcessStop/Stop ": 20,
" ProcessStart/Start ": 319,
" ProcessFreeze/Start ": 0,
" ThreadStart/Start ": 6051,
" ThreadStop/Stop ": 3810,
" ThreadWorkOnBehalfUpdate ": 6982,
" JobStart/Start ": 0,
" JobTerminate/Stop ": 0,
" Rename ": 0,
" Renamepath ": 0,
" Thisgroupofeventstrackstheperformanceofflushinghives ": 0,
" EventID(1) ": 68096,
" EventID(2) ": 661718,
" EventID(3) ": 0,
" EventID(4) ": 969494,
" EventID(5) ": 527,
" EventID(6) ": 0,
" EventID(7) ": 458279,
" EventID(8) ": 126168,
" EventID(9) ": 112712,
" EventID(10) ": 0,
" EventID(11) ": 14942,
" EventID(13) ": 411473,
" EventID(14) ": 634,
" EventID(15) ": 0,
" KERNEL_NETWORK_TASK_TCPIP/Datasent. ": 1401,
" KERNEL_NETWORK_TASK_TCPIP/Datareceived. ": 664,
" KERNEL_NETWORK_TASK_TCPIP/Connectionattempted. ": 3,
" KERNEL_NETWORK_TASK_TCPIP/Disconnectissued. ": 4,
" KERNEL_NETWORK_TASK_TCPIP/Dataretransmitted. ": 46,
" KERNEL_NETWORK_TASK_TCPIP/connectionaccepted. ": 0,
" KERNEL_NETWORK_TASK_TCPIP/Protocolcopieddataonbehalfofuser. ": 1,
" KERNEL_NETWORK_TASK_UDPIP/DatareceivedoverUDPprotocol. ": 2,
" KERNEL_NETWORK_TASK_UDPIP/DatasentoverUDPprotocol. ": 1,
" Unseen ": 96,

}

malware_event_count_dict_train_case2 = {
 " Cleanup ": 495144,
 " Close ": 483931,
 " Create ": 606712,
 " CreateNewFile ": 1465,
 " DeletePath ": 1336,
 " DirEnum ": 85501,
 " DirNotify ": 0,
 " Flush ": 0,
 " FSCTL ": 6119,
 " NameCreate ": 33480,
 " NameDelete ": 0,
 " OperationEnd ": 0,
 " QueryInformation ": 429478,
 " QueryEA ": 10,
 " QuerySecurity ": 6200,
 " Read ": 170465,
 " Write ": 10041,
 " SetDelete ": 1336,
 " SetInformation ": 3,
 " PagePriorityChange ": 860,
 " IoPriorityChange ": 0,
 " CpuBasePriorityChange ": 647,
 " CpuPriorityChange ": 41,
 " ImageLoad ": 52441,
 " ImageUnload ": 9973,
 " ProcessStop/Stop ": 87,
 " ProcessStart/Start ": 679,
 " ProcessFreeze/Start ": 0,
 " ThreadStart/Start ": 12415,
 " ThreadStop/Stop ": 8073,
 " ThreadWorkOnBehalfUpdate ": 0,
 " JobStart/Start ": 0,
 " JobTerminate/Stop ": 0,
 " Rename ": 0,
 " Renamepath ": 0,
 " Thisgroupofeventstrackstheperformanceofflushinghives ": 0,
 " EventID(1) ": 135401,
 " EventID(2) ": 1276767,
 " EventID(3) ": 0,
 " EventID(4) ": 1910468,
 " EventID(5) ": 255,
 " EventID(6) ": 4,
 " EventID(7) ": 860224,
 " EventID(8) ": 256078,
 " EventID(9) ": 232646,
 " EventID(10) ": 0,
 " EventID(11) ": 30206,
 " EventID(13) ": 818613,
 " EventID(14) ": 1389,
 " EventID(15) ": 81,
 " KERNEL_NETWORK_TASK_TCPIP/Datasent. ": 3453,
 " KERNEL_NETWORK_TASK_TCPIP/Datareceived. ": 1047,
 " KERNEL_NETWORK_TASK_TCPIP/Connectionattempted. ": 6,
 " KERNEL_NETWORK_TASK_TCPIP/Disconnectissued. ": 13,
 " KERNEL_NETWORK_TASK_TCPIP/Dataretransmitted. ": 109,
 " KERNEL_NETWORK_TASK_TCPIP/connectionaccepted. ": 0,
 " KERNEL_NETWORK_TASK_TCPIP/Protocolcopieddataonbehalfofuser. ": 0,
 " KERNEL_NETWORK_TASK_UDPIP/DatareceivedoverUDPprotocol. ": 17,
 " KERNEL_NETWORK_TASK_UDPIP/DatasentoverUDPprotocol. ": 26,
 " Unseen ": 3,

}

benign_event_count_dict_train_case2 = {
" Cleanup ": 542129,
" Close ": 527448,
" Create ": 669153,
" CreateNewFile ": 2069,
" DeletePath ": 1285,
" DirEnum ": 89911,
" DirNotify ": 0,
" Flush ": 61,
" FSCTL ": 7563,
" NameCreate ": 32237,
" NameDelete ": 0,
" OperationEnd ": 0,
" QueryInformation ": 479143,
" QueryEA ": 0,
" QuerySecurity ": 7388,
" Read ": 148780,
" Write ": 34761,
" SetDelete ": 1285,
" SetInformation ": 208,
" PagePriorityChange ": 306,
" IoPriorityChange ": 6,
" CpuBasePriorityChange ": 768,
" CpuPriorityChange ": 581,
" ImageLoad ": 59560,
" ImageUnload ": 19405,
" ProcessStop/Stop ": 176,
" ProcessStart/Start ": 727,
" ProcessFreeze/Start ": 0,
" ThreadStart/Start ": 13415,
" ThreadStop/Stop ": 9457,
" ThreadWorkOnBehalfUpdate ": 7318,
" JobStart/Start ": 0,
" JobTerminate/Stop ": 0,
" Rename ": 0,
" Renamepath ": 0,
" Thisgroupofeventstrackstheperformanceofflushinghives ": 0,
" EventID(1) ": 157110,
" EventID(2) ": 1548106,
" EventID(3) ": 0,
" EventID(4) ": 2255360,
" EventID(5) ": 4802,
" EventID(6) ": 4,
" EventID(7) ": 1053652,
" EventID(8) ": 293278,
" EventID(9) ": 258765,
" EventID(10) ": 0,
" EventID(11) ": 37182,
" EventID(13) ": 969206,
" EventID(14) ": 1410,
" EventID(15) ": 0,
" KERNEL_NETWORK_TASK_TCPIP/Datasent. ": 2881,
" KERNEL_NETWORK_TASK_TCPIP/Datareceived. ": 1318,
" KERNEL_NETWORK_TASK_TCPIP/Connectionattempted. ": 8,
" KERNEL_NETWORK_TASK_TCPIP/Disconnectissued. ": 14,
" KERNEL_NETWORK_TASK_TCPIP/Dataretransmitted. ": 104,
" KERNEL_NETWORK_TASK_TCPIP/connectionaccepted. ": 0,
" KERNEL_NETWORK_TASK_TCPIP/Protocolcopieddataonbehalfofuser. ": 1,
" KERNEL_NETWORK_TASK_UDPIP/DatareceivedoverUDPprotocol. ": 8,
" KERNEL_NETWORK_TASK_UDPIP/DatasentoverUDPprotocol. ": 1,
" Unseen ": 166,

}

malware_event_count_dict_test_case2 = {
 " Cleanup ": 138919,
 " Close ": 136382,
 " Create ": 168338,
 " CreateNewFile ": 413,
 " DeletePath ": 363,
 " DirEnum ": 36425,
 " DirNotify ": 0,
 " Flush ": 0,
 " FSCTL ": 1646,
 " NameCreate ": 10301,
 " NameDelete ": 0,
 " OperationEnd ": 0,
 " QueryInformation ": 117093,
 " QueryEA ": 0,
 " QuerySecurity ": 1530,
 " Read ": 44830,
 " Write ": 1599,
 " SetDelete ": 363,
 " SetInformation ": 3,
 " PagePriorityChange ": 258,
 " IoPriorityChange ": 0,
 " CpuBasePriorityChange ": 167,
 " CpuPriorityChange ": 8,
 " ImageLoad ": 13239,
 " ImageUnload ": 2733,
 " ProcessStop/Stop ": 27,
 " ProcessStart/Start ": 176,
 " ProcessFreeze/Start ": 0,
 " ThreadStart/Start ": 3040,
 " ThreadStop/Stop ": 1977,
 " ThreadWorkOnBehalfUpdate ": 0,
 " JobStart/Start ": 0,
 " JobTerminate/Stop ": 0,
 " Rename ": 0,
 " Renamepath ": 0,
 " Thisgroupofeventstrackstheperformanceofflushinghives ": 0,
 " EventID(1) ": 37021,
 " EventID(2) ": 337519,
 " EventID(3) ": 0,
 " EventID(4) ": 510493,
 " EventID(5) ": 36,
 " EventID(6) ": 2,
 " EventID(7) ": 224150,
 " EventID(8) ": 67761,
 " EventID(9) ": 60925,
 " EventID(10) ": 0,
 " EventID(11) ": 8257,
 " EventID(13) ": 218439,
 " EventID(14) ": 489,
 " EventID(15) ": 81,
 " KERNEL_NETWORK_TASK_TCPIP/Datasent. ": 905,
 " KERNEL_NETWORK_TASK_TCPIP/Datareceived. ": 265,
 " KERNEL_NETWORK_TASK_TCPIP/Connectionattempted. ": 1,
 " KERNEL_NETWORK_TASK_TCPIP/Disconnectissued. ": 8,
 " KERNEL_NETWORK_TASK_TCPIP/Dataretransmitted. ": 28,
 " KERNEL_NETWORK_TASK_TCPIP/connectionaccepted. ": 0,
 " KERNEL_NETWORK_TASK_TCPIP/Protocolcopieddataonbehalfofuser. ": 0,
 " KERNEL_NETWORK_TASK_UDPIP/DatareceivedoverUDPprotocol. ": 4,
 " KERNEL_NETWORK_TASK_UDPIP/DatasentoverUDPprotocol. ": 2,
 " Unseen ": 1,

}

benign_event_count_dict_test_case2 = {
" Cleanup ": 139265,
" Close ": 135888,
" Create ": 172992,
" CreateNewFile ": 669,
" DeletePath ": 324,
" DirEnum ": 23496,
" DirNotify ": 0,
" Flush ": 27,
" FSCTL ": 2133,
" NameCreate ": 10547,
" NameDelete ": 0,
" OperationEnd ": 0,
" QueryInformation ": 121811,
" QueryEA ": 0,
" QuerySecurity ": 1846,
" Read ": 40157,
" Write ": 6025,
" SetDelete ": 324,
" SetInformation ": 127,
" PagePriorityChange ": 0,
" IoPriorityChange ": 0,
" CpuBasePriorityChange ": 178,
" CpuPriorityChange ": 693,
" ImageLoad ": 17530,
" ImageUnload ": 6398,
" ProcessStop/Stop ": 57,
" ProcessStart/Start ": 208,
" ProcessFreeze/Start ": 0,
" ThreadStart/Start ": 3725,
" ThreadStop/Stop ": 2593,
" ThreadWorkOnBehalfUpdate ": 1576,
" JobStart/Start ": 0,
" JobTerminate/Stop ": 0,
" Rename ": 0,
" Renamepath ": 0,
" Thisgroupofeventstrackstheperformanceofflushinghives ": 0,
" EventID(1) ": 40279,
" EventID(2) ": 422275,
" EventID(3) ": 0,
" EventID(4) ": 599667,
" EventID(5) ": 486,
" EventID(6) ": 0,
" EventID(7) ": 290913,
" EventID(8) ": 77133,
" EventID(9) ": 67542,
" EventID(10) ": 0,
" EventID(11) ": 10634,
" EventID(13) ": 256906,
" EventID(14) ": 381,
" EventID(15) ": 0,
" KERNEL_NETWORK_TASK_TCPIP/Datasent. ": 859,
" KERNEL_NETWORK_TASK_TCPIP/Datareceived. ": 362,
" KERNEL_NETWORK_TASK_TCPIP/Connectionattempted. ": 2,
" KERNEL_NETWORK_TASK_TCPIP/Disconnectissued. ": 2,
" KERNEL_NETWORK_TASK_TCPIP/Dataretransmitted. ": 35,
" KERNEL_NETWORK_TASK_TCPIP/connectionaccepted. ": 0,
" KERNEL_NETWORK_TASK_TCPIP/Protocolcopieddataonbehalfofuser. ": 0,
" KERNEL_NETWORK_TASK_UDPIP/DatareceivedoverUDPprotocol. ": 1,
" KERNEL_NETWORK_TASK_UDPIP/DatasentoverUDPprotocol. ": 1,
" Unseen ": 79,

}

malware_event_count_dict_test_case1 = {
 " Cleanup ": 48535,
 " Close ": 47157,
 " Create ": 59329,
 " CreateNewFile ": 140,
 " DeletePath ": 128,
 " DirEnum ": 6797,
 " DirNotify ": 0,
 " Flush ": 0,
 " FSCTL ": 600,
 " NameCreate ": 3847,
 " NameDelete ": 0,
 " OperationEnd ": 0,
 " QueryInformation ": 43834,
 " QueryEA ": 0,
 " QuerySecurity ": 502,
 " Read ": 15830,
 " Write ": 594,
 " SetDelete ": 128,
 " SetInformation ": 0,
 " PagePriorityChange ": 0,
 " IoPriorityChange ": 0,
 " CpuBasePriorityChange ": 62,
 " CpuPriorityChange ": 0,
 " ImageLoad ": 4823,
 " ImageUnload ": 482,
 " ProcessStop/Stop ": 2,
 " ProcessStart/Start ": 64,
 " ProcessFreeze/Start ": 0,
 " ThreadStart/Start ": 1134,
 " ThreadStop/Stop ": 679,
 " ThreadWorkOnBehalfUpdate ": 0,
 " JobStart/Start ": 0,
 " JobTerminate/Stop ": 0,
 " Rename ": 0,
 " Renamepath ": 0,
 " Thisgroupofeventstrackstheperformanceofflushinghives ": 0,
 " EventID(1) ": 13419,
 " EventID(2) ": 124844,
 " EventID(3) ": 0,
 " EventID(4) ": 188059,
 " EventID(5) ": 0,
 " EventID(6) ": 0,
 " EventID(7) ": 82298,
 " EventID(8) ": 25816,
 " EventID(9) ": 23233,
 " EventID(10) ": 0,
 " EventID(11) ": 2941,
 " EventID(13) ": 80513,
 " EventID(14) ": 290,
 " EventID(15) ": 81,
 " KERNEL_NETWORK_TASK_TCPIP/Datasent. ": 355,
 " KERNEL_NETWORK_TASK_TCPIP/Datareceived. ": 97,
 " KERNEL_NETWORK_TASK_TCPIP/Connectionattempted. ": 0,
 " KERNEL_NETWORK_TASK_TCPIP/Disconnectissued. ": 3,
 " KERNEL_NETWORK_TASK_TCPIP/Dataretransmitted. ": 8,
 " KERNEL_NETWORK_TASK_TCPIP/connectionaccepted. ": 0,
 " KERNEL_NETWORK_TASK_TCPIP/Protocolcopieddataonbehalfofuser. ": 0,
 " KERNEL_NETWORK_TASK_UDPIP/DatareceivedoverUDPprotocol. ": 1,
 " KERNEL_NETWORK_TASK_UDPIP/DatasentoverUDPprotocol. ": 0,
 " Unseen ": 0,

}

benign_event_count_dict_test_case1 = {
" Cleanup ": 63712,
" Close ": 62158,
" Create ": 79751,
" CreateNewFile ": 443,
" DeletePath ": 142,
" DirEnum ": 10541,
" DirNotify ": 0,
" Flush ": 11,
" FSCTL ": 1093,
" NameCreate ": 4941,
" NameDelete ": 0,
" OperationEnd ": 0,
" QueryInformation ": 56970,
" QueryEA ": 0,
" QuerySecurity ": 899,
" Read ": 18931,
" Write ": 4401,
" SetDelete ": 142,
" SetInformation ": 109,
" PagePriorityChange ": 0,
" IoPriorityChange ": 0,
" CpuBasePriorityChange ": 77,
" CpuPriorityChange ": 574,
" ImageLoad ": 9114,
" ImageUnload ": 2956,
" ProcessStop/Stop ": 20,
" ProcessStart/Start ": 102,
" ProcessFreeze/Start ": 0,
" ThreadStart/Start ": 1911,
" ThreadStop/Stop ": 1265,
" ThreadWorkOnBehalfUpdate ": 1556,
" JobStart/Start ": 0,
" JobTerminate/Stop ": 0,
" Rename ": 0,
" Renamepath ": 0,
" Thisgroupofeventstrackstheperformanceofflushinghives ": 0,
" EventID(1) ": 18768,
" EventID(2) ": 203015,
" EventID(3) ": 0,
" EventID(4) ": 283989,
" EventID(5) ": 116,
" EventID(6) ": 0,
" EventID(7) ": 144776,
" EventID(8) ": 35072,
" EventID(9) ": 30324,
" EventID(10) ": 0,
" EventID(11) ": 4513,
" EventID(13) ": 119150,
" EventID(14) ": 181,
" EventID(15) ": 0,
" KERNEL_NETWORK_TASK_TCPIP/Datasent. ": 435,
" KERNEL_NETWORK_TASK_TCPIP/Datareceived. ": 213,
" KERNEL_NETWORK_TASK_TCPIP/Connectionattempted. ": 2,
" KERNEL_NETWORK_TASK_TCPIP/Disconnectissued. ": 1,
" KERNEL_NETWORK_TASK_TCPIP/Dataretransmitted. ": 22,
" KERNEL_NETWORK_TASK_TCPIP/connectionaccepted. ": 0,
" KERNEL_NETWORK_TASK_TCPIP/Protocolcopieddataonbehalfofuser. ": 0,
" KERNEL_NETWORK_TASK_UDPIP/DatareceivedoverUDPprotocol. ": 0,
" KERNEL_NETWORK_TASK_UDPIP/DatasentoverUDPprotocol. ": 0,
" Unseen ": 56,

}

#plot_events_with_higher_variance(total_eventcount_benign, total_eventcount_malware, 'Benign', 'Malware')
#plot_events_with_higher_variance(total_eventcount_benign, total_eventcount_malware, 'Benign', 'Malware')

#malware_benign_event_total_count_plot(malware_event_count_dict_train_case1,benign_event_count_dict_train_case1)

plot_4events(malware_event_count_dict_train_case2,benign_event_count_dict_train_case2,malware_event_count_dict_test_case2,benign_event_count_dict_test_case2)