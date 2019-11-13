import PySimpleGUI as sg
import MyData as md
import MyNN as my_nn
import datetime
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

matplotlib.use('TkAgg')  # draw inside main_window

MAX_HIDDEN = 5
neurons = ['1024', '512', '256', '128', '64', '32', '16', '10']

input_layout = [[sg.Text('Input size', size=(10, 1)), sg.Input('28', size=(3, 1), key='INPUT_WIDTH', tooltip='width'),
                 sg.Text('X'),
                 sg.Input('28', size=(3, 1), key='INPUT_HEIGHT', tooltip='height')],
                [sg.Checkbox('CONV 2D', size=(30,1), key='INPUT_CONV2D', tooltip='64 filters 3x3, polling 2x2')],
                [sg.Text('Neurons', size=(10, 1)), sg.Combo(values=neurons, default_value='128', key='INPUT_NEURONS'),
                 sg.Text('Activation'),
                 sg.Combo(['sigmoid', 'tanh', 'relu'], default_value='sigmoid', key='INPUT_ACTIV')]
                ]

train_layout = \
    [[sg.Frame(title='Train', layout=[
        [sg.Text('Batch size', size=(14, 1)), sg.Input('60', size=(5, 1), key='BATCH_SIZE')],
        [sg.Text('Epoch', size=(14, 1)), sg.Input('10', size=(5, 1), key='EPOCH')],
        [sg.Text('Learning rate', size=(14, 1)), sg.Input('0.01', size=(5, 1), key='LEARNING_RATE')],
        [sg.Checkbox('Early stop', size=(11, 1), key='EARLY_STOP'),
         sg.Input('2', size=(5, 1), key='EARLY_STOP_VALUE')],
        [sg.Checkbox('Drop out', size=(11, 1), key='DROP_OUT'),
         sg.Input('25', size=(5, 1), key='DROP_OUT_VALUE'), sg.Text('%')],
        [sg.Button('Train', size=(22, 1), key='TRAIN_BTN')]
    ])
      ]]

source_layout = [[sg.Radio('MNIST', group_id=2, default=True, key='RADIO_MNIST')
                  ]]

data_layout = \
    [[sg.Frame(title='Data', layout=[
        [sg.Radio('Training', group_id=1, key='RADIO_TRAIN'),
         sg.Radio('Testing', group_id=1, default=True, key='RADIO_TEST')],
        [sg.Frame('Source', layout=source_layout)],
        [sg.Button('Test', size=(20, 1))]])
      ]]

G_SIZE = (400, 250)
images_layout = \
    [[sg.Frame(title='Graph', layout=[
        [sg.Graph(canvas_size=G_SIZE, graph_bottom_left=(0, 0), graph_top_right=G_SIZE, key='GRAPH',
                  background_color='white')]])],
     [sg.Button('Plot graph', key='PLOT_BTN')]
     ]

hidden_count = [[sg.Text('Count', size=(10, 1)), sg.Spin([1, 2, 3, 4, 5], initial_value=1, key='HID_COUNT',
                                                         enable_events=True)]]

hidden_layout1 = [[sg.Text('Neurons', size=(10, 1)), sg.Combo(values=neurons, default_value='64', key='HID_NEURONS1'),
                   sg.Text('Activation'),
                   sg.Combo(['sigmoid', 'tanh', 'relu'], default_value='sigmoid', key='HID_ACTIV1')]]

hidden_layout2 = [[sg.Text('Neurons', size=(10, 1)), sg.Combo(values=neurons, default_value='32', key='HID_NEURONS2'),
                   sg.Text('Activation'),
                   sg.Combo(['sigmoid', 'tanh', 'relu'], default_value='sigmoid', key='HID_ACTIV2')]]

hidden_layout3 = [[sg.Text('Neurons', size=(10, 1)), sg.Combo(values=neurons, default_value='32', key='HID_NEURONS3'),
                   sg.Text('Activation'),
                   sg.Combo(['sigmoid', 'tanh', 'relu'], default_value='sigmoid', key='HID_ACTIV3')]]

hidden_layout4 = [[sg.Text('Neurons', size=(10, 1)), sg.Combo(values=neurons, default_value='32', key='HID_NEURONS4'),
                   sg.Text('Activation'),
                   sg.Combo(['sigmoid', 'tanh', 'relu'], default_value='sigmoid', key='HID_ACTIV4')]]

hidden_layout5 = [[sg.Text('Neurons', size=(10, 1)), sg.Combo(values=neurons, default_value='32', key='HID_NEURONS5'),
                   sg.Text('Activation'),
                   sg.Combo(['sigmoid', 'tanh', 'relu'], default_value='sigmoid', key='HID_ACTIV5')]]

hidden_layout_out = [[sg.Text('Neurons', size=(10, 1)), sg.Combo(values=neurons, default_value='10',
                                                                 key='HID_NEURONS_OUT'),
                      sg.Text('Activation'),
                      sg.Combo(['sigmoid', 'softmax'], default_value='softmax', key='HID_ACTIV_OUT')]]

config_layout = [[sg.Frame('Input/hidden', layout=input_layout, pad=(0, 4))],
                 [sg.Frame('Hidden layers', layout=hidden_count, key='HC', pad=(0, 4))],
                 [sg.Frame('Hidden 1', layout=hidden_layout1, key='H1', pad=(0, 4))],
                 [sg.Frame('Hidden 2', layout=hidden_layout2, key='H2', pad=(0, 4), visible=False)],
                 [sg.Frame('Hidden 3', layout=hidden_layout3, key='H3', pad=(0, 4), visible=False)],
                 [sg.Frame('Hidden 4', layout=hidden_layout4, key='H4', pad=(0, 4), visible=False)],
                 [sg.Frame('Hidden 5', layout=hidden_layout5, key='H5', pad=(0, 4), visible=False)],
                 [sg.Frame('Output', layout=hidden_layout_out, key='HOUT', pad=(0, 4), visible=True)],
                 [sg.Button('Create', key='CREATE_BTN', size=(35, 1))]
                 ]
col_bottom = []
main_layout = [
    [sg.Column(layout=config_layout, background_color='white'),
     sg.Column(layout=[[sg.Column(layout=train_layout), sg.Column(layout=data_layout)],
                       [sg.Column(layout=images_layout)]])],
    [sg.Column(layout=col_bottom)]
]


class History:
    def __init__(self):
        self.accuracy = []
        self.val_accuracy = []


# globals
main_window = None
data = None
nn = None
history = History()
fig = None
fig_agg = None
ax = None


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


# draw via plot
def plot_draw(epoch=None, logs=None):
    global history
    if logs is not None:
        if epoch == 0:
            history.accuracy.clear()
            history.val_accuracy.clear()
        history.accuracy.append(float(logs['accuracy']))
        history.val_accuracy.append(float(logs['val_accuracy']))
    if ax is not None:
        l1, = ax.plot(history.accuracy, color='blue')
        l2, = ax.plot(history.val_accuracy, color='orange')
        ax.legend([l1, l2], ['accuracy', 'validate'])
        fig_agg.draw()
        main_window.Refresh()


# load data
def create_data():
    global main_window
    if bool(main_window['RADIO_MNIST'].get()):
        src = 1
    else:
        src = 0
    dt = md.Data(src)
    dt.load()
    return dt


# read interface and create net
def create_nn():
    global main_window
    n_n = my_nn.Neural_net()

    iw = int(main_window['INPUT_WIDTH'].get())
    ih = int(main_window['INPUT_HEIGHT'].get())
    inn = int(main_window['INPUT_NEURONS'].get())
    iat = main_window['INPUT_ACTIV'].get()
    c2d = bool(main_window['INPUT_CONV2D'].get())
    # input
    n_n.add_input(inn, iat, iw, ih, c2d)
    # hidden's
    for i in range(1, MAX_HIDDEN + 1):
        hc = int(main_window['HID_COUNT'].get())
        if i <= hc:
            hn = int(main_window['HID_NEURONS' + str(i)].get())
            ht = main_window['HID_ACTIV' + str(i)].get()
            n_n.add_hidden(hn, ht)
    # out
    hn = int(main_window['HID_NEURONS_OUT'].get())
    ht = main_window['HID_ACTIV_OUT'].get()
    n_n.add_hidden(hn, ht)
    return n_n


# main loop
def main():
    global main_window, data, nn, fig_agg, ax, fig
    first = True
    main_window = sg.Window('Simple neural network constructor', main_layout, resizable=True, background_color='gray',
                            button_color=['black', 'white'], icon='logo.ico')

    # Event Loop to process "events"
    while True:
        event, values = main_window.Read(timeout=1000)
        if event in (None, 'Cancel'):
            break
        #elif event == '__TIMEOUT__':
            #    print(datetime.datetime.now())
            #if first:
            #    main_window['H2'].update(visible=False)
            #first = False
        elif event == 'HID_COUNT':
            hc = int(main_window['HID_COUNT'].get())
            for i in range(1, MAX_HIDDEN + 1):
                if i <= hc:
                    main_window['H' + str(i)].update(visible=True)
                else:
                    main_window['H' + str(i)].update(visible=False)
            main_window.Refresh()
        elif event == 'PLOT_BTN':
            plot_draw()
        elif event == 'CREATE_BTN':
            if data is None:
                data = create_data()
            nn = create_nn()
        elif event == 'TRAIN_BTN':
            if data is None:
                data = create_data()

            # prepare draw
            if fig is None:
                fig = Figure(figsize=(5, 4))  # inches
                ax = fig.add_subplot(111)
                ax.set_title('model accuracy')
                plt.legend(['training', 'validation'], loc='best')
                ax.set_ylabel('accuracy')
                ax.set_xlabel('epoch')
                ax.grid(b=True)
                canvas = main_window['GRAPH'].TKCanvas
                fig_agg = draw_figure(canvas, fig)
            # clear before each training
            ax.cla()
            # axis on
            ax.grid(b=True)

            #if nn is None:
            nn = create_nn()
            bs = int(main_window['BATCH_SIZE'].get())
            ep = int(main_window['EPOCH'].get())
            lr = float(main_window['LEARNING_RATE'].get())
            if bool(main_window['EARLY_STOP'].get()):
                es = float(main_window['EARLY_STOP_VALUE'].get())
            else:
                es = float(0)
            if bool(main_window['DROP_OUT'].get()):
                do = float(main_window['DROP_OUT_VALUE'].get())
            else:
                do = float(0)

            if nn.conv:
                nn.fit(data.x_train2d, data.y_train, data.x_test2d, data.y_test, bs, ep, lr, es, do, plot_draw)
            else:
                nn.fit(data.x_train, data.y_train, data.x_test, data.y_test, bs, ep, lr, es, do, plot_draw)

    main_window.Close()


if __name__ == '__main__':
    main()
