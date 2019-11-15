import PySimpleGUI as sg
import MyData as md
import MyNN as my_nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

matplotlib.use('TkAgg')  # draw inside main_window

MAX_HIDDEN = 5
neurons = ['1024', '512', '256', '128', '64', '32', '16', '10', '5', '2', '1']

i_layout= [[sg.Text('Input size', size=(10, 1)), sg.Input('28', size=(3, 1), key='INPUT_WIDTH', tooltip='width'),
                 sg.Text('X'),
                 sg.Input('28', size=(3, 1), key='INPUT_HEIGHT', tooltip='height')]]
input_layout = [[sg.Frame('', layout=i_layout, key='INPUT_SIZE')],
                [sg.Checkbox('CONV 2D', size=(30, 1), key='INPUT_CONV2D', tooltip='64 filters 3x3, polling 2x2', enable_events=True),
                 sg.Checkbox('Text preprocess', size=(30,1), default=True, disabled=True, key='INPUT_TEXT_PP', tooltip='text preporcess with google/tf2-preview/gnews-swivel-20dim/1', visible=False)],
                [sg.Text('Neurons', size=(10, 1)), sg.Combo(values=neurons, default_value='128', key='INPUT_NEURONS'),
                 sg.Text('Activation'),
                 sg.Combo(['sigmoid', 'tanh', 'relu'], default_value='sigmoid', key='INPUT_ACTIV', readonly=True)]
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
        [sg.Button('Train', size=(10, 1), key='TRAIN_BTN', tooltip='start training'),
         sg.Button('Show graph', size=(10, 1), key='PLOT_BTN', tooltip='training graph')]
    ])
      ]]

source_layout = [[sg.Combo(['cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'imdb', 'reuters'],
                           default_value='mnist', key='DATABASE', readonly=True, enable_events=True)
                  ]]

data_layout = \
    [[sg.Frame(title='Data', layout=[
        [sg.Radio('Training', group_id=1, key='RADIO_TRAIN'),
         sg.Radio('Testing', group_id=1, default=True, key='RADIO_TEST')],
        [sg.Frame('Source', layout=source_layout)],
        [sg.Button('Show', size=(8, 1), key='SHOW_DB'), sg.Button('Test', size=(8, 1), key='TEST_DB')]])
      ]]

G_SIZE = (400, 250)
images_layout = \
    [[sg.Frame(title='Graph', layout=[
        [sg.Graph(canvas_size=G_SIZE, graph_bottom_left=(0, 0), graph_top_right=G_SIZE, key='GRAPH',
                  background_color='white')]])]
     ]

hidden_count = [[sg.Text('Count', size=(10, 1)), sg.Spin([1, 2, 3, 4, 5], initial_value=1, key='HID_COUNT',
                                                         enable_events=True)]]

hidden_layout1 = [[sg.Text('Neurons', size=(10, 1)), sg.Combo(values=neurons, default_value='64', key='HID_NEURONS1'),
                   sg.Text('Activation'),
                   sg.Combo(['sigmoid', 'tanh', 'relu'], default_value='sigmoid', key='HID_ACTIV1', readonly=True)]]

hidden_layout2 = [[sg.Text('Neurons', size=(10, 1)), sg.Combo(values=neurons, default_value='32', key='HID_NEURONS2'),
                   sg.Text('Activation'),
                   sg.Combo(['sigmoid', 'tanh', 'relu'], default_value='sigmoid', key='HID_ACTIV2', readonly=True)]]

hidden_layout3 = [[sg.Text('Neurons', size=(10, 1)), sg.Combo(values=neurons, default_value='32', key='HID_NEURONS3'),
                   sg.Text('Activation'),
                   sg.Combo(['sigmoid', 'tanh', 'relu'], default_value='sigmoid', key='HID_ACTIV3', readonly=True)]]

hidden_layout4 = [[sg.Text('Neurons', size=(10, 1)), sg.Combo(values=neurons, default_value='32', key='HID_NEURONS4'),
                   sg.Text('Activation'),
                   sg.Combo(['sigmoid', 'tanh', 'relu'], default_value='sigmoid', key='HID_ACTIV4', readonly=True)]]

hidden_layout5 = [[sg.Text('Neurons', size=(10, 1)), sg.Combo(values=neurons, default_value='32', key='HID_NEURONS5'),
                   sg.Text('Activation'),
                   sg.Combo(['sigmoid', 'tanh', 'relu'], default_value='sigmoid', key='HID_ACTIV5', readonly=True)]]

hidden_layout_out = [[sg.Text('Neurons', size=(10, 1)), sg.Combo(values=neurons, default_value='10',
                                                                 key='HID_NEURONS_OUT'),
                      sg.Text('Activation'),
                      sg.Combo(['sigmoid', 'softmax'], default_value='softmax', key='HID_ACTIV_OUT', readonly=True)]]

config_layout = [[sg.Frame('Input/hidden', layout=input_layout, pad=(0, 4), key='CONFIG')],
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
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # force int x-axis (after plot)
        fig_agg.draw()
        main_window.Refresh()


def plot_db():
    global main_window, data
    data = create_data()
    if data.source in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']:
        plt.figure('Database '+data.source, figsize=(5, 5))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            if bool(main_window['RADIO_TRAIN'].get()):
                plt.imshow(data.x_train[i])
                plt.xlabel(data.y_train_label[i])
            else:
                plt.imshow(data.x_test[i])
                plt.xlabel(data.y_test_label[i])
            if data.class_names is not None:
                plt.xlabel(data.class_names[int(data.y_test_label[i])])
        plt.show(block=False)


def plot_clear():
    ax.cla()


# prepare draw
def plot_prepare():
    global fig, ax, fig_agg
    if fig is None:
        fig = Figure(figsize=(5, 4))  # inches
        ax = fig.add_subplot(111)
        ax.set_title('model accuracy')
        ax.set_ylabel('accuracy')
        ax.set_xlabel('epoch')
        ax.grid(b=True)
        canvas = main_window['GRAPH'].TKCanvas
        fig_agg = draw_figure(canvas, fig)


# load data
def create_data():
    global main_window
    db = main_window['DATABASE'].get()
    dt = md.Data(db)
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
    db = main_window['DATABASE'].get()
    # input
    n_n.add_input(inn, iat, iw, ih, c2d, db)
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
        elif event == 'DATABASE':
            if main_window['DATABASE'].get() in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']:  # images
                main_window['INPUT_CONV2D'].update(visible=True)
                main_window['INPUT_SIZE'].update(visible=True)
                main_window['INPUT_TEXT_PP'].update(visible=False)
                af = 'sigmoid'
                afo = 'softmax'
            else: # text
                main_window['INPUT_CONV2D'].update(visible=False)
                main_window['INPUT_SIZE'].update(visible=False)
                main_window['INPUT_TEXT_PP'].update(visible=True)
                af = 'relu'
                afo = 'sigmoid'

            out = str(md.Data.db_categories(main_window['DATABASE'].get()))
            main_window['HID_NEURONS_OUT'].update(value=out)

            main_window['INPUT_ACTIV'].update(value=af)
            for i in range(1, MAX_HIDDEN + 1):
                main_window['HID_ACTIV' + str(i)].update(values=af)
            main_window['HID_ACTIV_OUT'].update(value=afo)

            main_window.Refresh()
        elif event == 'HID_COUNT':
            hc = int(main_window['HID_COUNT'].get())
            for i in range(1, MAX_HIDDEN + 1):
                if i <= hc:
                    main_window['H' + str(i)].update(visible=True)
                else:
                    main_window['H' + str(i)].update(visible=False)
            #need recreate window for resize elements
            main_window.Refresh()
        elif event == 'INPUT_CONV2D':
            if bool(main_window['INPUT_CONV2D'].get()):
                af = 'relu'
            else:
                af = 'sigmoid'
            main_window['INPUT_ACTIV'].update(value=af)
            for i in range(1, MAX_HIDDEN + 1):
                main_window['HID_ACTIV' + str(i)].update(values=af)
        elif event == 'SHOW_DB':
            plot_db()
        elif event == 'PLOT_BTN':
            plot_prepare()
            plot_clear()
            plot_draw()
        elif event == 'CREATE_BTN':
            if data is None:
                data = create_data()
            nn = create_nn()
        elif event == 'TRAIN_BTN':
            data = create_data()

            plot_prepare()
            # clear before each training
            plot_clear()

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
            elif data.is_text:
                nn.fit(data.x_train, data.y_train, data.x_test, data.y_test, bs, ep, lr, es, do, plot_draw)
            else:
                nn.fit(data.x_train1d, data.y_train, data.x_test1d, data.y_test, bs, ep, lr, es, do, plot_draw)

    main_window.Close()


if __name__ == '__main__':
    main()
